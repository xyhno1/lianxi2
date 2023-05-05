#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from apex import amp

import os
import random
import timm
import wandb
from tqdm import trange
import multiprocessing
from functools import partial
import numpy as np
import PIL
from torchvision.transforms import InterpolationMode

from sampler import UniqueClassSempler
from proxy_anchor.utils import calc_recall_at_k
from proxy_anchor.dataset import CUBirds, SOP, Cars
from proxy_anchor.dataset.Inshop import Inshop_Dataset
from hyptorch.pmath import dist_matrix
import hyptorch.nn as hypnn

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from datetime import timedelta

# In[2]:


seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# # 超参数

# In[3]:


num = 1
path = '/data/xuyunhao/datasets'
ds = 'Inshop'
num_samples = 2
bs = 200

lr = 1e-5

t = 0.2
emb = 128
freeze = 0
ep = 500
hyp_c = 0.1
eval_ep = 'r(' + str(ep - 100) + ',' + str(ep + 10) + ',10)'

#model = 'vit_small_patch16_224'
model = 'dino_vits16'
#model = 'deit_small_distilled_patch16_224'

save_emb = False
emb_name = 'emb'
clip_r = 2.3
resize = 224
crop = 224
local_rank = 0
save_path = "/data/xuyunhao/Mixed curvature/result/{}_{}_best_epd_{}_{}_checkout.pth".format(model, ds, emb, num)
load_model = "/data/xuyunhao/Mixed curvature/result/{}_{}_best_p_{}_1_checkout.pth".format(model, ds, emb)


# load_model = "/data/xuyunhao/Mixed curvature/result/{}_{}_best_d_{}_{}_checkout.pth".format(model,ds,emb,num)


# # 庞加莱模型

# In[4]:


def _tensor_dot(x, y):
    res = torch.einsum("ij,kj->ik", (x, y))
    return res


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def _mobius_addition_batch(x, y, c):
    xy = _tensor_dot(x, y)  # B x C
    x2 = x.pow(2).sum(-1, keepdim=True)  # B x 1
    y2 = y.pow(2).sum(-1, keepdim=True)  # C x 1
    num = 1 + 2 * c * xy + c * y2.permute(1, 0)  # B x C
    num = num.unsqueeze(2) * x.unsqueeze(1)
    num = num + (1 - c * x2).unsqueeze(2) * y  # B x C x D
    denom_part1 = 1 + 2 * c * xy  # B x C
    denom_part2 = c ** 2 * x2 * y2.permute(1, 0)
    denom = denom_part1 + denom_part2
    res = num / (denom.unsqueeze(2) + 1e-5)
    return res


def _dist_matrix(x, y, c):
    sqrt_c = c ** 0.5
    return (
            2
            / sqrt_c
            * artanh(sqrt_c * torch.norm(_mobius_addition_batch(-x, y, c=c), dim=-1))
    )


def dist_matrix(x, y, c=1.0):
    c = torch.as_tensor(c).type_as(x)
    return _dist_matrix(x, y, c)


def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


def expmap0(u, *, c=1.0):
    c = torch.as_tensor(c).type_as(u)
    return _expmap0(u, c)


def _expmap0(u, c):
    sqrt_c = c ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1


def project(x, *, c=1.0):
    c = torch.as_tensor(c).type_as(x)
    return _project(x, c)


def _project(x, c):
    norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
    maxnorm = (1 - 1e-3) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


class ToPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Euclidean space
    to n-dim Poincare ball
    Also implements clipping from https://arxiv.org/pdf/2107.11472.pdf
    """

    def __init__(self, c, clip_r=None):
        super(ToPoincare, self).__init__()
        self.register_parameter("xp", None)

        self.c = c

        self.clip_r = clip_r
        self.grad_fix = lambda x: x

    def forward(self, x):
        if self.clip_r is not None:
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac = torch.minimum(
                torch.ones_like(x_norm),
                self.clip_r / x_norm
            )
            x = x * fac
        return self.grad_fix(project(expmap0(x, c=self.c), c=self.c))


# # 投影超球模型

# In[5]:


def project(x, *, c=1.0):
    c = torch.as_tensor(c).type_as(x)
    return _project(x, c)


def _project(x, c):
    norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
    maxnorm = (1 - 1e-3) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def dexp0(u, *, c=1.0):
    c = torch.as_tensor(c).type_as(u)
    return _dexp0(u, c)


def _dexp0(u, c):
    sqrt_c = c ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
    gamma_1 = torch.tan(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1


def _dist_matrix_d(x, y, c):
    xy = torch.einsum("ij,kj->ik", (x, y))  # B x C
    x2 = x.pow(2).sum(-1, keepdim=True)  # B x 1
    y2 = y.pow(2).sum(-1, keepdim=True)  # C x 1
    sqrt_c = c ** 0.5
    num1 = 2 * c * (x2 + y2.permute(1, 0) - 2 * xy) + 1e-5
    num2 = torch.mul((1 + c * x2), (1 + c * y2.permute(1, 0)))
    return (1 / sqrt_c * torch.acos(1 - num1 / num2))


def dist_matrix_d(x, y, c=1.0):
    c = torch.as_tensor(c).type_as(x)
    return _dist_matrix_d(x, y, c)


class ToProjection_hypersphere(nn.Module):
    def __init__(self, c, clip_r=None):
        super(ToProjection_hypersphere, self).__init__()
        self.register_parameter("xp", None)
        self.c = c
        self.clip_r = clip_r

    def forward(self, x):
        if self.clip_r is not None:
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac = torch.minimum(
                torch.ones_like(x_norm),
                self.clip_r / x_norm
            )
            x = x * fac
        return project(dexp0(x, c=self.c), c=self.c)


# # 损失函数

# In[6]:


def contrastive_loss_e(e0, e1, tau):
    # x0 and x1 - positive pair
    # tau - temperature
    # hyp_c - hyperbolic curvature, "0" enables sphere mode
    dist_e = lambda x, y: -torch.cdist(x, y, p=2)

    dist_e0 = dist_e(e0, e0)
    dist_e1 = dist_e(e0, e1)

    bsize = e0.shape[0]
    target = torch.arange(bsize).cuda()
    eye_mask = torch.eye(bsize).cuda() * 1e9

    logits00 = dist_e0 / tau - eye_mask
    logits01 = dist_e1 / tau

    logits = torch.cat([logits01, logits00], dim=1)
    logits -= logits.max(1, keepdim=True)[0].detach()
    loss = F.cross_entropy(logits, target)
    return loss


# In[7]:


def contrastive_loss_p(x0, x1, tau, hyp_c):
    # x0 and x1 - positive pair
    # tau - temperature
    # hyp_c - hyperbolic curvature, "0" enables sphere mode

    dist_f = lambda x, y: -dist_matrix(x, y, c=hyp_c)
    bsize = x0.shape[0]
    target = torch.arange(bsize).cuda()
    eye_mask = torch.eye(bsize).cuda() * 1e9
    logits00 = dist_f(x0, x0) / tau - eye_mask
    logits01 = dist_f(x0, x1) / tau
    logits = torch.cat([logits01, logits00], dim=1)
    logits -= logits.max(1, keepdim=True)[0].detach()
    loss = F.cross_entropy(logits, target)
    return loss


# In[8]:


def contrastive_loss_d(x0, x1, tau, hyp_c):
    # x0 and x1 - positive pair
    # tau - temperature
    # hyp_c - hyperbolic curvature, "0" enables sphere mode

    dist_f = lambda x, y: -dist_matrix_d(x, y, c=hyp_c)
    bsize = x0.shape[0]
    target = torch.arange(bsize).cuda()
    eye_mask = torch.eye(bsize).cuda() * 1e9
    logits00 = dist_f(x0, x0) / tau - eye_mask
    logits01 = dist_f(x0, x1) / tau
    logits = torch.cat([logits01, logits00], dim=1)
    logits -= logits.max(1, keepdim=True)[0].detach()
    loss = F.cross_entropy(logits, target)
    return loss


# # 模型

# In[9]:


def init_model(model=model, hyp_c=0.1, emb=128, clip_r=2.3, freeze=0):
    if model.startswith("dino"):
        body = torch.hub.load("facebookresearch/dino:main", model)
    else:
        body = timm.create_model(model, pretrained=True)
    bdim = 2048 if model == "resnet50" else 384

    Elayer = NormLayer()
    embedding_e = nn.Sequential(nn.Linear(bdim, emb), nn.BatchNorm1d(emb))

    Player = ToPoincare(
        c=hyp_c,
        clip_r=clip_r,
    )
    embedding_p = nn.Sequential(nn.Linear(bdim, emb), nn.BatchNorm1d(emb), Player)

    Dlayer = ToProjection_hypersphere(
        c=hyp_c,
        clip_r=clip_r,
    )
    embedding_d = nn.Sequential(nn.Linear(bdim, emb), nn.BatchNorm1d(emb), Dlayer)

    nn.init.constant_(embedding_e[0].bias.data, 0)
    nn.init.orthogonal_(embedding_e[0].weight.data)
    nn.init.constant_(embedding_p[0].bias.data, 0)
    nn.init.orthogonal_(embedding_p[0].weight.data)
    nn.init.constant_(embedding_d[0].bias.data, 0)
    nn.init.orthogonal_(embedding_d[0].weight.data)

    rm_head(body)
    if freeze is not None:
        freezer(body, freeze)
    model = HeadSwitch(body, embedding_e, embedding_p, embedding_d)
    model.cuda().train()
    return model


class HeadSwitch(nn.Module):
    def __init__(self, body, embedding_e, embedding_p, embedding_d):
        super(HeadSwitch, self).__init__()
        self.body = body
        self.embedding_e = embedding_e
        self.embedding_p = embedding_p
        self.embedding_d = embedding_d
        self.norm = NormLayer()

    def forward(self, x, skip_head=False):
        x = self.body(x)
        if type(x) == tuple:
            x = x[0]
        if not skip_head:
            x_e = self.embedding_e(x)
            x_p = self.embedding_p(x)
            x_d = self.embedding_d(x)
            return x_e, x_p, x_d
        else:
            x = self.norm(x)
            return x


class NormLayer(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


def freezer(model, num_block):
    def fr(m):
        for param in m.parameters():
            param.requires_grad = False

    fr(model.patch_embed)
    fr(model.pos_drop)
    for i in range(num_block):
        fr(model.blocks[i])


def rm_head(m):
    names = set(x[0] for x in m.named_children())
    target = {"head", "fc", "head_dist"}
    for x in names & target:
        m.add_module(x, nn.Identity())


# In[10]:


class MultiSample:
    def __init__(self, transform, n=2):
        self.transform = transform
        self.num = n

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num))


def evaluate(get_emb_f, ds_name, hyp_c):
    if ds_name == "CUB" or ds_name == "Cars":
        emb_head = get_emb_f(ds_type="eval")
        recall_head = get_recall(*emb_head, ds_name, hyp_c)
    elif ds_name == "SOP":
        emb_head = get_emb_f(ds_type="eval")
        recall_head = get_recall_sop(*emb_head, ds_name, hyp_c)
    else:
        emb_head_query = get_emb_f(ds_type="query")
        emb_head_gal = get_emb_f(ds_type="gallery")
        recall_head = get_recall_inshop(*emb_head_query, *emb_head_gal, hyp_c)
    return recall_head


def get_recall(e, p, d, y, ds_name, hyp_c):
    if ds_name == "CUB" or ds_name == "Cars":
        k_list = [1, 2, 4, 8, 16, 32]
    elif ds_name == "SOP":
        k_list = [1, 10, 100, 1000]

    dist_m = torch.empty(len(e), len(e), device="cuda")
    for i in range(len(e)):
        dist_m[i: i + 1] = -torch.cdist(e[i: i + 1], e, p=2) - dist_matrix_d(d[i: i + 1], d, hyp_c) - dist_matrix(
            p[i: i + 1], p, hyp_c)

    y_cur = y[dist_m.topk(1 + max(k_list), largest=True)[1][:, 1:]]
    y = y.cpu()
    y_cur = y_cur.float().cpu()
    recall = [calc_recall_at_k(y, y_cur, k) for k in k_list]
    print(recall)
    return recall


def get_recall_sop(e, p, d, y, ds_name, hyp_c):
    y_cur = torch.tensor([]).cuda().int()
    number = 1000
    k_list = [1, 10, 100, 1000]
    for i in range(len(e) // number + 1):
        if (i + 1) * number > len(e):
            e_s = e[i * number:]
            p_s = p[i * number:]
            d_s = d[i * number:]
        else:
            e_s = e[i * number: (i + 1) * number]
            p_s = p[i * number: (i + 1) * number]
            d_s = d[i * number: (i + 1) * number]
        #         import ipdb;
        #         ipdb.set_trace()
        dist = torch.empty(len(e_s), len(e), device="cuda")
        for i in range(len(e_s)):
            dist[i: i + 1] = -torch.cdist(e_s[i: i + 1], e, p=2) - dist_matrix_d(d_s[i: i + 1], d, hyp_c) - dist_matrix(
                p_s[i: i + 1], p, hyp_c)
        dist = y[dist.topk(1 + max(k_list), largest=True)[1][:, 1:]]
        y_cur = torch.cat([y_cur, dist])
    y = y.cpu()
    y_cur = y_cur.float().cpu()
    recall = [calc_recall_at_k(y, y_cur, k) for k in k_list]
    print(recall)
    return recall


def get_recall_inshop(eq, pq, dq, yq, eg, pg, dg, yg, hyp_c):
    dist_m = torch.empty(len(eq), len(eg), device="cuda")
    for i in range(len(eq)):
        dist_m[i: i + 1] = -torch.cdist(eq[i: i + 1], eg, p=2) - dist_matrix_d(dq[i: i + 1], dg, hyp_c) - dist_matrix(
            pq[i: i + 1], pg, hyp_c)

    def recall_k(cos_sim, query_T, gallery_T, k):
        m = len(cos_sim)
        match_counter = 0
        for i in range(m):
            pos_sim = cos_sim[i][gallery_T == query_T[i]]
            neg_sim = cos_sim[i][gallery_T != query_T[i]]
            thresh = torch.max(pos_sim).item()
            if torch.sum(neg_sim > thresh) < k:
                match_counter += 1
        return match_counter / m

    recall = [recall_k(dist_m, yq, yg, k) for k in [1, 10, 20, 30, 40, 50]]
    print(recall)
    return recall


def get_emb(
        model,
        ds,
        path,
        mean_std,
        resize=224,
        crop=224,
        ds_type="eval",
        world_size=1,
        skip_head=False,
):
    eval_tr = T.Compose(
        [
            T.Resize(resize, interpolation=PIL.Image.BICUBIC),
            T.CenterCrop(crop),
            T.ToTensor(),
            T.Normalize(*mean_std),
        ]
    )
    ds_eval = ds(path, ds_type, eval_tr)
    if world_size == 1:
        sampler = None
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(ds_eval)
    dl_eval = DataLoader(
        dataset=ds_eval,
        batch_size=100,
        shuffle=False,
        num_workers=multiprocessing.cpu_count() // world_size,
        pin_memory=True,
        drop_last=False,
        sampler=sampler,
    )
    model.eval()
    if skip_head == True:
        x, y = eval_dataset(model, dl_eval, skip_head)
        y = y.cuda()
        if world_size > 1:
            all_x = [torch.zeros_like(x) for _ in range(world_size)]
            all_y = [torch.zeros_like(y) for _ in range(world_size)]
            torch.distributed.all_gather(all_x, x)
            torch.distributed.all_gather(all_y, y)
            x, y = torch.cat(all_x), torch.cat(all_y)
        model.train()
        return x, y
    else:
        e, p, d, y = eval_dataset(model, dl_eval, skip_head)
        y = y.cuda()
        if world_size > 1:
            all_e = [torch.zeros_like(e) for _ in range(world_size)]
            all_p = [torch.zeros_like(p) for _ in range(world_size)]
            all_d = [torch.zeros_like(d) for _ in range(world_size)]
            all_y = [torch.zeros_like(y) for _ in range(world_size)]
            torch.distributed.all_gather(all_e, e)
            torch.distributed.all_gather(all_p, p)
            torch.distributed.all_gather(all_d, d)
            torch.distributed.all_gather(all_y, y)
            e, p, d, y = torch.cat(all_e), torch.cat(all_p), torch.cat(all_d), torch.cat(all_y)
        model.train()
        return e, p, d, y


def eval_dataset(model, dl, skip_head):
    all_x, all_xe, all_xp, all_xd, all_y = [], [], [], [], []
    for x, y in dl:
        with torch.no_grad():
            x = x.cuda(non_blocking=True)
            e, p, d = model(x, skip_head=skip_head)
            all_xe.append(e)
            all_xp.append(p)
            all_xd.append(d)
        all_y.append(y)
    return torch.cat(all_xe), torch.cat(all_xp), torch.cat(all_xd), torch.cat(all_y)


# # 主函数

# In[ ]:


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
if local_rank == 0:
    wandb.init(project="hyp_metric")

world_size = int(os.environ.get("WORLD_SIZE", 1))

if model.startswith("vit"):
    mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
else:
    mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
train_tr = T.Compose(
    [
        T.RandomResizedCrop(
            crop, scale=(0.2, 1.0), interpolation=InterpolationMode.BICUBIC
        ),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(*mean_std),
    ]
)

ds_list = {"CUB": CUBirds, "SOP": SOP, "Cars": Cars, "Inshop": Inshop_Dataset}
ds_class = ds_list[ds]
ds_train = ds_class(path, "train", train_tr)
assert len(ds_train.ys) * num_samples >= bs * world_size
sampler = UniqueClassSempler(
    ds_train.ys, num_samples, local_rank, world_size
)
dl_train = DataLoader(
    dataset=ds_train,
    sampler=sampler,
    batch_size=bs,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
)
model = init_model(model=model, hyp_c=hyp_c, emb=emb, clip_r=clip_r, freeze=freeze)
optimizer = optim.AdamW(model.parameters(), lr=lr)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(load_model), False)

loss_e = partial(contrastive_loss_e, tau=t)
loss_p = partial(contrastive_loss_p, tau=t, hyp_c=hyp_c)
loss_d = partial(contrastive_loss_d, tau=t, hyp_c=hyp_c)

get_emb_f = partial(
    get_emb,
    model=model,
    ds=ds_class,
    path=path,
    mean_std=mean_std,
    world_size=world_size,
    resize=resize,
    crop=crop,
)

eval_ep = eval(eval_ep.replace("r", "list(range").replace(")", "))"))

cudnn.benchmark = True
all_rh = []
best_rh = []
best_ep = 0
lower_cnt = 0

print("save_path:", save_path)
print("load_model:", load_model)

for ep in trange(ep):
    sampler.set_epoch(ep)
    stats_ep = []
    for x, y in dl_train:
        y = y.view(len(y) // num_samples, num_samples)
        assert (y[:, 0] == y[:, -1]).all()
        s = y[:, 0].tolist()
        assert len(set(s)) == len(s)

        x = x.cuda(non_blocking=True)
        e, p, d = model(x)
        e = e.view(len(x) // num_samples, num_samples, emb)
        p = p.view(len(x) // num_samples, num_samples, emb)
        d = d.view(len(x) // num_samples, num_samples, emb)
        if world_size > 1:
            with torch.no_grad():
                all_e = [torch.zeros_like(e) for _ in range(world_size)]
                torch.distributed.all_gather(all_e, e)
            all_e[local_rank] = e
            e = torch.cat(all_e)
            with torch.no_grad():
                all_p = [torch.zeros_like(p) for _ in range(world_size)]
                torch.distributed.all_gather(all_p, p)
            all_p[local_rank] = p
            p = torch.cat(all_p)
            with torch.no_grad():
                all_d = [torch.zeros_like(d) for _ in range(world_size)]
                torch.distributed.all_gather(all_d, d)
            all_d[local_rank] = d
            d = torch.cat(all_d)
        loss = 0
        for i in range(num_samples):
            for j in range(num_samples):
                if i != j:
                    l = loss_e(e[:, i], e[:, j]) + loss_p(p[:, i], p[:, j]) + loss_d(d[:, i], d[:, j])
                    loss += l
                    stats_ep.append({"loss": l.item()})

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 3)
        optimizer.step()

    #     if (ep + 1) in eval_ep:
    #         rh= evaluate(get_emb_f, ds, hyp_c)

    if (ep + 1) % 10 == 0 or ep == 0:
        rh = evaluate(get_emb_f, ds, hyp_c)
        all_rh.append(rh)
        if ep == 0:
            best_rh = rh
        else:
            if isinstance(rh, list):
                if rh[0] >= best_rh[0]:
                    lower_cnt = 0
                    best_rh = rh
                    best_ep = ep
                    print("save model........")
                    torch.save(model.state_dict(), save_path)
                else:
                    lower_cnt += 1
            else:
                if rh >= best_rh:
                    lower_cnt = 0
                    best_rh = rh
                    best_ep = ep
                    print("save model........")
                    torch.save(model.state_dict(), save_path)
                else:
                    lower_cnt += 1

    if lower_cnt >= 20:
        break

    if local_rank == 0:
        stats_ep = {k: np.mean([x[k] for x in stats_ep]) for k in stats_ep[0]}
        if (ep + 1) in eval_ep:
            stats_ep = {"recall": rh, **stats_ep}
        wandb.log({**stats_ep, "ep": ep})

print("best:", best_ep + 1, best_rh)
print("save_path:", save_path)
print("load_model:", load_model)

if save_emb:
    ds_type = "gallery" if ds == "Inshop" else "eval"
    x, y = get_emb_f(ds_type=ds_type)
    x, y = x.float().cpu(), y.long().cpu()
    torch.save((x, y), path + "/" + emb_name + "_eval.pt")

    x, y = get_emb_f(ds_type="train")
    x, y = x.float().cpu(), y.long().cpu()
    torch.save((x, y), path + "/" + emb_name + "_train.pt")

# In[ ]:
