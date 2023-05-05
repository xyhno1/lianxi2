model = 'vit_small_patch16_224'
hyp_c = 0.1
freeze = 0
emb = 128
resize = 256
crop = 224
clip_r = 2.3
ds = 'CUB'
path = '/data4/czt'
ds_type = "train"
model_path = "/data/chenzhentao/face_forgery_detection/Mixed_curvature/result/vit_small_patch16_224_CUB_best_d_128_1_checkout.pth"

if model.startswith("vit"):
    mean_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
else:
    mean_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

#定义模型
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

def init_model(model=model, hyp_c=0.1, emb=128, clip_r=2.3, freeze=0):
    if model.startswith("dino"):
        body = torch.hub.load("facebookresearch/dino:main", model)
    else:
        body = timm.create_model(model, pretrained=True)
    last = ToProjection_hypersphere(
        c=hyp_c,
        clip_r=clip_r,
    )
    last2 = NormLayer()
    bdim = 2048 if model == "resnet50" else 384
    head = nn.Sequential(nn.Linear(bdim, emb), last2, last)
    nn.init.constant_(head[0].bias.data, 0)
    nn.init.orthogonal_(head[0].weight.data)
    rm_head(body)
    if freeze is not None:
        freezer(body, freeze)
    model = HeadSwitch(body, head)
    model.cuda().train()
    return model


class HeadSwitch(nn.Module):
    def __init__(self, body, head):
        super(HeadSwitch, self).__init__()
        self.body = body
        self.head = head
        self.norm = NormLayer()

    def forward(self, x, skip_head=False):
        x = self.body(x)
        if type(x) == tuple:
            x = x[0]
        if not skip_head:
            x = self.head(x)
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

#定义数据集
def eval_dataset(model, dl, skip_head=True):
    all_x, all_y = [], []
    for x, y in dl:
        with torch.no_grad():
            x = x.cuda(non_blocking=True)
            all_x.append(model(x, skip_head=skip_head))
        all_y.append(y)
    return torch.cat(all_x), torch.cat(all_y)

#得到模型输出
world_size = int(os.environ.get("WORLD_SIZE", 1))

ds_list = {"CUB": CUBirds, "SOP": SOP, "Cars": Cars, "Inshop": Inshop_Dataset}
ds_class = ds_list[ds]

model = init_model(model=model, hyp_c=hyp_c, emb=emb, clip_r=clip_r, freeze=freeze)
model.load_state_dict(torch.load(model_path), False)

eval_tr = T.Compose(
    [
        T.Resize(resize, interpolation=PIL.Image.BICUBIC),
        T.CenterCrop(crop),
        T.ToTensor(),
        T.Normalize(*mean_std),
    ]
)
ds_eval = ds_class(path, ds_type, eval_tr)

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
x, y = eval_dataset(model, dl_eval)
y = y.cuda()