import math
import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn, einsum


def exists(x):
    return x is not None


# ============================== 工具与 CADEF 模块 ==============================

DSM_IS_LAST_CHANNEL = True  # 如果你的 DSM 不在最后一通道，改这里或在 forward 里改切片

def sobel_grad(x):
    # x: [B,1,H,W]  (已归一化的 DSM)
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return gx, gy

def build_2d_sincos_pos_embed(H, W, dim, device):
    assert dim % 4 == 0, "pos dim 必须是 4 的倍数"
    y = torch.linspace(-1.0, 1.0, steps=H, device=device).unsqueeze(1).repeat(1, W)
    x = torch.linspace(-1.0, 1.0, steps=W, device=device).unsqueeze(0).repeat(H, 1)
    omega = torch.arange(dim // 4, device=device).float()
    omega = 1.0 / (10000 ** (omega / (dim // 4)))
    pos_x = x[..., None] * omega
    pos_y = y[..., None] * omega
    pe = torch.cat([torch.sin(pos_x), torch.cos(pos_x), torch.sin(pos_y), torch.cos(pos_y)], dim=-1)  # [H,W,dim]
    return pe.permute(2,0,1).unsqueeze(0)  # [1,dim,H,W]

class HAPE(nn.Module):
    """
    Height-Aware Positional Encoding:
    生成给 Key/Value 注入的高度感知偏置（用 1x1 conv 把 [Pi|h|∇h] 投到 dim_k）
    """
    def __init__(self, dim_k, pos_dim=64):
        super().__init__()
        self.pos_dim = pos_dim
        self.mlp = nn.Conv2d(pos_dim + 3, dim_k, kernel_size=1)  # [Pi:pos_dim, h:1, gx:1, gy:1] -> dim_k

    def forward(self, H, W, dsm_norm):
        # dsm_norm: [B,1,H,W]
        B, _, _, _ = dsm_norm.shape
        device = dsm_norm.device
        # 2D sincos
        pe = build_2d_sincos_pos_embed(H, W, self.pos_dim, device=device).repeat(B,1,1,1)  # [B,pos_dim,H,W]
        gx, gy = sobel_grad(dsm_norm)  # [B,1,H,W] each
        hape_in = torch.cat([pe, dsm_norm, gx, gy], dim=1)  # [B,pos_dim+3,H,W]
        hape = self.mlp(hape_in)  # [B, dim_k, H, W]
        return hape, gx, gy

class CrossAttention2D(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, fine=False, shrink=1, max_tokens_sq=4096):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden = heads * dim_head
        self.to_q = nn.Conv2d(dim, hidden, 1, bias=False)
        self.to_kv = nn.Conv2d(dim + 1, hidden * 2, 1, bias=False)
        self.to_out = nn.Conv2d(hidden, dim, 1)
        self.hape = HAPE(dim_k=hidden, pos_dim=64)
        self.fine = fine

        # ✅ 新增：分辨率收缩系数（>1 时在更小特征图上算注意力）
        self.shrink = shrink
        self.max_tokens_sq = max_tokens_sq  # 超过这个阈值也会自动收缩

        if fine:
            self.alpha_gen = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, x, dsm_at_scale):
        B, C, H, W = x.shape

        # ✅ 动态决定 shrink（大图自动更强收缩）
        shrink = self.shrink
        if (H * W) > self.max_tokens_sq:
            # 例如把 HW 降到不超过阈值（取 2 的幂，避免奇怪的插值尺寸）
            import math
            need = math.ceil(math.sqrt((H * W) / self.max_tokens_sq))
            # 保守一点，至少取 2 的幂
            pow2 = 1
            while pow2 < need:
                pow2 *= 2
            shrink = max(shrink, pow2)

        # 归一化 DSM
        d = dsm_at_scale
        d = (d - d.amin(dim=[2,3], keepdim=True)) / (d.amax(dim=[2,3], keepdim=True) - d.amin(dim=[2,3], keepdim=True) + 1e-6)

        # ✅ 如果 shrink>1：先在低分辨率上做注意力
        if shrink > 1:
            Hs, Ws = H // shrink, W // shrink
            # 平均池化更稳（也可用 stride 卷积）
            x_low = F.avg_pool2d(x, kernel_size=shrink, stride=shrink, ceil_mode=False)
            d_low = F.avg_pool2d(d, kernel_size=shrink, stride=shrink, ceil_mode=False)

            # === 以下与原逻辑一致，但在 (Hs, Ws) 上 ===
            hape_k, gx, gy = self.hape(Hs, Ws, d_low)

            q = self.to_q(x_low)
            q = rearrange(q, "b (h c) x y -> b h c (x y)", h=self.heads) * self.scale

            kv_in = torch.cat([x_low, d_low], dim=1)
            k, v = self.to_kv(kv_in).chunk(2, dim=1)
            k = k + hape_k

            k = rearrange(k, "b (h c) x y -> b h c (x y)", h=self.heads)
            v = rearrange(v, "b (h c) x y -> b h c (x y)", h=self.heads)

            sim = einsum("b h d i, b h d j -> b h i j", q, k)
            sim = sim - sim.amax(dim=-1, keepdim=True).detach()
            attn = sim.softmax(dim=-1)
            out = einsum("b h i j, b h d j -> b h i d", attn, v)
            out = rearrange(out, "b h (x y) d -> b (h d) x y", x=Hs, y=Ws)
            out = self.to_out(out)

            # 上采样回原尺寸 + 细尺度门控
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)

            if self.fine:
                gx_full = F.interpolate(gx, size=(H, W), mode='bilinear', align_corners=False)
                gy_full = F.interpolate(gy, size=(H, W), mode='bilinear', align_corners=False)
                mag = torch.sqrt(gx_full.clamp_min(0)**2 + gy_full.clamp_min(0)**2 + 1e-12)
                alpha = self.alpha_gen(mag)
                out = out * alpha

            return x + out

        # === 原有高精度分支（小图时走这里）===
        hape_k, gx, gy = self.hape(H, W, d)
        q = self.to_q(x)
        q = rearrange(q, "b (h c) x y -> b h c (x y)", h=self.heads) * self.scale

        kv_in = torch.cat([x, d], dim=1)
        k, v = self.to_kv(kv_in).chunk(2, dim=1)
        k = k + hape_k

        k = rearrange(k, "b (h c) x y -> b h c (x y)", h=self.heads)
        v = rearrange(v, "b (h c) x y -> b h c (x y)", h=self.heads)

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=H, y=W)
        out = self.to_out(out)

        if self.fine:
            mag = torch.sqrt(gx.clamp_min(0)**2 + gy.clamp_min(0)**2 + 1e-12)
            alpha = self.alpha_gen(mag)
            out = out * alpha

        return x + out


# ============================== 你原有的模块（未动） ==============================

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim_in, dim_out):
    return nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1)

def Downsample(dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, 4, 2, 1)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
    def forward(self, x):
        x = self.proj(x); x = self.norm(x); x = self.act(x); return x

class ResNetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
                    if exists(time_emb_dim) else None)
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1)
    def forward(self, x, time_emb=None):
        h = self.block1(x)
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q = q * self.scale
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class NetworkConfig:
    """Configuration for the network."""
    # Default configuration
    image_channels = 4
    n_classes = 6
    dim = 32
    dim_mults = (1, 2, 4, 8)
    resnet_block_groups = 8

    # diffusion parameters
    n_timesteps = 25
    n_scales = 3
    max_patch_size = 512
    scale_procedure = "loop"  # "linear" or "loop"

    # ensemble parameters
    built_in_ensemble = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Network(nn.Module):
    def __init__(self, network_config=NetworkConfig()):
        super().__init__()
        self.config = network_config
        image_channels = self.config.image_channels
        n_classes = self.config.n_classes
        dim = self.config.dim
        dim_mults = self.config.dim_mults
        resnet_block_groups = self.config.resnet_block_groups

        # determine dimensions
        self.image_channels = image_channels
        self.n_classes = n_classes
        self.dims = [c * dim for c in dim_mults]

        # time embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # image initial
        self.image_initial = nn.ModuleList([
            ResNetBlock(image_channels, self.dims[0], time_emb_dim=time_dim, groups=resnet_block_groups),
            ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups),
            ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups)
        ])

        # segmentation initial（保持不变：如果你把 seg 作为条件）
        self.seg_initial = nn.ModuleList([
            ResNetBlock(n_classes, self.dims[0], time_emb_dim=time_dim, groups=resnet_block_groups),
            ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups),
            ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups)
        ])

        # encoder (down)
        self.down = nn.ModuleList([])
        # 同步准备 CADEF：与 down 的每一层分辨率对齐
        self.cadef_blocks = nn.ModuleList([])
        for i in range(len(dim_mults) - 1):
            dim_in = self.dims[i]
            dim_out = self.dims[i + 1]
            self.down.append(
                nn.ModuleList([
                    ResNetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                    ResNetBlock(dim_in, dim_in, groups=resnet_block_groups),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out),
                ])
            )
            # 第0层分辨率最大 -> 粗暴 shrink=4；第1层 shrink=2；第2层 shrink=1
            shrink = 4 if i == 0 else (2 if i == 1 else 1)
            self.cadef_blocks.append(
                CrossAttention2D(dim_in, heads=4, dim_head=32, fine=(i < len(dim_mults)-2), shrink=shrink, max_tokens_sq=4096)
            )


        # decoder (up)
        self.up = nn.ModuleList([])
        for i in range(len(dim_mults) - 1):  # each ublock
            dim_in = self.dims[-i - 1]
            dim_out = self.dims[-i - 2]
            if i == 0:
                dim_in_plus_concat = dim_in
            else:
                dim_in_plus_concat = dim_in * 2
            self.up.append(
                nn.ModuleList([
                    ResNetBlock(dim_in_plus_concat, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                    ResNetBlock(dim_in, dim_in, groups=resnet_block_groups),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Upsample(dim_in, dim_out),
                ])
            )

        # final
        self.final = nn.Sequential(
            ResNetBlock(self.dims[0] * 2, self.dims[0], groups=resnet_block_groups),
            ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups),
            nn.Conv2d(self.dims[0], n_classes, 1)
        )

    def _split_top_dsm(self, img):
        # img: [B,C,H,W]
        if DSM_IS_LAST_CHANNEL:
            top = img[:, :-1, :, :]   # 假设前 C-1 通道为 TOP (如 RGB)
            dsm = img[:, -1:, :, :]
        else:
            # 如果不是最后一通道，请按实际顺序切片
            top = img[:, :3, :, :]
            dsm = img[:, 3:4, :, :]
        return top, dsm

    def forward(self, seg, img, time):
        # time embedding
        t = self.time_mlp(time)

        # 初始两支
        resnetblock1, resnetblock2, resnetblock3 = self.seg_initial
        seg_emb = resnetblock1(seg, t); seg_emb = resnetblock2(seg_emb); seg_emb = resnetblock3(seg_emb)

        resnetblock1, resnetblock2, resnetblock3 = self.image_initial
        img_emb = resnetblock1(img, t); img_emb = resnetblock2(img_emb); img_emb = resnetblock3(img_emb)

        # 主干融合起点
        x = seg_emb + img_emb

        # 取出 DSM 并准备一个“与 x 对齐尺度”的金字塔生成器（这里用 interpolate 简化）
        _, dsm_full = self._split_top_dsm(img)  # dsm_full: [B,1,H,W]

        # skip connections
        h = []

        # downsample（每层插入 CADEF）
        for (resnetblock1, resnetblock2, attn, downsample), cadef in zip(self.down, self.cadef_blocks):
            # 先做两个 ResBlock
            x = resnetblock1(x, t)
            x = resnetblock2(x)
            # 与当前分辨率对齐的 DSM
            B, C, H, W = x.shape
            dsm_s = F.interpolate(dsm_full, size=(H, W), mode='bilinear', align_corners=False)
            # CADEF：细尺度（前两层）==> 有门控 α； 粗尺度（最后一层）==> 强几何引导
            x = cadef(x, dsm_s)
            # 注意力 & skip
            x = attn(x)
            h.append(x)
            # 下采样进入下一层
            x = downsample(x)

        # upsample
        for resnetblock1, resnetblock2, attn, upsample in self.up:
            x = resnetblock1(x, t)
            x = resnetblock2(x)
            x = attn(x)
            x = upsample(x)
            x = torch.cat((x, h.pop()), dim=1)

        return self.final(x)
