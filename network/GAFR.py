import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.pointwise_conv(self.conv1(x))
        return x


class Agg_0(nn.Module):
    def __init__(self, seg_dim):
        super().__init__()
        self.conv = SeparableConv2d(seg_dim, seg_dim, 3, 1, 1)
        self.norm = nn.GroupNorm(num_channels=seg_dim, num_groups=2) if seg_dim % 2 == 0 else nn.GroupNorm(num_channels=seg_dim, num_groups=1)
        self.act = nn.Hardswish()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(self.norm(x))

        return x


class Aggregator(nn.Module):
    def __init__(self, dim, seg=4):
        super().__init__()
        self.dim = dim
        self.seg = seg

        seg_dim = self.dim // self.seg

        self.norm0 = nn.SyncBatchNorm(seg_dim)
        self.act0 = nn.Hardswish()

        self.agg1 = SeparableConv2d(seg_dim, seg_dim, 3, 1, 1)
        self.norm1 = nn.SyncBatchNorm(seg_dim)
        self.act1 = nn.Hardswish()

        self.agg3 = SeparableConv2d(seg_dim, seg_dim, 3, 1, 1)
        self.norm3 = nn.SyncBatchNorm(seg_dim)
        self.act3 = nn.Hardswish()

        self.agg4 = SeparableConv2d(seg_dim, seg_dim, 5, 1, 1)
        self.norm4 = nn.SyncBatchNorm(seg_dim)
        self.act4 = nn.Hardswish()

        self.agg2 = SeparableConv2d(seg_dim, seg_dim, 5, 1, 2)
        self.norm2 = nn.SyncBatchNorm(seg_dim)
        self.act2 = nn.Hardswish()

        self.agg0 = Agg_0(seg_dim)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        x = x.transpose(1, 2).view(B, C, H, W)
        seg_dim = self.dim // self.seg
        x = x.split([seg_dim] * self.seg, dim=1)

        x_local = x[-1].reshape(B, seg_dim, H, W)
        x_local = self.agg0(x_local)
        if self.seg == 4:
            x0 = self.act0(self.norm0(x[0]))
            x1 = self.act1(self.norm1(self.agg1(x[1])))
            x2 = self.act2(self.norm2(self.agg2(x[2])))
            x = torch.cat([x0, x1, x2, x_local], dim=1)
        elif self.seg == 5:
            x0 = self.act0(self.norm0(x[0]))
            x1 = self.act1(self.norm1(self.agg1(x[1])))
            x2 = self.act2(self.norm2(self.agg2(x[2])))
            x3 = self.act3(self.norm3(self.agg3(x[3])))
            x = torch.cat([x0, x1, x2, x3, x_local], dim=1)
        elif self.seg == 6:
            x0 = self.act0(self.norm0(x[0]))
            x1 = self.act1(self.norm1(self.agg1(x[1])))
            x2 = self.act2(self.norm2(self.agg2(x[2])))
            x3 = self.act3(self.norm3(self.agg3(x[3])))
            x4 = self.act4(self.norm4(self.agg3(x[4])))
            x = torch.cat([x0, x1, x2, x3, x4, x_local], dim=1)

        return x.reshape(B, C, N)



norm = nn.LayerNorm
class GAFR(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attr_num=312):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.norm_q, self.norm_k, self.norm_v = norm(self.dim), norm(self.dim), norm(self.dim)
        self.proj_q, self.proj_k, self.proj_v = (
            nn.Linear(self.dim, self.dim, bias=qkv_bias), nn.Linear(self.dim, self.dim, bias=qkv_bias),
            nn.Linear(self.dim, self.dim, bias=qkv_bias))
        self.attn_drop = nn.Dropout(attn_drop)  # Note: attn_drop is actually not used.
        self.proj = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(proj_drop)
        if attr_num == 312:
            seg = 4
        elif attr_num == 102:
            seg = 6
        else:
            seg = 5
        self.aggregator = Aggregator(dim=attr_num, seg=seg)

    def get_qkv(self, q, k, v):
        q, k, v = self.norm_q(q), self.norm_k(k), self.norm_v(v)
        q, k, v = (
            self.proj_q(q),
            self.proj_k(k),
            self.proj_v(v))
        return q, k, v

    def forward(self, x, size):
        B, C, N = x.shape  # 32 x 312 x 2048
        q, k, v = self.get_qkv(x, x, x)   # shape is batch x C x N
        q = self.aggregator(q.transpose(1, 2), size).reshape(B, C, self.num_heads, N // self.num_heads).permute(0, 2, 1, 3)
        k = self.aggregator(k.transpose(1, 2), size).reshape(B, C, self.num_heads, N // self.num_heads).permute(0, 2, 1, 3)
        v = self.aggregator(v.transpose(1, 2), size).reshape(B, C, self.num_heads, N // self.num_heads).permute(0, 2, 1, 3)
        attn_mask = torch.einsum('b h c n, b h n k -> b h c k', q, k.permute(0, 1, 3, 2))
        attn_mask = F.gelu(attn_mask) * self.scale
        attn_mask = F.softmax(attn_mask, dim=-1)
        out = torch.einsum('b h c k, b h k n -> b h c n', attn_mask, v.float())
        out = out.reshape(B, C, N)
        out = F.normalize(self.proj(out), dim=-1)
        out = self.dropout(out)
        return out, attn_mask
