import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from .utils import _init_weights
from .GAFR import GAFR
from .CDM import CDM
from .VBL import VBL
import math


class ETN(nn.Module):
    def __init__(self, dim, attr_num, drop_rate=.3, n_head=8, glove_vector_length=300):
        super().__init__()
        self.encoder_dim = dim
        self.vector_length = glove_vector_length
        self.attr_num = attr_num
        self.v_encoder = Resnet(196, self.encoder_dim)
        self.coarse_extractor = nn.Linear(self.encoder_dim, attr_num, bias=False)  # predict region weight
        self.W_g = nn.Linear(self.encoder_dim, attr_num, bias=False)  # predict global logit
        self.W_l = nn.Linear(self.encoder_dim, attr_num, bias=False)
        self.W1 = nn.Linear(glove_vector_length, self.encoder_dim, bias=True)
        self.APP = APP(self.encoder_dim, drop_rate=drop_rate, n_head=n_head)
        self.GAFR = GAFR(self.encoder_dim, num_heads=n_head, qkv_bias=True, attr_num=attr_num)
        self.VBL = VBL(self.encoder_dim)
        self.CPP = CPP(self.encoder_dim, attr_num)
        self.CDM = CDM(self.encoder_dim)
        self.act = nn.GELU
        self.norm = nn.LayerNorm(self.encoder_dim)
        self.apply(_init_weights)    # Use Kaiming Uniform for parameter initialization

    def forward(self, x, w2v):
        # get feature representation and get attr feature init
        local_feature, global_feature = self.v_encoder(x)
        local_feature = F.normalize(local_feature, dim=-1)
        region_weight = torch.einsum('bre, ae -> bra', local_feature, self.coarse_extractor.weight)
        region_weight = F.softmax(region_weight, dim=1)
        attr_feature_init = torch.einsum('bre, bra -> bae', local_feature, region_weight)
        attr_feature_init = F.normalize(attr_feature_init, dim=-1)

        """  get augmentation global feature  """
        mask_ori = self.CDM(global_feature)
        mask = torch.where(mask_ori > .5, torch.ones_like(mask_ori), torch.zeros_like(mask_ori))
        global_feature = global_feature * (1 + .5 * mask)

        """  get refined local feature  """
        attr_mask = self.VBL(attr_feature_init)
        w2v_corr = self.W1(w2v.unsqueeze(0).expand(attr_mask.shape[0], self.attr_num, self.vector_length))
        w2v_corr = w2v_corr * attr_mask
        w2v_attr_f, attn_mask = self.APP(w2v_corr, attr_feature_init, attr_feature_init)
        w2v_attr = self.norm(w2v_attr_f + attr_feature_init)

        attr_refine, _ = self.GAFR(w2v_attr, (32, 64))
        attr_refine = F.normalize(attr_refine + w2v_attr, dim=-1)

        local_result = torch.einsum('bae, ae -> ba', attr_refine, self.W_l.weight)
        global_result = self.W_g(global_feature)

        """  CPP Learns The Semantic Bias Representation  """
        global_bias = self.CPP(global_feature)
        local_bias = self.CPP(attr_refine).mean(1)

        package = {'F_coar': attr_feature_init, 'local_result': local_result, 'global_result': global_result,
                   "global_bias": global_bias, 'local_bias': local_bias, 'delta_l': attr_mask.mean(-1)}
        return package


class Resnet(nn.Module):
    def __init__(self, region_num, v_embedding):
        super(Resnet, self).__init__()
        resnet = torchvision.models.resnet101(pretrained=True)
        self.region_num = region_num
        self.v_embedding = v_embedding
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.pooling = nn.Sequential(*list(resnet.children())[-2:-1])

    def forward(self, x):
        x = self.resnet(x)
        local_feature = x
        local_feature = local_feature.reshape(x.shape[0], self.v_embedding, self.region_num).permute(0, 2, 1)
        x = self.pooling(x)
        x = x.reshape(-1, self.v_embedding)
        global_feature = x
        return local_feature, global_feature


class CPP(nn.Module):
    def __init__(self, v_embedding, attr_num, activation='relu'):
        super(CPP, self).__init__()
        self.fc1 = nn.Linear(v_embedding, v_embedding // 2, bias=True)
        self.fc2 = nn.Linear(v_embedding // 2, v_embedding // 4, bias=True)
        self.fc3 = nn.Linear(v_embedding // 4, attr_num, bias=True)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()

    def forward(self, x):
        bias = self.act(self.fc1(x))
        bias = self.act(self.fc2(bias))
        bias = self.fc3(bias)

        return bias


norm = nn.LayerNorm


class APP(nn.Module):
    def __init__(self, dim, qkv_bias=True, n_head=8, drop_rate=.2):
        super().__init__()
        self.dim = dim
        self.norm_q, self.norm_k, self.norm_v = norm(self.dim), norm(self.dim), norm(self.dim)
        self.proj_q, self.proj_k, self.proj_v = (
            nn.Linear(self.dim, self.dim, bias=qkv_bias), nn.Linear(self.dim, self.dim, bias=qkv_bias),
            nn.Linear(self.dim, self.dim, bias=qkv_bias))
        self.head = n_head
        self.head_dim = self.dim // self.head
        self.proj = nn.Linear(self.dim, self.dim)
        self.dropout = nn.Dropout(drop_rate)
        self.scale = 1 / math.sqrt(self.head_dim)
        self.act = nn.GELU()
        self.apply(_init_weights)

    def get_qkv(self, q, k, v):
        B, C, N = q.shape
        B, R, N = k.shape
        s, k, v = self.norm_q(q), self.norm_k(k), self.norm_v(v)
        q, k, v = (
            self.proj_q(q),
            self.proj_k(k),
            self.proj_v(v))
        q = q.reshape(B, C, self.head, N // self.head).permute(0, 2, 1, 3)
        k = k.reshape(B, C, self.head, N // self.head).permute(0, 2, 1, 3)
        v = v.reshape(B, C, self.head, N // self.head).permute(0, 2, 1, 3)
        return q, k, v

    def forward(self, q=None, k=None, v=None):
        B, C, N = k.shape
        q, k, v = self.get_qkv(q, k, v)
        attn_mask = torch.einsum('b h c n, b h n k -> b h c k', q, k.permute(0, 1, 3, 2))
        attn_mask = F.gelu(attn_mask) * self.scale
        attn_mask = F.softmax(attn_mask, dim=-1)
        out = torch.einsum('b h c k, b h k n -> b h c n', attn_mask, v.float())
        out = out.reshape(B, C, N)
        out = F.normalize(self.proj(out), dim=-1)
        out = self.dropout(out)
        return out, attn_mask
