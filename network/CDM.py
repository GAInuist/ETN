import torch.nn as nn
import torch.nn.functional as F
from .utils import _init_weights


class CDM(nn.Module):
    def __init__(self, dim):
        super(CDM, self).__init__()
        self.dim = dim
        self.layer1 = nn.Linear(dim, int(dim // 2), bias=True)
        self.layer2 = nn.Linear(int(dim // 2), dim, bias=True)
        self.act1 = nn.GELU()
        self.apply(_init_weights)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        return F.sigmoid(x)

