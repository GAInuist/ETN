import torch.nn as nn
import math


def _init_weights(moudle):
    if isinstance(moudle, nn.Linear):
        nn.init.kaiming_uniform_(moudle.weight, a=math.sqrt(5))
        if moudle.bias is not None:
            fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(moudle.weight)
            bound1 = 1 / math.sqrt(fan_in1)
            nn.init.uniform_(moudle.bias, -bound1, bound1)
    if isinstance(moudle, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d)):
        nn.init.zeros_(moudle.bias)
        nn.init.ones_(moudle.weight)


class MLP(nn.Module):
    def __init__(self, dim, hidden_layer=None, activation=nn.GELU(), drop_path=.1):
        super().__init__()
        self.dim = dim
        self.hidden_layers = hidden_layer
        if self.hidden_layers is None:
            self.hidden_layers = [dim, int(dim // 2), dim]
        self.layer1 = nn.Linear(self.hidden_layers[0], self.hidden_layers[1], bias=True)
        self.layer2 = nn.Linear(self.hidden_layers[1], self.hidden_layers[2], bias=True)
        self.dropout = nn.Dropout(p=drop_path)
        self.act = activation
        self.apply(_init_weights)

    def forward(self, x):
        out = self.act(self.layer1(x))
        out = self.dropout(out)
        out = self.act(self.layer2(out))
        return out
