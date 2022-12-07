import torch
from torch import nn, Tensor
import math

class TransformerEncoderLayer(nn.Module):
    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
    src = torch.rand(32, 10, 512)
    out = encoder_layer(src)