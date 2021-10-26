import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual block with conv + batch normalization + ReLU"""
    def __init__(self, in_channels, out_channels, down_stride=False):
        super(ResidualBlock, self).__init__()
        self.