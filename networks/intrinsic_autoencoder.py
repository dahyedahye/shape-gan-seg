import torch
import torch.nn as nn

def conv7x7(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """7x7 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=7,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

class ResidualBlock(nn.Module):
    """Residual block with conv + batch normalization + ReLU"""
    def __init__(self, in_channels, out_channels, down_stride=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = 