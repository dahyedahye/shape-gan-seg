import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """Double conv block with conv + batch normalization + ReLU"""
    def __init__(self, in_channels, out_channels, down_stride=False):
        super().__init__()
        stride_size = 1
        if down_stride:
            stride_size = 2
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)            
        )
    def forward(self, x):
        return self.double_conv(x)

class UpscaleDoubleConv(nn.Module):
    """Upscale then double conv"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
            )
    def forward(self, x):
        x = self.up(x)
        return self.double_conv(x)

class DoubleConvDownStride(nn.Module):
    """Downscale with stride = 2 and double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels, down_stride=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class OneConv(nn.Module):
    """Adjust depth by one convolution"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.one_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.one_conv(x)

# =============================================================
#                     Two Branch Autoencoder
# =============================================================
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.increase_depth = DoubleConv(in_channels, 64) # 64x64xn_ch -> 64x64X64
        self.down1 = DoubleConvDownStride(64, 128) # 64x64X64 -> 32x32x128

    def forward(self, x):
        x1 = self.increase_depth(x)
        x2 = self.down1(x1)
        return x2


class DecoderSeg(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderSeg, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.decrease_depth = DoubleConv(128, 64) # 32x32x128 -> 32x32x64
        self.up1 = UpscaleDoubleConv(64, out_channels) # 64x64x64 + 64x64x64 -> 64x64x3
        self.adj_out_depth = OneConv(out_channels, 1) # 64x64x3 -> 64x64x1

    def forward(self, latent):
        x = self.decrease_depth(latent)
        x = self.up1(x)
        logits = self.adj_out_depth(x)
        mask = torch.sigmoid(logits)
        return mask


class DecoderRegionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderRegionModule, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.decrease_depth = DoubleConv(128, 64) # 32x32x128 -> 32x32x64
        self.up1 = UpscaleDoubleConv(64, out_channels) # 64x64x64 + 64x64x64 -> 64x64x3

    def forward(self, latent):
        x = self.decrease_depth(latent)
        logits = self.up1(x)
        image = torch.sigmoid(logits)
        return image

class DecoderRegion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderRegion, self).__init__()
        self.net_fg = DecoderRegionModule(in_channels, out_channels)
        self.net_bg = DecoderRegionModule(in_channels, out_channels)

    def forward(self, latent):
        fg = self.net_fg(latent)
        bg = self.net_bg(latent)
        return fg, bg