"""
Intrinsic decomposition autoencoder consisting of residual blocks.
Residual block part is built based on the official PyTorch code for ResNet (https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py).
Note that in this code, identity shortcut connections happen before activation.
"""

import torch
import torch.nn as nn

def conv7x7(in_channels: int, out_channels: int, stride: int=1, padding: int=3) -> nn.Conv2d:
    """7x7 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=padding, bias=False)

def conv3x3(in_channels: int, out_channels: int, stride: int=1, padding: int=1) -> nn.Conv2d:
    """3x3 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv1x1(in_channels: int, out_channels: int, stride: int=1, padding: int=0) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=False)

class EncoderResidualBlock(nn.Module):
    """Residual block with conv + batch normalization + LeakyReLU"""
    def __init__(self, in_channels, out_channels, down_stride=False, down_identity=False):
        super(EncoderResidualBlock, self).__init__()
        stride1 = 1
        if down_stride:
            stride1 = 2
        self.act = nn.LeakyReLU(0.2, inplace=False)
        self.conv1 = conv3x3(in_channels, out_channels, stride1)
        self.bn1 = nn.BatchNorm2d(out_channels, 0.8)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, 0.8)
        self.down_identity = down_identity
        if self.down_identity:
            self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.one_conv = conv1x1(in_channels, out_channels)
    
    def forward(self, x):
        identity = x # note that at this point, x hasn't yet been processed by an activation function.
        out = self.act(x)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_identity:
            identity = self.max_pool(x)
            identity = self.one_conv(identity)

        out += identity # identity shortcut connection before feeding to activiation function

        return out

class DecoderResidualBlock(nn.Module):
    """Residual block with conv + batch normalization + LeakyReLU"""
    def __init__(self, in_channels, out_channels, up_sample=False, normalize=True):
        super(DecoderResidualBlock, self).__init__()
        self.act = nn.LeakyReLU(0.2, inplace=False)
        self.conv1 = conv3x3(in_channels, in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels, 0.8)
        self.conv2 = conv3x3(in_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, 0.8)
        self.up_sample = up_sample
        self.normalize = normalize
        if self.up_sample:
            self.up_output = nn.Sequential(
                nn.Upsample(scale_factor = 2, mode = 'nearest'),
                nn.ReflectionPad2d(1)
                )
            self.up_identity = nn.Upsample(scale_factor = 2, mode = 'nearest')
            self.one_conv = conv1x1(in_channels, out_channels)
    
    def forward(self, x):
        identity = x # note that at this point, x hasn't yet been processed by an activation function.
        out = self.act(x)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.act(out)

        if self.up_sample:
            out = self.up_output(out)
        out = self.conv2(out)
        if self.normalize:
            out = self.bn2(out)

        if self.up_sample:
            identity = self.up_identity(x)
            identity = self.one_conv(identity)

        out += identity # identity shortcut connection before feeding to activiation function

        return out

class ResidualEncoder(nn.Module):
    def __init__(self, in_channels):
        super(ResidualEncoder, self).__init__()
        self.conv1 = conv7x7(in_channels, 64) # in 64x64xin_channels - out 64x64x64
        self.res1 = EncoderResidualBlock(64, 64) # in 64x64x64 - out 64x64x64
        self.res2 = EncoderResidualBlock(64, 64) # in 64x64x64 - out 64x64x64
        self.res3 = EncoderResidualBlock(64, 128, down_stride=True, down_identity=True) # in 64x64x64 - out 32x32x128
        self.res4 = EncoderResidualBlock(128, 128) # in 32x32x128 - out 32x32x128
        self.res5 = EncoderResidualBlock(128, 256, down_stride=True, down_identity=True) # in 32x32x128 - out 16x16x256
        self.res6 = EncoderResidualBlock(256, 256) # in 16x16x256 - out 16x16x256
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        return x

class IntrinsicResidualDecoder(nn.Module):
    def __init__(self, out_channels):
        super(IntrinsicResidualDecoder, self).__init__()
        self.out_channels = out_channels
        self.res1 = DecoderResidualBlock(256, 256) # in 16x16x256 - out 16x16x256
        self.res2 = DecoderResidualBlock(256, 128, up_sample=True) # in 16x16x256 - out 32x32x128
        self.res3 = DecoderResidualBlock(128, 128) # in 32x32x128 - out 32x32x128
        self.res4 = DecoderResidualBlock(128, 64, up_sample=True) # in 32x32x128 - out 64x64x64
        self.res5 = DecoderResidualBlock(64, 64) # in 64x64x64 - out 64x64x64
        self.res6 = DecoderResidualBlock(64, out_channels+1, up_sample=False, normalize=False) # in 64x64x64 - out 64x64x(out_channels+1)
    
    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        out_intrin = torch.sigmoid(x[:,:self.out_channels,:,:].float())
        out_bias = torch.sigmoid(x[:,self.out_channels,:,:].unsqueeze(1).float()) 

        return out_intrin, out_bias

class IntrinsicResidualAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IntrinsicResidualAutoencoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = ResidualEncoder(self.in_channels)
        self.decoder = ResidualDecoder(self.out_channels)

    def forward(self, x):
        x = self.encoder(x)
        out_intrin, out_bias = self.decoder(x)
        return out_intrin, out_bias