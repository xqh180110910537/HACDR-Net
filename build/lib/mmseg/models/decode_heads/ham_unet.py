# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead

import torch
from torch import nn
import torch.nn.functional as F

import os
import torch
import torch.nn as nn
from torchvision.transforms import *
import torch.nn.functional as F


class freup_Areadinterpolation(nn.Module):
    def __init__(self, channels):
        super(freup_Areadinterpolation, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))

        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        amp_fuse = Mag.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        pha_fuse = Pha.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)

        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        crop = torch.zeros_like(x)
        crop[:, :, 0:int(H / 2), 0:int(W / 2)] = output[:, :, 0:int(H / 2), 0:int(W / 2)]
        crop[:, :, int(H / 2):H, 0:int(W / 2)] = output[:, :, int(H * 1.5):2 * H, 0:int(W / 2)]
        crop[:, :, 0:int(H / 2), int(W / 2):W] = output[:, :, 0:int(H / 2), int(W * 1.5):2 * W]
        crop[:, :, int(H / 2):H, int(W / 2):W] = output[:, :, int(H * 1.5):2 * H, int(W * 1.5):2 * W]
        crop = F.interpolate(crop, (2 * H, 2 * W))

        return self.post(crop)


class freup_Periodicpadding(nn.Module):
    def __init__(self, channels):
        super(freup_Periodicpadding, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))

        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        amp_fuse = torch.tile(Mag, (2, 2))
        pha_fuse = torch.tile(Pha, (2, 2))

        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)

        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        return self.post(output)


class freup_Cornerdinterpolation(nn.Module):
    def __init__(self, channels):
        super(freup_Cornerdinterpolation, self).__init__()

        self.amp_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))
        self.pha_fuse = nn.Sequential(nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
                                      nn.Conv2d(channels, channels, 1, 1, 0))

        # self.post = nn.Conv2d(channels,channels,1,1,0)

    def forward(self, x):
        N, C, H, W = x.shape

        fft_x = torch.fft.fft2(x)  # n c h w
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        r = x.size(2)  # h
        c = x.size(3)  # w

        I_Mup = torch.zeros((N, C, 2 * H, 2 * W)).to(x.device)
        I_Pup = torch.zeros((N, C, 2 * H, 2 * W)).to(x.device)

        if r % 2 == 1:  # odd
            ir1, ir2 = r // 2 + 1, r // 2 + 1
        else:  # even
            ir1, ir2 = r // 2 + 1, r // 2
        if c % 2 == 1:  # odd
            ic1, ic2 = c // 2 + 1, c // 2 + 1
        else:  # even
            ic1, ic2 = c // 2 + 1, c // 2

        I_Mup[:, :, :ir1, :ic1] = Mag[:, :, :ir1, :ic1]
        I_Mup[:, :, :ir1, ic2 + c:] = Mag[:, :, :ir1, ic2:]
        I_Mup[:, :, ir2 + r:, :ic1] = Mag[:, :, ir2:, :ic1]
        I_Mup[:, :, ir2 + r:, ic2 + c:] = Mag[:, :, ir2:, ic2:]

        if r % 2 == 0:  # even
            I_Mup[:, :, ir2, :] = I_Mup[:, :, ir2, :] * 0.5
            I_Mup[:, :, ir2 + r, :] = I_Mup[:, :, ir2 + r, :] * 0.5
        if c % 2 == 0:  # even
            I_Mup[:, :, :, ic2] = I_Mup[:, :, :, ic2] * 0.5
            I_Mup[:, :, :, ic2 + c] = I_Mup[:, :, :, ic2 + c] * 0.5

        I_Pup[:, :, :ir1, :ic1] = Pha[:, :, :ir1, :ic1]
        I_Pup[:, :, :ir1, ic2 + c:] = Pha[:, :, :ir1, ic2:]
        I_Pup[:, :, ir2 + r:, :ic1] = Pha[:, :, ir2:, :ic1]
        I_Pup[:, :, ir2 + r:, ic2 + c:] = Pha[:, :, ir2:, ic2:]

        if r % 2 == 0:  # even
            I_Pup[:, :, ir2, :] = I_Pup[:, :, ir2, :] * 0.5
            I_Pup[:, :, ir2 + r, :] = I_Pup[:, :, ir2 + r, :] * 0.5
        if c % 2 == 0:  # even
            I_Pup[:, :, :, ic2] = I_Pup[:, :, :, ic2] * 0.5
            I_Pup[:, :, :, ic2 + c] = I_Pup[:, :, :, ic2 + c] * 0.5

        real = I_Mup * torch.cos(I_Pup)
        imag = I_Mup * torch.sin(I_Pup)
        out = torch.complex(real, imag)

        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        return output


## the plug-and-play operator

class fresadd(nn.Module):
    def __init__(self, channels=32):
        super(fresadd, self).__init__()

        self.Fup = freup_Areadinterpolation(channels)

        self.fuse = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        x1 = x

        x2 = F.interpolate(x1, scale_factor=2, mode='bilinear')

        x3 = self.Fup(x1)

        xm = x2 + x3
        xn = self.fuse(xm)

        return xn


class frescat(nn.Module):
    def __init__(self, channels=32):
        super(fresadd, self).__init__()

        self.Fup = freup_Areadinterpolation(channels)

        self.fuse = nn.Conv2d(2 * channels, channels, 1, 1, 0)

    def forward(self, x):
        x1 = x

        x2 = F.interpolate(x1, scale_factor=2, mode='bilinear')

        x3 = self.Fup(x1)

        xn = self.fuse(torch.cat([x2, x3], dim=1))

        return xn


# Taken from https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14


class Hamburger(nn.Module):
    """Hamburger Module. It consists of one slice of "ham" (matrix
    decomposition) and two slices of "bread" (linear transformation).

    Args:
        ham_channels (int): Input and output channels of feature.
        ham_kwargs (dict): Config of matrix decomposition module.
        norm_cfg (dict | None): Config of norm layers.
    """

    def __init__(self,
                 ham_channels=512,
                 ham_kwargs=dict(),
                 norm_cfg=None,
                 **kwargs):
        super().__init__()

        self.ham_in = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=None, act_cfg=None)

        # self.ham = NMF2D(ham_kwargs)

        self.ham_out = ConvModule(
            ham_channels, ham_channels, 1, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        ham = F.relu(x + enjoy, inplace=True)

        return ham


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x


def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)


class gnconv(nn.Module):
    def __init__(self, dim, order=4, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 3, False)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s
        print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)

        return x


class Mlp(nn.Module):
    """Multi Layer Perceptron (MLP) Module.

    Args:
        in_features (int): The dimension of input features.
        hidden_features (int): The dimension of hidden features.
            Defaults: None.
        out_features (int): The dimension of output features.
            Defaults: None.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        drop (float): The number of dropout rate in MLP block.
            Defaults: 0.0.
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=True,
            groups=hidden_features)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)


class ChannelGatingNetwork(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1,
                 hidden_size_factor=1, ):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"
        # self.up_channel = int(alpha * hidden_size)
        # self.low_channel =hidden_size - self.up_channel
        # self.squeeze1 = nn.Conv2d(self.up_channel, self.up_channel // squeeze_radio, kernel_size=1, bias=False)
        # self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # drop_path = 0.1

        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.sparsity_threshold = sparsity_threshold
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        # self.w3 = nn.Parameter(
        #     self.scale * torch.randn(2, self.hidden_size, self.block_size * self.hidden_size_factor, self.block_size))
        # self.conv1=nn.Conv2d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=(1,5),padding=(0,2),stride=(1,1),bias=False)
        # self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(5, 1), padding=(2, 0),
        #                        stride=(1, 1), bias=False)
        # self.conv3 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(1, 5), padding=(0, 2),
        #                        stride=(1, 1), bias=False)
        # self.conv4 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(5, 1), padding=(2, 0),
        #                        stride=(1, 1), bias=False)
        self.conv1 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1,
                               padding=0)
        self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, stride=1,
                               padding=0)
        # self.bn = nn.BatchNorm2d(2 * hidden_size)
        # self.avg=nn.AvgPool2d(2,1,padding=0)
        # layer_scale_init_value = 1e-2
        # self.layer_scale_1 = nn.Parameter(
        #     layer_scale_init_value * torch.ones((hidden_size)),
        #     requires_grad=True)
        # self.mlp = Mlp(
        #     in_features=hidden_size,
        #     hidden_features=4*hidden_size,
        #     drop=0.1)
        # # self.norm_layer1 = get_normalization_layer(opts=opts, num_features=out_channels)
        # self.drop_path = DropPath(
        #     drop_path) if drop_path > 0. else nn.Identity()
        # self.norm1 = build_norm_layer(norm_cfg,hidden_size)[1]

    def forward(self, x, spatial_size=None):
        # x = x + self.drop_path(
        #     self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
        #     self.mlp(self.norm1(x)))
        bias = x.clone()
        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        # x = x.reshape(B, -1, C)
        # qkv = self.qkv(x).reshape(B, H * W, 3, self.num_blocks, self.block_size).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]
        # q, k, v = q.reshape(B, C, H, W), k.reshape(B, C, H, W), v.reshape(B, C, H, W)
        # x = self.fu(x)
        # y = F.relu(x, inplace=True)
        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        origin_ffted = x
        # origin_ffted = x
        x_imag = x.imag
        x_real = x.real
        x_real = self.conv1(x_real)
        x_imag = self.conv2(x_imag)
        # y = self.bn(F.relu(y, inplace=True))+y
        # y = F.sigmoid(y)
        # y_real, y_imag = torch.chunk(y, 2, dim=1)
        x = torch.complex(x_real, x_imag)
        # y = self.conv1(x.real)
        # # x.real = self.conv2(x.real)
        # z = self.conv2(x.imag)
        # x = torch.stack([y, z], dim=-1)
        # x = torch.view_as_complex(x)
        # x.imag = self.conv2(x.imag)
        x = x.reshape(B, self.num_blocks, self.block_size, x.shape[2], x.shape[3])
        # torch.einsum('bkihw,kio->bkohw', x.real, self.w1[0]) - \
        o1_real = F.relu(
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[0]) - \
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[1]) + \
            self.b1[0, :, :, None, None]
        )

        o1_imag = F.relu(
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[0]) + \
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[1]) + \
            self.b1[1, :, :, None, None]
        )

        # o1_imag = o1_imag.reshape(B, C, origin_ffted.shape[2], origin_ffted.shape[3])
        # o1_imag = self.conv1(o1_imag)
        # o1_imag = o1_imag.reshape(B, self.num_blocks, self.block_size, o1_imag.shape[2], o1_imag.shape[3])
        # o1_imag = F.relu(o1_imag, inplace=True)
        #
        # o1_real = o1_real.reshape(B, C, origin_ffted.shape[2], origin_ffted.shape[3])
        # o1_real = self.conv2(o1_real)
        # o1_real = o1_real.reshape(B, self.num_blocks, self.block_size, o1_real.shape[2], o1_real.shape[3])
        # o1_real = F.relu(o1_real, inplace=True)

        o2_real = (
                torch.einsum('bkihw,kio->bkohw', o1_real, self.w2[0]) - \
                torch.einsum('bkihw,kio->bkohw', o1_imag, self.w2[1]) + \
                self.b2[0, :, :, None, None]
        )

        o2_imag = (
                torch.einsum('bkihw,kio->bkohw', o1_imag, self.w2[0]) + \
                torch.einsum('bkihw,kio->bkohw', o1_real, self.w2[1]) + \
                self.b2[1, :, :, None, None]
        )

        #
        # o2_real = o2_real.reshape(B, C, origin_ffted.shape[2], origin_ffted.shape[3])
        # o2_real = self.conv1(o2_real)
        # o2_real = self.conv2(o2_real)
        # o2_real = o2_real.reshape(B, self.num_blocks, self.block_size, o2_real.shape[2], o2_real.shape[3])
        # o2_imag = o2_imag.reshape(B, C, origin_ffted.shape[2], origin_ffted.shape[3])
        # o2_imag = self.conv3(o2_imag)
        # o2_imag = self.conv4(o2_imag)
        # o2_imag = o2_imag.reshape(B, self.num_blocks, self.block_size, o2_imag.shape[2], o2_imag.shape[3])

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)

        # x=F.softmax(x)

        x = torch.view_as_complex(x)
        x = x.reshape(B, C, x.shape[3], x.shape[4])
        # x.real=self.conv1(x.real)
        # x.imag=self.conv2(x.imag)
        # x = x * origin_ffted

        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        x = x.type(dtype)
        # x=torch.matmul(x,bias)

        return x + bias


class SpectralGatingNetwork(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1,
                 hidden_size_factor=1, alpha=0.2, squeeze_radio=2,
                 norm_cfg=dict(type='BN', requires_grad=True), ker=1, ):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"
        # self.up_channel = int(alpha * hidden_size)
        # self.low_channel =hidden_size - self.up_channel
        # self.squeeze1 = nn.Conv2d(self.up_channel, self.up_channel // squeeze_radio, kernel_size=1, bias=False)
        # self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        drop_path = 0.1

        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.sparsity_threshold = sparsity_threshold
        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        # self.w3 = nn.Parameter(
        #     self.scale * torch.randn(2, self.hidden_size, self.block_size * self.hidden_size_factor, self.block_size))
        # self.conv1=nn.Conv2d(in_channels=hidden_size,out_channels=hidden_size,kernel_size=(1,5),padding=(0,2),stride=(1,1),bias=False)
        # self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(5, 1), padding=(2, 0),
        #                        stride=(1, 1), bias=False)
        # self.conv3 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(1, 5), padding=(0, 2),
        #                        stride=(1, 1), bias=False)
        # self.conv4 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(5, 1), padding=(2, 0),
        #                        stride=(1, 1), bias=False)
        self.conv1 = nn.Conv2d(in_channels=2 * hidden_size, out_channels=2 * hidden_size, kernel_size=ker, stride=1,
                               padding=(ker-1)//2, groups=2 * hidden_size)
        # self.bn = nn.BatchNorm2d(2 * hidden_size)
        # self.avg=nn.AvgPool2d(2,1,padding=0)
        # layer_scale_init_value = 1e-2
        # self.layer_scale_1 = nn.Parameter(
        #     layer_scale_init_value * torch.ones((hidden_size)),
        #     requires_grad=True)
        # self.mlp = Mlp(
        #     in_features=hidden_size,
        #     hidden_features=4*hidden_size,
        #     drop=0.1)
        # # self.norm_layer1 = get_normalization_layer(opts=opts, num_features=out_channels)
        # self.drop_path = DropPath(
        #     drop_path) if drop_path > 0. else nn.Identity()
        # self.norm1 = build_norm_layer(norm_cfg,hidden_size)[1]

    def forward(self, x, spatial_size=None):
        # x = x + self.drop_path(
        #     self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
        #     self.mlp(self.norm1(x)))
        bias = x.clone()
        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        # x = x.reshape(B, -1, C)
        # qkv = self.qkv(x).reshape(B, H * W, 3, self.num_blocks, self.block_size).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]
        # q, k, v = q.reshape(B, C, H, W), k.reshape(B, C, H, W), v.reshape(B, C, H, W)
        # x = self.fu(x)
        # y = F.relu(x, inplace=True)
        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        origin_ffted = x
        # origin_ffted = x
        x_imag = x.imag
        x_real = x.real
        y = torch.cat([x_real, x_imag], dim=1)
        y = self.conv1(y)
        # y = self.bn(F.relu(y, inplace=True))+y
        # y = F.sigmoid(y)
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        x = torch.complex(y_real, y_imag)
        # y = self.conv1(x.real)
        # # x.real = self.conv2(x.real)
        # z = self.conv2(x.imag)
        # x = torch.stack([y, z], dim=-1)
        # x = torch.view_as_complex(x)
        # x.imag = self.conv2(x.imag)
        x = x.reshape(B, self.num_blocks, self.block_size, x.shape[2], x.shape[3])
        # torch.einsum('bkihw,kio->bkohw', x.real, self.w1[0]) - \
        o1_real = F.relu(
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[0]) - \
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[1]) + \
            self.b1[0, :, :, None, None]
        )

        o1_imag = F.relu(
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[0]) + \
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[1]) + \
            self.b1[1, :, :, None, None]
        )

        # o1_imag = o1_imag.reshape(B, C, origin_ffted.shape[2], origin_ffted.shape[3])
        # o1_imag = self.conv1(o1_imag)
        # o1_imag = o1_imag.reshape(B, self.num_blocks, self.block_size, o1_imag.shape[2], o1_imag.shape[3])
        # o1_imag = F.relu(o1_imag, inplace=True)
        #
        # o1_real = o1_real.reshape(B, C, origin_ffted.shape[2], origin_ffted.shape[3])
        # o1_real = self.conv2(o1_real)
        # o1_real = o1_real.reshape(B, self.num_blocks, self.block_size, o1_real.shape[2], o1_real.shape[3])
        # o1_real = F.relu(o1_real, inplace=True)

        o2_real = (
                torch.einsum('bkihw,kio->bkohw', o1_real, self.w2[0]) - \
                torch.einsum('bkihw,kio->bkohw', o1_imag, self.w2[1]) + \
                self.b2[0, :, :, None, None]
        )

        o2_imag = (
                torch.einsum('bkihw,kio->bkohw', o1_imag, self.w2[0]) + \
                torch.einsum('bkihw,kio->bkohw', o1_real, self.w2[1]) + \
                self.b2[1, :, :, None, None]
        )

        #
        # o2_real = o2_real.reshape(B, C, origin_ffted.shape[2], origin_ffted.shape[3])
        # o2_real = self.conv1(o2_real)
        # o2_real = self.conv2(o2_real)
        # o2_real = o2_real.reshape(B, self.num_blocks, self.block_size, o2_real.shape[2], o2_real.shape[3])
        # o2_imag = o2_imag.reshape(B, C, origin_ffted.shape[2], origin_ffted.shape[3])
        # o2_imag = self.conv3(o2_imag)
        # o2_imag = self.conv4(o2_imag)
        # o2_imag = o2_imag.reshape(B, self.num_blocks, self.block_size, o2_imag.shape[2], o2_imag.shape[3])

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)

        # x=F.softmax(x)

        x = torch.view_as_complex(x)
        x = x.reshape(B, C, x.shape[3], x.shape[4])
        # x.real=self.conv1(x.real)
        # x.imag=self.conv2(x.imag)
        # x = x * origin_ffted

        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        x = x.type(dtype)
        # x=torch.matmul(x,bias)

        return x + bias


@HEADS.register_module()
class UnetHamHead(BaseDecodeHead):
    """SegNeXt decode head.

    This decode head is the implementation of `SegNeXt: Rethinking
    Convolutional Attention Design for Semantic
    Segmentation <https://arxiv.org/abs/2209.08575>`_.
    Inspiration from https://github.com/visual-attention-network/segnext.

    Specifically, LightHamHead is inspired by HamNet from
    `Is Attention Better Than Matrix Decomposition?
    <https://arxiv.org/abs/2109.04553>`.

    Args:
        ham_channels (int): input channels for Hamburger.
            Defaults: 512.
        ham_kwargs (int): kwagrs for Ham. Defaults: dict().
    """

    def __init__(self, ham_channels=512, reduction_ratio=16, **kwargs):
        super(UnetHamHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        # self.conv1 = nn.Sequential(
        #     nn.BatchNorm2d(self.in_channels[0]),
        #     SpectralGatingNetwork(hidden_size=self.in_channels[0], ker=1),
        #     nn.BatchNorm2d(self.in_channels[0]),
        #     ChannelGatingNetwork(hidden_size=self.in_channels[0]),
        # )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels[1]),
            SpectralGatingNetwork(hidden_size=self.in_channels[1], ker=1),
            nn.BatchNorm2d(self.in_channels[1]),
            ChannelGatingNetwork(hidden_size=self.in_channels[1]),
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels[2]),
            SpectralGatingNetwork(hidden_size=self.in_channels[2], ker=1),
            nn.BatchNorm2d(self.in_channels[2]),
            ChannelGatingNetwork(hidden_size=self.in_channels[2]),
        )
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels[3]),
            SpectralGatingNetwork(hidden_size=self.in_channels[3], ker=1),
            nn.BatchNorm2d(self.in_channels[3]),
            ChannelGatingNetwork(hidden_size=self.in_channels[3]),
        )
        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels[4]),
            SpectralGatingNetwork(hidden_size=self.in_channels[4], ker=1),
            nn.BatchNorm2d(self.in_channels[4]),
            ChannelGatingNetwork(hidden_size=self.in_channels[4]),
        )
        # self.conv6 = nn.Sequential(
        #     SpectralGatingNetwork(hidden_size=64, ker=1),
        #     nn.BatchNorm2d(64),
        #     ChannelGatingNetwork(hidden_size=64),
        # )
        # self.up1 = nn.Upsample()
        # self.conv1 = ConvModule(
        #     self.in_channels[4],
        #     self.in_channels[3],
        #     1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=dict(type='BN', requires_grad=True),
        #     act_cfg=None)
        # # self.up2 = fresadd(channels=self.in_channels[3])
        self.conv = ConvModule(
            ham_channels,
            64,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=None)
        # # self.up3 = fresadd(channels=self.in_channels[2])
        # self.conv3 = ConvModule(
        #     self.in_channels[2],
        #     self.in_channels[1],
        #     1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=dict(type='BN', requires_grad=True),
        #     act_cfg=None)
        # # self.up4 = fresadd(channels=self.in_channels[1])
        # self.conv4 = ConvModule(
        #     self.in_channels[1],
        #     self.in_channels[0],
        #     1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=dict(type='BN', requires_grad=True),
        #     act_cfg=None)
        # # self.ham_channels = ham_channels
        # self.conv_ya = nn.Sequential(nn.Conv2d(in_channels=ham_channels,
        #                                        out_channels=self.in_channels[4],
        #                                        kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(self.in_channels[4]))
        # self.conv_ya2 = nn.Sequential(nn.Conv2d(in_channels=ham_channels,
        #                                        out_channels=self.in_channels[3],
        #                                        kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(self.in_channels[3]))
        # self.conv_ya3 = nn.Sequential(nn.Conv2d(in_channels=ham_channels,
        #                                         out_channels=self.in_channels[2],
        #                                         kernel_size=1, stride=1, padding=0),
        #                               nn.BatchNorm2d(self.in_channels[2]))
        # self.conv_ya4 = nn.Sequential(nn.Conv2d(in_channels=ham_channels,
        #                                         out_channels=self.in_channels[1],
        #                                         kernel_size=1, stride=1, padding=0),
        #                               nn.BatchNorm2d(self.in_channels[1]))
        # self.conv_ya5 = nn.Sequential(nn.Conv2d(in_channels=ham_channels,
        #                                         out_channels=self.in_channels[0],
        #                                         kernel_size=1, stride=1, padding=0),
        #                               nn.BatchNorm2d(self.in_channels[0]))

        # self.conv = Mlp(in_features=ham_channels, hidden_features=ham_channels, out_features=64, drop=0.2)
        # self.fourier_up1 = freup_Areadinterpolation(channels=self.in_channels[4])
        # self.fourier_up2 = freup_Areadinterpolation(channels=self.in_channels[2])
        # self.low5 = nn.Sequential(nn.Conv2d(in_channels=self.in_channels[4], out_channels=self.in_channels[3],
        #                                     kernel_size=1, stride=1, padding=0),
        #                           nn.BatchNorm2d(self.in_channels[3]))
        # self.low4 = nn.Sequential(nn.Conv2d(in_channels=self.in_channels[3] * 2, out_channels=self.in_channels[2],
        #                                     kernel_size=1, stride=1, padding=0), )
        # self.low3 = nn.Sequential(
        #     nn.Conv2d(in_channels=self.in_channels[2] * 2, out_channels=64,
        #               kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(64))
        # num_inputs = len(self.in_channels)
        #
        # assert num_inputs == len(self.in_index)
        # self.convs = nn.ModuleList()
        # for i in range(num_inputs):
        #     self.convs.append(
        #         ConvModule(
        #             in_channels=self.in_channels[i],
        #             out_channels=self.in_channels[i],
        #             kernel_size=1,
        #             stride=1,
        #             norm_cfg=self.norm_cfg,
        #             act_cfg=self.act_cfg))

        # self.conv2 = nn.Conv2d(in_channels=self.in_channels[3] + self.in_channels[2], out_channels=self.in_channels[2],
        #                        kernel_size=1, stride=1, padding=0)
        # self.conv1 = nn.Conv2d(in_channels=self.in_channels[2] + self.in_channels[1], out_channels=self.in_channels[1],
        #                        kernel_size=1, stride=1, padding=0)
        # self.conv0 = nn.Conv2d(in_channels=self.in_channels[1] + self.in_channels[0], out_channels=self.in_channels[0],
        #                        kernel_size=1, stride=1, padding=0)
        self.align = ConvModule(
            64,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=self.act_cfg)
        # self.low1 = ConvModule(
        #     ham_channels,
        #     64,
        #     1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=dict(type='BN', requires_grad=True),
        #     act_cfg=self.act_cfg)
        # self.low2 = ConvModule(
        #     ham_channels,
        #     64,
        #     1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=dict(type='BN', requires_grad=True),
        #     act_cfg=self.act_cfg)
        # self.low3 = ConvModule(
        #     ham_channels,
        #     64,
        #     1,
        #     conv_cfg=self.conv_cfg,
        #     norm_cfg=dict(type='BN', requires_grad=True),
        #     act_cfg=self.act_cfg)
        # self.decoder1 = DecoderBottleneck(self.in_channels[4] + self.in_channels[3], self.in_channels[3])
        # self.decoder2 = DecoderBottleneck(self.in_channels[3] + self.in_channels[2], self.in_channels[2])
        # self.decoder3 = DecoderBottleneck(self.in_channels[2] + self.in_channels[1], self.in_channels[1])
        # self.decoder4 = DecoderBottleneck(self.in_channels[1] + self.in_channels[0], self.in_channels[0])
        # self.cbam1 = CBAM(self.in_channels[3], reduction_ratio=reduction_ratio)
        # self.cbam2 = CBAM(self.in_channels[2], reduction_ratio=reduction_ratio)
        # self.cbam3 = CBAM(self.in_channels[1], reduction_ratio=reduction_ratio)

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)
        inputs[0] = inputs[0]
        inputs[1] = self.conv2(inputs[1])
        inputs[2] = self.conv3(inputs[2])
        inputs[3] = self.conv4(inputs[3])
        inputs[4] = self.conv5(inputs[4])
        inputs= [resize(
            x,
            size=inputs[0].shape[2:],
            mode='bilinear',
            align_corners=self.align_corners) for x in inputs]
        # mid = [resize(
        #     x,
        #     size=inputs[2].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners) for x in inputs]
        # low = [resize(
        #     x,
        #     size=inputs[4].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners) for x in inputs]
        # low = resize(self.low1(torch.cat(low, dim=1)), size=inputs[2].shape[2:],
        #                        mode='bilinear',
        #                        align_corners=self.align_corners)
        # mid = resize(self.low2(torch.cat(mid, dim=1))+low, size=inputs[0].shape[2:],
        #                        mode='bilinear',
        #                        align_corners=self.align_corners)
        # high = resize(self.low3(torch.cat(high, dim=1)) + mid, size=inputs[0].shape[2:],
        #              mode='bilinear',
        #              align_corners=self.align_corners)
        # low = [F.max_pool2d(inputs[i], kernel_size=(32 // pow(2, i + 1), 32 // pow(2, i + 1))) for i in
        #        range(len(inputs))]
        # low=torch.cat(low, dim=1)
        # low1 = F.sigmoid(self.conv_ya(low))
        # low2 = F.sigmoid(resize(
        #     self.conv_ya2(low),
        #     size=inputs[3].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners))
        # low3 = F.sigmoid(resize(
        #     self.conv_ya3(low),
        #     size=inputs[2].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners))
        # low4 = F.sigmoid(resize(
        #     self.conv_ya4(low),
        #     size=inputs[1].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners))
        # low5 = F.sigmoid(resize(
        #     self.conv_ya5(low),
        #     size=inputs[0].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners))
        # inputs[4] = low1 * inputs[4]+inputs[4]
        # inputs[3] = low2 * inputs[3]+inputs[3]
        # inputs[2] = low3 * inputs[2]+inputs[2]
        # inputs[1] = low4 * inputs[1]+inputs[1]
        # inputs[0] = low5 * inputs[0]+inputs[0]
        # low = [resize(
        #     x,
        #     size=inputs[0].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners) for x in inputs]
        # low=torch.cat(low,dim=1)

        # outputs = []
        #
        # outputs.append(F.sigmoid(self.conv4(F.interpolate(inputs[1], scale_factor=2, mode='bilinear'))) * inputs[0])
        #
        # outputs.append(F.sigmoid(self.conv3(F.interpolate(inputs[2], scale_factor=2, mode='bilinear'))) * inputs[1])
        #
        # outputs.append(F.sigmoid(self.conv2(F.interpolate(inputs[3], scale_factor=2, mode='bilinear'))) * inputs[2])
        # outputs.append(F.sigmoid(self.conv1(F.interpolate(inputs[4], scale_factor=2, mode='bilinear'))) * inputs[3])
        # outputs.append(inputs[4])
        # inputs = [resize(
        #     x,
        #     size=inputs[0].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners) for x in outputs]

        # inputs[3] = self.conv1(torch.cat([inputs[3], inputs[4]], dim=1))
        # inputs[4] = resize(
        #     inputs[4],
        #     size=inputs[3].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # inputs[3] = self.conv3(torch.cat([inputs[4], inputs[3]], dim=1))
        # inputs[3] = resize(
        #     inputs[3],
        #     size=inputs[2].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # inputs[2] = self.conv2(torch.cat([inputs[3], inputs[2]], dim=1))
        # inputs[2] = resize(
        #     inputs[2],
        #     size=inputs[1].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # inputs[1] = self.conv1(torch.cat([inputs[2], inputs[1]], dim=1))
        # inputs[1] = resize(
        #     inputs[1],
        #     size=inputs[0].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # inputs[0] = self.conv0(torch.cat([inputs[1], inputs[0]], dim=1))
        # print(inputs[0].shape)

        # high = [
        #     resize(
        #         level,
        #         size=inputs[0].shape[2:],
        #         mode='bilinear',
        #         align_corners=self.align_corners) for level in inputs[0:3]
        # ]
        # high = self.conv3(torch.cat(high, dim=1))
        #
        # low_5 = self.low5(self.fourier_up1(inputs[4]) + resize(
        #     inputs[4],
        #     size=inputs[3].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners))
        #
        # low_4 = self.low4(torch.cat([low_5, inputs[3]], dim=1))
        #
        # low = self.fourier_up2(low_4) + resize(
        #     low_4,
        #     size=inputs[2].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        # low = resize(
        #     self.low3(torch.cat([low, inputs[2]], dim=1)),
        #     size=inputs[0].shape[2:],
        #     mode='bilinear',
        #     align_corners=self.align_corners)

        # inputs=0.5*inputs[0]+0.5*inputs[0]+0.5*inputs[0]+0.5*inputs[0]
        #
        inputs = self.conv(torch.cat(inputs,dim=1))
        # inputs=self.conv6(inputs)
        # inputs=self.gconv(inputs)
        # inputs[3] = resize(inputs[3], size=inputs[2].shape[2:],
        #                    mode='bilinear',
        #                    align_corners=self.align_corners)
        # # inputs[3]=self.cbam1( inputs[3])
        # inputs[2] = self.decoder2(inputs[3], inputs[2])
        # inputs[2] = resize(inputs[2], size=inputs[1].shape[2:],
        #                    mode='bilinear',
        #                    align_corners=self.align_corners)
        # # inputs[2] = self.cbam2(inputs[2])
        # inputs[1] = self.decoder3(inputs[2], inputs[1])
        # inputs[1] = resize(inputs[1], size=inputs[0].shape[2:],
        #                    mode='bilinear',
        #                    align_corners=self.align_corners)
        # # inputs[1] = self.cbam3(inputs[1])
        # inputs[0] = self.decoder4(inputs[1], inputs[0])
        # # inputs = torch.cat(inputs, dim=1)
        # ## apply a conv block to squeeze feature map
        # # x = self.squeeze(inputs)
        # # # apply hamburger module
        # # x = self.hamburger(x)
        #
        # # apply a conv block to align feature map
        output = self.align(inputs)
        # output = self.align(inputs[0])
        output = self.cls_seg(output)
        return output
