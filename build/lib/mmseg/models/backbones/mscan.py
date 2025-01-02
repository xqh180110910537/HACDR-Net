# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
import math
import warnings
from mmseg.ops import resize
import torch.nn.functional as F
import numpy
import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from mmcv.runner import BaseModule, Sequential

from mmseg.models.builder import BACKBONES
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd


class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class Mlp(BaseModule):
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

    def forward(self, x):
        """Forward function."""

        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class StemConv(BaseModule):
    """Stem Block at the beginning of Semantic Branch.

    Args:
        in_channels (int): The dimension of input channels.
        out_channels (int): The dimension of output channels.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True), deploy=False):
        super(StemConv, self).__init__()
        self.deploy = deploy
        self.start = True
        self.proj1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
        )
        self.proj2 = nn.Sequential(

            nn.Conv2d(
                out_channels // 2,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )
        if deploy:
            self.reparm1 = Sequential(nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)), )

            self.reparm2 = Sequential(nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)), )

    def deploy_(self):
        if self.deploy and self.start:
            proj1_weight, proj1_bias = self._fuse_bn_tensor(self.proj1)
            proj2_weight, proj2_bias = self._fuse_bn_tensor(self.proj2)
            self.reparm1[0].weight.data = proj1_weight
            self.reparm1[0].bias.data = proj1_bias
            self.reparm2[0].weight.data = proj2_weight
            self.reparm2[0].bias.data = proj2_bias
            for param in self.parameters():
                param.detach_()
            delattr(self, 'proj1')
            delattr(self, 'proj2')
            self.start = False

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            kernel = 0
            running_mean = 0
            running_var = 0
            gamma = 0
            beta = 0
            eps = 0
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def forward(self, x):
        """Forward function."""
        self.deploy_()
        if self.deploy:
            x = self.reparm1(x)
            x = F.relu(x, inplace=True)
            x = self.reparm2(x)
        else:
            x = self.proj1(x)
            x = F.relu(x, inplace=True)
            x = self.proj2(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class MSCAAttention(BaseModule):
    """Attention Module in Multi-Scale Convolutional Attention Module (MSCA).

    Args:
        channels (int): The dimension of channels.
        kernel_sizes (list): The size of attention
            kernel. Defaults: [5, [1, 7], [1, 11], [1, 21]].
        paddings (list): The number of
            corresponding padding value in attention module.
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
    """

    def __init__(self,
                 channels,
                 kernel_sizes=[1, [1, 7], [1, 11], [1, 21]],
                 paddings=[0, [0, 3], [0, 5], [0, 10]],
                 exp_ratios=2,
                 norm_cfg=dict(type='BN', requires_grad=True), deploy=False, ker=13):
        super().__init__()
        self.deploy = deploy
        self.channels = channels
        self.exp_ratios = exp_ratios
        self.ker = ker
        self.conv0 = nn.Conv2d(
            channels,
            int(channels * exp_ratios),
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
        )
        # for i, (kernel_size,
        #         padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
        #     kernel_size_ = [kernel_size, kernel_size[::-1]]
        #     padding_ = [padding, padding[::-1]]
        #     conv_name = [f'conv{i}_1', f'conv{i}_2']
        #     for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,
        #                                        conv_name):
        #         self.add_module(
        #             i_conv,
        #             nn.Conv2d(
        #                 int(channels*exp_ratios),
        #                 int(channels*exp_ratios),
        #                 tuple(i_kernel),
        #                 padding=i_pad,
        #                 groups=channels))
        self.start = True
        self.branch77 = Sequential(
            nn.Conv2d(
                channels * exp_ratios,
                channels * exp_ratios,
                kernel_size=(ker, ker),
                stride=(1, 1),
                padding=(ker // 2, ker // 2),
                bias=True,
                groups=channels * exp_ratios
            ),
            # build_norm_layer(norm_cfg, channels * exp_ratios)[1],
        )
        # self.branch772 = Sequential(
        #     nn.Conv2d(
        #         channels * exp_ratios,
        #         channels * exp_ratios,
        #         kernel_size=(7, 7),
        #         stride=(1, 1),
        #         padding=(3, 3),
        #         bias=False,
        #         groups=channels * exp_ratios
        #     ),
        #     # build_norm_layer(norm_cfg, channels * exp_ratios)[1],
        # )
        self.branch33 = nn.Sequential(
            nn.Conv2d(
                channels * exp_ratios,
                channels * exp_ratios,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=True,
                groups=channels * exp_ratios
            ),
            # build_norm_layer(norm_cfg, channels * exp_ratios)[1],
        )
        # if self.deploy:
        #
        #     # print(self.branch33[0].bias.shape,self.branch77[0].bias.shape)
        #     for param in self.parameters():
        #         param.detach_()
        #     delattr(self, 'branch33')
        #     delattr(self, 'branch77')
        if self.deploy:
            self.reparm = Sequential(
                nn.Conv2d(
                    self.channels * self.exp_ratios,
                    self.channels * self.exp_ratios,
                    kernel_size=(ker, ker),
                    stride=(1, 1),
                    padding=(ker // 2, ker // 2),
                    groups=self.channels * self.exp_ratios
                ),
                # build_norm_layer(norm_cfg, channels*exp_ratios)[1],
            )
        # self.attention = Attention(self.channels * self.exp_ratios, self.channels * self.exp_ratios, 7,
        #                            groups=self.channels * self.exp_ratios,
        #                            kernel_num=4)
        # # (ker_num,)
        # self.weight = nn.Parameter(
        #     torch.randn(4, self.channels * self.exp_ratios,
        #                 self.channels * self.exp_ratios // (self.channels * self.exp_ratios), 7, 7),
        #     requires_grad=True)
        self.conv3 = nn.Conv2d(int(channels * exp_ratios), channels, 1)

        def _pad_1x1_to_3x3_tensor(self, kernel1x1):
            if kernel1x1 is None:
                return 0
            else:
                return F.pad(kernel1x1, [1, 1, 1, 1])  # 在1x1四周补padding

    def deploy_(self):
        if self.deploy and self.start:
            # self.reparm = Sequential(
            #     nn.Conv2d(
            #         self.channels * self.exp_ratios,
            #         self.channels * self.exp_ratios,
            #         kernel_size=(7, 7),
            #         stride=(1, 1),
            #         padding=(3, 3),
            #         groups=self.channels * self.exp_ratios
            #     ),
            #     # build_norm_layer(norm_cfg, channels*exp_ratios)[1],
            # )
            kernel_value = numpy.zeros(
                (self.channels * self.exp_ratios, self.channels * self.exp_ratios // (self.channels * self.exp_ratios),
                 self.ker, self.ker), dtype=numpy.float32
            )
            # 下面的3为最大卷积核7的一半
            for i in range(self.channels * self.exp_ratios):
                kernel_value[i, i % 1, self.ker // 2, self.ker // 2] = 1
            self.id_tensor = torch.from_numpy(kernel_value).to(self.branch77[0].weight.device)

            self.reparm[0].weight.data = self.branch77[0].weight.data + F.pad(self.branch33[0].weight.data,
                                                                              [(self.ker - 3) // 2, (self.ker - 3) // 2,
                                                                               (self.ker - 3) // 2,
                                                                               (self.ker - 3) // 2]) + self.id_tensor
            self.reparm[0].bias.data = self.branch77[0].bias.data + self.branch33[0].bias.data
            # print(self.reparm[0].weight.data.shape, self.reparm[0].bias.data.shape, )
            for param in self.parameters():
                param.detach_()
            delattr(self, 'branch33')
            delattr(self, 'branch77')
            # delattr(self, 'branch772')
            self.start = False

    def forward(self, x):
        """Forward function."""
        # print(self.branch77[0].bias.shape)
        self.deploy_()
        u = x.clone()

        attn = self.conv0(x)

        # # Multi-Scale Feature extraction
        # attn_0 = self.conv0_1(attn)
        # attn_0 = self.conv0_2(attn_0)
        #
        # attn_1 = self.conv1_1(attn)
        # attn_1 = self.conv1_2(attn_1)
        #
        # attn_2 = self.conv2_1(attn)
        # attn_2 = self.conv2_2(attn_2)
        if self.deploy:
            # self.reparm[0].weight.data = self.branch77[0].weight.data + F.pad(self.branch33[0].weight, [2, 2, 2, 2])
            # self.reparm[0].bias.data = self.branch77[0].bias.data + self.branch33[0].bias.data
            attn = self.reparm(attn)
            # # 动态卷积注意力
            # channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(attn)
            # batch_size, in_planes, height, width = attn.size()
            # attn = attn * channel_attention
            # attn = attn.reshape(1, -1, height, width)
            # aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
            # # 动态权重
            # aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            #     [-1, self.channels * self.exp_ratios // (self.channels * self.exp_ratios), 7, 7])
            #
            # attn = F.conv2d(attn, weight=aggregate_weight, bias=None,
            #                 stride=1, padding=3,
            #                 dilation=1, groups=self.channels * self.exp_ratios * batch_size)
            # attn = attn.view(batch_size, self.channels * self.exp_ratios, attn.size(-2), attn.size(-1))
            # attn=attn+attn3
            # print(attn.shape)
        else:
            attn1 = self.branch33(attn)
            attn2 = self.branch77(attn)
            attn = attn + attn1 + attn2
            # 动态卷积注意力
            # channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(attn)
            # batch_size, in_planes, height, width = x.size()
            # attn = attn * channel_attention
            # attn = attn.reshape(1, -1, height, width)
            # aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
            # #动态权重
            # aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            #     [-1, self.channels * self.exp_ratios // (self.channels * self.exp_ratios), 7, 7])
            #
            # attn = F.conv2d(attn, weight=aggregate_weight, bias=None,
            #                 stride=1, padding=3,
            #                 dilation=1, groups=self.channels * self.exp_ratios * batch_size)
            # attn = attn.view(batch_size, self.channels * self.exp_ratios, attn.size(-2), attn.size(-1))
        # Channel Mixing
        attn = self.conv3(attn)

        # Convolutional Attention
        x = attn * u

        return x


class MSCASpatialAttention(BaseModule):
    """Spatial Attention Module in Multi-Scale Convolutional Attention Module
    (MSCA).

    Args:
        in_channels (int): The dimension of channels.
        attention_kernel_sizes (list): The size of attention
            kernel. Defaults: [5, [1, 7], [1, 11], [1, 21]].
        attention_kernel_paddings (list): The number of
            corresponding padding value in attention module.
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
    """

    def __init__(self,
                 in_channels,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 act_cfg=dict(type='GELU'), exp_ratios=2, deploy=False, ker=13,
                 ):
        super().__init__()
        self.proj_1 = nn.Conv2d(in_channels, in_channels, 1)
        self.activation = build_activation_layer(act_cfg)
        self.spatial_gating_unit = MSCAAttention(in_channels,
                                                 attention_kernel_sizes,
                                                 attention_kernel_paddings, exp_ratios=2, deploy=deploy, ker=ker)
        self.proj_2 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        """Forward function."""

        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class MSCABlock(BaseModule):
    """Basic Multi-Scale Convolutional Attention Block. It leverage the large-
    kernel attention (LKA) mechanism to build both channel and spatial
    attention. In each branch, it uses two depth-wise strip convolutions to
    approximate standard depth-wise convolutions with large kernels. The kernel
    size for each branch is set to 7, 11, and 21, respectively.

    Args:
        channels (int): The dimension of channels.
        attention_kernel_sizes (list): The size of attention
            kernel. Defaults: [5, [1, 7], [1, 11], [1, 21]].
        attention_kernel_paddings (list): The number of
            corresponding padding value in attention module.
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
        mlp_ratio (float): The ratio of multiple input dimension to
            calculate hidden feature in MLP layer. Defaults: 4.0.
        drop (float): The number of dropout rate in MLP block.
            Defaults: 0.0.
        drop_path (float): The ratio of drop paths.
            Defaults: 0.0.
        act_cfg (dict): Config dict for activation layer in block.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 channels,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 exp_ratios=2,
                 ker=13,
                 deploy=False):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, channels)[1]
        self.attn = MSCASpatialAttention(channels, attention_kernel_sizes,
                                         attention_kernel_paddings, act_cfg, exp_ratios=2, deploy=deploy, ker=ker)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        # self.norm2 = build_norm_layer(norm_cfg, channels)[1]
        mlp_hidden_channels = int(channels * mlp_ratio)
        # self.mlp = Mlp(
        #     in_features=channels,
        #     hidden_features=mlp_hidden_channels,
        #     act_cfg=act_cfg,
        #     drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((channels)),
            requires_grad=True)

    def forward(self, x, H, W):
        """Forward function."""

        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.attn(self.norm1(x)))
        # x = x + self.drop_path(
        #     self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
        #     self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(BaseModule):
    """Image to Patch Embedding.

    Args:
        patch_size (int): The patch size.
            Defaults: 7.
        stride (int): Stride of the convolutional layer.
            Default: 4.
        in_channels (int): The number of input channels.
            Defaults: 3.
        embed_dims (int): The dimensions of embedding.
            Defaults: 768.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 patch_size=7,
                 stride=4,
                 in_channels=3,
                 embed_dim=768,
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2)
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        """Forward function."""

        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W


def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)


class HLFMLP(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1,
                 hidden_size_factor=1, ):
        super().__init__()
        # self.HF = [SpectralGatingNetwork(hidden_size, num_blocks=num_blocks, sparsity_threshold=0.01,
        #                                  hard_thresholding_fraction=1,
        #                                  hidden_size_factor=1, ),
        #            SpectralGatingNetwork(hidden_size, num_blocks=num_blocks, sparsity_threshold=0.01,
        #                                  hard_thresholding_fraction=1,
        #                                  hidden_size_factor=1, ),
        #            SpectralGatingNetwork(hidden_size, num_blocks=num_blocks, sparsity_threshold=0.01,
        #                                  hard_thresholding_fraction=1,
        #                                  hidden_size_factor=1, ),
        #            SpectralGatingNetwork(hidden_size, num_blocks=num_blocks, sparsity_threshold=0.01,
        #                                  hard_thresholding_fraction=1,
        #                                  hidden_size_factor=1, ), ]
        self.LF = SpectralGatingNetwork(hidden_size, num_blocks=num_blocks, sparsity_threshold=0.01,
                                        hard_thresholding_fraction=1,
                                        hidden_size_factor=1, )
        # self.conv = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1, stride=1,
        #                       groups=hidden_size)
        # self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1, padding=0, stride=1,
        #                        )

    def forward(self, x):
        B, C, H, W = x.shape
        bias = x.clone()
        y = F.max_pool2d(x, 2, 2) + F.avg_pool2d(x, 2, 2)
        # x1, x2, x3, x4 = x[:, :, 0:H // 2, 0:W // 2], x[:, :, H // 2:H, 0:W // 2], x[:, :, 0:H // 2, W // 2:W], x[:, :,
        #                                                                                                         H // 2:H,W // 2:W]
        #
        # x1=x1.to(x.device)
        # x2=x2.to(x.device)
        # x3 = x3.to(x.device)
        # x4 = x4.to(x.device)
        y = y.to(y.device)
        y = self.LF(y)
        y = resize(
            y,
            size=bias.shape[2:],
            mode='bilinear',
            align_corners=False)
        # x1 = self.HF[0](x1)
        # x2 = self.HF[1](x2)
        # x3 = self.HF[2](x3)
        # x4 = self.HF[3](x4)
        # u = torch.cat([x1, x2], dim=2)
        # v = torch.cat([x3, x4], dim=2)
        # x = torch.cat([u, v], dim=3)
        # x = self.conv2(self.conv(x))
        return y + bias


class SpectralGatingNetwork(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1,
                 hidden_size_factor=1, ):
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
        if hidden_size == 64:
            self.h = 256 // 8
        elif hidden_size == 96:
            self.h = 256 // 16
        elif hidden_size == 128:
            self.h = 256 // 32
        elif hidden_size == 160:
            self.h = 256 // 64
        # self.w3 = nn.Parameter(
        #     self.scale * torch.randn(2, self.block_size * self.hidden_size_factor, self.h, self.h, ))
        # self.w4 = nn.Parameter(
        #     self.scale * torch.randn(2, self.block_size , self.h, self.h, ))

    def forward(self, x, spatial_size=None):
        # x = x + self.drop_path(
        #     self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
        #     self.mlp(self.norm1(x)))
        bias = x.clone()
        dtype = x.dtype
        # x = F.max_pool2d(x, 2, 2) + F.avg_pool2d(x, 2, 2)
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
        o1_real = (
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[0].to(x.device)) - \
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[1].to(x.device)) + \
            self.b1[0, :, :, None, None].to(x.device)
        )
        o1_imag = (
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[0].to(x.device)) + \
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[1].to(x.device)) + \
            self.b1[1, :, :, None, None].to(x.device)
        )
        # o1_real2 = F.relu(
        #     torch.einsum('bkihw,iho->bkiow', x.real, self.w3[0].to(x.device)) - \
        #     torch.einsum('bkihw,iho->bkiow', x.imag, self.w3[1].to(x.device))
        # )
        # o1_imag2 = F.relu(
        #     torch.einsum('bkihw,iho->bkiow', x.imag, self.w3[0].to(x.device)) + \
        #     torch.einsum('bkihw,iho->bkiow', x.real, self.w3[1].to(x.device))
        # )
        o2_real = (
                torch.einsum('bkihw,kio->bkohw', o1_real, self.w2[0].to(x.device)) - \
                torch.einsum('bkihw,kio->bkohw', o1_imag, self.w2[1].to(x.device)) + \
                self.b2[0, :, :, None, None].to(x.device)
        )

        o2_imag = (
                torch.einsum('bkihw,kio->bkohw', o1_imag, self.w2[0].to(x.device)) + \
                torch.einsum('bkihw,kio->bkohw', o1_real, self.w2[1].to(x.device)) + \
                self.b2[1, :, :, None, None].to(x.device)
        )
        # o2_real2 = F.relu(
        #     torch.einsum('bkihw,iho->bkiow', o2_real, self.w4[0].to(x.device)) - \
        #     torch.einsum('bkihw,iho->bkiow', o2_imag, self.w4[1].to(x.device))
        # )
        # o2_imag2 = F.relu(
        #     torch.einsum('bkihw,iho->bkiow', o2_real, self.w4[0].to(x.device)) + \
        #     torch.einsum('bkihw,iho->bkiow', o2_imag, self.w4[1].to(x.device))
        # )
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, C, x.shape[3], x.shape[4])
        x = x * origin_ffted
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        x = x.type(dtype)
        return x + bias


@BACKBONES.register_module()
class MSCAN(BaseModule):
    """SegNeXt Multi-Scale Convolutional Attention Network (MCSAN) backbone.

    This backbone is the implementation of `SegNeXt: Rethinking
    Convolutional Attention Design for Semantic
    Segmentation <https://arxiv.org/abs/2209.08575>`_.
    Inspiration from https://github.com/visual-attention-network/segnext.

    Args:
        in_channels (int): The number of input channels. Defaults: 3.
        embed_dims (list[int]): Embedding dimension.
            Defaults: [64, 128, 256, 512].
        mlp_ratios (list[int]): Ratio of mlp hidden dim to embedding dim.
            Defaults: [4, 4, 4, 4].
        drop_rate (float): Dropout rate. Defaults: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.
        depths (list[int]): Depths of each Swin Transformer stage.
            Default: [3, 4, 6, 3].
        num_stages (int): MSCAN stages. Default: 4.
        attention_kernel_sizes (list): Size of attention kernel in
            Attention Module (Figure 2(b) of original paper).
            Defaults: [5, [1, 7], [1, 11], [1, 21]].
        attention_kernel_paddings (list): Size of attention paddings
            in Attention Module (Figure 2(b) of original paper).
            Defaults: [2, [0, 3], [0, 5], [0, 10]].
        norm_cfg (dict): Config of norm layers.
            Defaults: dict(type='SyncBN', requires_grad=True).
        pretrained (str, optional): model pretrained path.
            Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels=3,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 depths=[3, 4, 6, 3],
                 exp_ratios=[2, 2, 2, 2],
                 num_stages=4,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 pretrained=None,
                 ker=[7, 7, 13, 13],
                 init_cfg=None, deploy=False):
        super(MSCAN, self).__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.depths = depths
        self.num_stages = num_stages
        self.embed_dims = embed_dims
        self.proj2 = nn.Sequential(
            nn.Conv2d(
                embed_dims[0] // 2,
                embed_dims[0],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            nn.BatchNorm2d(embed_dims[0]),
        )
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        # self.conv1=torch.fft.fft()
        # self.conv1=nn.Conv2d(in_channels=self.embed_dims[0],out_channels=self.embed_dims[0]//4,kernel_size=1)
        # self.conv2 = nn.Conv2d(in_channels=self.embed_dims[1], out_channels=self.embed_dims[1] // 4, kernel_size=1)
        # self.conv3 = nn.Conv2d(in_channels=self.embed_dims[2], out_channels=self.embed_dims[2] // 4, kernel_size=1)
        # self.conv4 = nn.Conv2d(in_channels=self.embed_dims[3], out_channels=self.embed_dims[3] // 4, kernel_size=1)

        depths2 = [1, 1, 1, 1]

        for i in range(num_stages):
            fft_block = []
            for j in range(depths2[i]):
                fft_block.append(HLFMLP(hidden_size=embed_dims[i])
                                 )
            fft = nn.ModuleList(
                fft_block
            )

            if i == 0:
                patch_embed = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        embed_dims[0] // 2,
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=(1, 1)),
                    nn.BatchNorm2d(embed_dims[0] // 2),
                )
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_channels=in_channels if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                    norm_cfg=norm_cfg)

            block = nn.ModuleList([
                MSCABlock(
                    exp_ratios=exp_ratios[i],
                    channels=embed_dims[i],
                    attention_kernel_sizes=attention_kernel_sizes,
                    attention_kernel_paddings=attention_kernel_paddings,
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    ker=ker[i],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg, deploy=deploy) for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)
            setattr(self, f'fft{i + 1}', fft)

    # def eval(self, mode=True):
    #     super(MSCAN, self).eval()
    #
    #     # self._freeze_stages()
    #     # if mode and self.norm_eval:
    #     print(1)
    #     for m in self.modules():
    #         if isinstance(m, MSCAAttention):
    #             m.deploy_()

    def init_weights(self):
        """Initialize modules of MSCAN."""

        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(MSCAN, self).init_weights()

    def forward(self, x):
        """Forward function."""

        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            fft = getattr(self, f'fft{i + 1}')

            # conv=getattr(self, f'conv{i + 1}')
            if i == 0:
                x = patch_embed(x)
                x = F.relu(x, inplace=True)
                outs.append(x)
                x = self.proj2(x)
                x = F.relu(x, inplace=True)
                _, _, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)
            else:
                x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)

            # x = x.permute(0,2,1)
            # print(x.shape)
            # qkv = qkv(x)
            #
            # q = qkv[:,:, 0:dim]
            #
            # k = qkv[:, :,dim:2 * dim]
            # v = qkv[:,:, 2 * dim:3 * dim]
            # x = x.permute(0, 2, 1)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            # x = x.reshape(B, H, W, -1)
            # u=x.clone()
            for fft_b in fft:
                x = fft_b(x)
            # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            # y=conv(x)
            # u = x[:, self.embed_dims[i] * 0 // 4:self.embed_dims[i] * 1 // 4, :, :]
            # v = x[:, self.embed_dims[i] * 1 // 4:self.embed_dims[i] * 2 // 4, :, :]
            # w = x[:, self.embed_dims[i] * 2 // 4:self.embed_dims[i] * 3 // 4, :, :]
            # z = x[:, self.embed_dims[i] * 3 // 4:self.embed_dims[i] * 4 // 4, :, :]

            # print(torch.norm(u * y) / (torch.norm(u) * torch.norm(y)))
            # print(torch.norm(v * y) / (torch.norm(v) * torch.norm(y)))
            # print(torch.norm(w * y) / (torch.norm(w) * torch.norm(y)))
            # print(torch.norm(z * y) / (torch.norm(z) * torch.norm(y)))
            # print(torch.norm(x[:,self.embed_dims[i]*i//4:self.embed_dims[i]*(i+1)//4,:,:]*x[:,self.embed_dims[i]*i//4:self.embed_dims[i]*(i+1)//4,:,:])/(torch.norm(x[:,self.embed_dims[i]*i//4:self.embed_dims[i]*(i+1)//4,:,:])*torch.norm(x[:,0:4,:,:])))
            # print(torch.norm(x[:, 4:8, :, :] * x[:, 8:12, :, :] )/ (torch.norm(x[:, 4:8, :, :]) * torch.norm(x[:, 8:12, :, :])))
            outs.append(x)

        return outs
