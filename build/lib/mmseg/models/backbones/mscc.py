# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
import math
import warnings
import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from mmcv.runner import BaseModule
from ..builder import BACKBONES
from mmcv.ops import ModulatedDeformConv2dPack as DCNv2
from mmcv.ops import DeformConv2dPack as DCN
import torch.nn.functional as F


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
        # self.dwconv =DCNv2(
        #         hidden_features,
        #         hidden_features,
        #         3,
        #         1,
        #         1,
        #         bias=True,
        #         deform_groups=2
        # )
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
        # self.convdcn1 = DCNv2(
        #     hidden_features,
        #     hidden_features,
        #     kernel_size=(1,3),
        #     padding=(0,1),
        #     deform_groups=hidden_features,
        # )
        # self.convdcn2 = DCNv2(
        #     hidden_features,
        #     hidden_features,
        #     kernel_size=(3, 1),
        #     padding=(1,0),
        #     deform_groups=hidden_features,
        # )

    def forward(self, x):
        """Forward function."""

        x = self.fc1(x)
        # x = self.convdcn1(x)
        # x = self.convdcn2(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # x1 = x1.view(B, C, -1).permute(0, 2, 1).contiguous()
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
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels // 2)[1],
            build_activation_layer(act_cfg),
            nn.Conv2d(
                out_channels // 2,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            build_norm_layer(norm_cfg, out_channels)[1],
        )

    def forward(self, x):
        """Forward function."""

        x = self.proj(x)
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
                 kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 paddings=[2, [0, 3], [0, 5], [0, 10]]):
        super().__init__()
        # self.conv0 = nn.Conv2d(
        #     channels,
        #     channels,
        #     kernel_size=kernel_sizes[0],
        #     padding=paddings[0],
        #     groups=channels)
        self.conv0 = DCNv2(
            channels,
            channels,
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            groups=channels)
        # self.conv0 = DCNv2(
        #     channels,
        #     channels,
        #     kernel_size=kernel_sizes[0],
        #     padding=paddings[0],
        #     deform_groups=channels,
        # )
        for i, (kernel_size,
                padding) in enumerate(zip(kernel_sizes[1:], paddings[1:])):

            kernel_size_ = [kernel_size, kernel_size[::-1]]
            padding_ = [padding, padding[::-1]]
            conv_name = [f'conv{i}_1', f'conv{i}_2']
            for i_kernel, i_pad, i_conv in zip(kernel_size_, padding_,
                                               conv_name):
                # if i == 3:
                #     self.add_module(
                #         i_conv,
                #         # nn.Conv2d(
                #         #     channels,
                #         #     channels,
                #         #     tuple(i_kernel),
                #         #     padding=i_pad,
                #         #     groups=channels)
                #         DCNv2(
                #             in_channels=channels,
                #             out_channels=channels,
                #             kernel_size=tuple(i_kernel),
                #             padding=i_pad,
                #             deform_groups=channels,
                #         )
                #     )
                # else:
                self.add_module(
                    i_conv,
                    nn.Conv2d(
                        channels,
                        channels,
                        tuple(i_kernel),
                        padding=i_pad,
                        groups=channels)
                    # DCNv2(
                    #         in_channels=channels,
                    #         out_channels=channels,
                    #         kernel_size=tuple(i_kernel),
                    #         padding=i_pad,
                    #         deform_groups=channels,
                    # )
                )

        self.conv3 = nn.Conv2d(channels, channels, 1)
        self.conv3_1 = DCNv2(
            in_channels=channels,
            out_channels=channels,
            kernel_size=tuple([1, 3]),
            padding=[0, 1],
            deform_groups=channels,
        )
        self.conv3_2 = DCNv2(
            in_channels=channels,
            out_channels=channels,
            kernel_size=tuple([3, 1]),
            padding=[1, 0],
            deform_groups=channels,
        )
        # self.conv4_1 = DCNv2(
        #     in_channels=channels,
        #     out_channels=channels,
        #     kernel_size=tuple([1, 5]),
        #     padding=[0, 2],
        #     deform_groups=channels,
        # )
        # self.conv4_2 = DCNv2(
        #     in_channels=channels,
        #     out_channels=channels,
        #     kernel_size=tuple([5, 1]),
        #     padding=[2, 0],
        #     deform_groups=channels,
        # )
        self.pin = SpectralGatingNetwork(hidden_size=channels)

    def forward(self, x):
        """Forward function."""

        u = x.clone()

        attn = self.conv0(x)

        # Multi-Scale Feature extraction
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn_3 = self.conv3_1(attn)
        attn_3 = self.conv3_2(attn_3)
        # attn_4 = self.conv4_1(attn)
        # attn_4 = self.conv4_2(attn_4)
        branchs = attn_0 + attn_1 + attn_2 + attn_3
        # attn = attn + attn_0 + attn_1 + attn_2 + attn_3
        # Channel Mixing
        attn = self.pin(attn, branchs)
        attn = self.conv3(attn)

        # Convolutional Attention
        x = attn * u

        return x


class SpectralGatingNetwork(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1,
                 hidden_size_factor=1, alpha=0.2, squeeze_radio=2,
                 norm_cfg=dict(type='BN', requires_grad=True), ):
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
        self.w3 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w4 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b4 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x, branch, spatial_size=None):
        bias2 = branch.clone()
        dtype = branch.dtype
        branch = branch.float()
        B, C, H, W = x.shape
        branch = torch.fft.rfft2(branch, dim=(2, 3), norm="ortho")
        origin_ffted2 = branch
        # origin_ffted = x
        branch_imag = branch.imag
        branch_real = branch.real
        branch = torch.complex(branch_real, branch_imag)
        branch = branch.reshape(B, self.num_blocks, self.block_size, branch.shape[2], branch.shape[3])
        branch1_real = F.relu(
            torch.einsum('bkihw,kio->bkohw', branch.real, self.w3[0]) - \
            torch.einsum('bkihw,kio->bkohw', branch.imag, self.w3[1]) + \
            self.b3[0, :, :, None, None]
        )

        branch1_imag = F.relu(
            torch.einsum('bkihw,kio->bkohw', branch.imag, self.w3[0]) + \
            torch.einsum('bkihw,kio->bkohw', branch.real, self.w3[1]) + \
            self.b3[1, :, :, None, None]
        )
        branch2_real = (
                torch.einsum('bkihw,kio->bkohw', branch1_real, self.w4[0]) - \
                torch.einsum('bkihw,kio->bkohw', branch1_imag, self.w4[1]) + \
                self.b4[0, :, :, None, None]
        )

        branch2_imag = (
                torch.einsum('bkihw,kio->bkohw', branch1_imag, self.w4[0]) + \
                torch.einsum('bkihw,kio->bkohw', branch1_real, self.w4[1]) + \
                self.b4[1, :, :, None, None]
        )
        branch = torch.stack([branch2_real, branch2_imag], dim=-1)
        branch = F.softshrink(branch, lambd=self.sparsity_threshold)

        branch = torch.view_as_complex(branch)
        branch = branch.reshape(B, C, branch.shape[3], branch.shape[4])

        bias = x.clone()
        dtype = x.dtype
        x = x.float()

        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")
        origin_ffted = x
        # origin_ffted = x
        x_imag = x.imag
        x_real = x.real
        y = torch.cat([x_real, x_imag], dim=1)
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        x = torch.complex(y_real, y_imag)
        x = x.reshape(B, self.num_blocks, self.block_size, x.shape[2], x.shape[3])
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
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)

        x = torch.view_as_complex(x)
        x = x.reshape(B, C, x.shape[3], x.shape[4])

        x = x * branch
        # branch = branch * origin_ffted2
              
        branch = torch.fft.irfft2(branch, s=(H, W), dim=(2, 3), norm="ortho")
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        x = x.type(dtype)
        branch=branch.type(dtype)

        return x + bias+bias2


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
                 act_cfg=dict(type='GELU')):
        super().__init__()
        self.proj_1 = nn.Conv2d(in_channels, in_channels, 1)
        self.activation = build_activation_layer(act_cfg)
        self.spatial_gating_unit = MSCAAttention(in_channels,
                                                 attention_kernel_sizes,
                                                 attention_kernel_paddings)
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
                 norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, channels)[1]
        self.attn = MSCASpatialAttention(channels, attention_kernel_sizes,
                                         attention_kernel_paddings, act_cfg)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, channels)[1]
        mlp_hidden_channels = int(channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=channels,
            hidden_features=mlp_hidden_channels,
            act_cfg=act_cfg,
            drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((channels)),
            requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((channels)),
            requires_grad=True)

    def forward(self, x, H, W):
        """Forward function."""

        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
            self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
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

        # self.proj = nn.Conv2d(
        #     in_channels,
        #     embed_dim,
        #     kernel_size=patch_size,
        #     stride=stride,
        #     padding=patch_size // 2)
        self.proj = DCNv2(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
            deform_groups=2,
        )
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]

    def forward(self, x):
        """Forward function."""

        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)

        x = x.flatten(2).transpose(1, 2)

        return x, H, W


@BACKBONES.register_module()
class MSCC(BaseModule):
    """

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
                 num_stages=4,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 pretrained=None,
                 init_cfg=None):
        super(MSCC, self).__init__(init_cfg=init_cfg)

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
        # self.out_indice = out_indice
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0], norm_cfg=norm_cfg)
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=7 if i == 0 else 3,
                    stride=4 if i == 0 else 2,
                    in_channels=in_channels if i == 0 else embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                    norm_cfg=norm_cfg)

            block = nn.ModuleList([
                MSCABlock(
                    channels=embed_dims[i],
                    attention_kernel_sizes=attention_kernel_sizes,
                    attention_kernel_paddings=attention_kernel_paddings,
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg) for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

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
            super(MSCC, self).init_weights()

    def forward(self, x):
        """Forward function."""

        B = x.shape[0]
        outs = []
        count=0
        for i in range(self.num_stages):

            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            x, H, W = patch_embed(x)

            for blk in block:
                x = blk(x, H, W)
                # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


                # from tools.feature_visualization import draw_feature_map
                # draw_feature_map(x,count=count)
                # count+=1
                # x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            outs.append(x)


        # if self.out_indice != -1: return outs[self.out_indice]
        return outs
