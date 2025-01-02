import warnings
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import torch.nn as nn

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm


# from ..builder import BACKBONES
# from ..utils import InvertedResidual, make_divisible
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, head_num):
        super().__init__()

        self.head_num = head_num
        self.dk = (embedding_dim // head_num) ** (1 / 2)

        self.qkv_layer = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)
        self.out_attention = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x, mask=None):
        qkv = self.qkv_layer(x)

        query, key, value = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.head_num))
        energy = torch.einsum("... i d , ... j d -> ... i j", query, key) * self.dk

        if mask is not None:
            energy = energy.masked_fill(mask, -np.inf)

        attention = torch.softmax(energy, dim=-1)

        x = torch.einsum("... i j , ... j d -> ... i d", attention, value)

        x = rearrange(x, "b h t d -> b t (h d)")
        x = self.out_attention(x)

        return x


class MLP(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()

        self.mlp_layers = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = self.mlp_layers(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, head_num)
        self.mlp = MLP(embedding_dim, mlp_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        _x = self.multi_head_attention(x)
        _x = self.dropout(_x)
        x = x + _x
        x = self.layer_norm1(x)

        _x = self.mlp(x)
        x = x + _x
        x = self.layer_norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, head_num, mlp_dim, block_num=12):
        super().__init__()

        self.layer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(embedding_dim, head_num, mlp_dim) for _ in range(block_num)])

    def forward(self, x):
        for layer_block in self.layer_blocks:
            x = layer_block(x)

        return x


class ViT(nn.Module):
    def __init__(self, img_dim, in_channels, embedding_dim, head_num, mlp_dim,
                 block_num, patch_dim):
        super().__init__()

        self.patch_dim = patch_dim
        self.num_tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)

        self.projection = nn.Linear(self.token_dim, embedding_dim)
        self.embedding = nn.Parameter(torch.rand(self.num_tokens + 1, embedding_dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        self.dropout = nn.Dropout(0.1)

        self.transformer = TransformerEncoder(embedding_dim, head_num, mlp_dim, block_num)

        # if self.classification:
        #     self.mlp_head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        img_patches = rearrange(x,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.patch_dim, patch_y=self.patch_dim)

        batch_size, tokens, _ = img_patches.shape

        project = self.projection(img_patches)
        token = repeat(self.cls_token, 'b ... -> (b batch_size) ...',
                       batch_size=batch_size)

        patches = torch.cat([token, project], dim=1)
        patches += self.embedding[:tokens + 1, :]

        x = self.dropout(patches)
        x = self.transformer(x)
        x = x[:, 1:, :]

        return x


# if __name__ == '__main__':
#     vit = ViT(img_dim=32,
#               in_channels=1024,
#               patch_dim=1,
#               embedding_dim=768,
#               block_num=12,
#               head_num=4,
#               mlp_dim=3072,)
#     print(sum(p.numel() for p in vit.parameters()))
#     print(vit(torch.rand(1, 1024, 32, 32)).shape)
# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer


from ..builder import BACKBONES
from ..utils import ResLayer
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNetV1d
from .vit import VisionTransformer




class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        width = int(out_channels * (base_width / 64))

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, groups=1, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = x + x_down
        x = self.relu(x)

        return x


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        self.vit_img_dim = img_dim // patch_dim
        # self.vit=VisionTransformer()
        self.vit = ViT(self.vit_img_dim, out_channels * 8, out_channels * 8,
                       head_num, mlp_dim, block_num, patch_dim=1)
        print(self.vit_img_dim, out_channels * 8, out_channels * 8,
              head_num, mlp_dim, block_num, )
        self.conv2 = nn.Conv2d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x1 = self.relu(x)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x = self.encoder3(x3)
        print(x.shape)
        x = self.vit(x)
        print(x.shape)
        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        print(x.shape, x1.shape, x2.shape, x3.shape)
        return x, x1, x2, x3



import torch


@BACKBONES.register_module()
class resnet50_vit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mlp_ratio,
                 num_layers,
                 num_heads,
                 embed_dims,
                 qkv_bias,
                 norm_cfg,
                 act_cfg,
                 conv_cfg):
        super().__init__()
        self.in_channels = in_channels
        self.resnet = ResNetV1d(depth=50, in_channels=self.in_channels,
                                init_cfg=dict(
                                    type='Pretrained', checkpoint='./pth/resnest50_imagenet_converted-1ebf0afe.pth'
                                    )
                                )
        self.conv0 = build_conv_layer(
            conv_cfg,
            64,
            64 * 2,
            kernel_size=1,
            bias=False)
        self.vit = VisionTransformer(img_size=64,
                                     patch_size=1,
                                     in_channels=1024,
                                     embed_dims=embed_dims,
                                     num_layers=num_layers,
                                     num_heads=num_heads,
                                     mlp_ratio=mlp_ratio,
                                     out_indices=-1,
                                     qkv_bias=True,
                                     drop_rate=0.,
                                     attn_drop_rate=0.,
                                     drop_path_rate=0.,
                                     with_cls_token=True,
                                     output_cls_token=False,
                                     norm_cfg=dict(type='LN'),
                                     act_cfg=dict(type='GELU'),
                                     patch_norm=False,
                                     final_norm=False,
                                     interpolate_mode='bicubic',
                                     num_fcs=2,
                                     norm_eval=False,
                                     with_cp=False,
                                     init_cfg=dict(
                                         type='Pretrained',checkpoint='./pth/vit-base-p16_in21k-pre-3rdparty_ft'
                                                                      '-64xb64_in1k-384_20210928-98e8652b.pth',
                                     ))
        self.conv2 = build_conv_layer(
            conv_cfg,
            embed_dims,
            out_channels[3],
            kernel_size=1,
            bias=False)

    def forward(self, x):
        x, x1, x2, x3, x4 = self.resnet(x)
        x = self.conv0(x)
        x3 = self.vit(x3)[0]
        x3 = self.conv2(x3)
        outs= []
        outs.append(x)
        outs.append(x1)
        outs.append(x2)
        outs.append(x3)
        return outs


# model = resnet50_vit(in_channels=3,
#                      out_channels=[128, 256, 512, 512],
#                      embed_dims=768,
#                      num_layers=12,
#                      num_heads=12,
#                      mlp_ratio=4,
#                      qkv_bias=True,
#                      norm_cfg=dict(type='LN'),
#                      act_cfg=dict(type='GELU'),
#                      conv_cfg=None,
#                      )
# inputs = torch.rand(1, 3, 512, 512)
# level_outputs = model(inputs)
#
# for level_out in level_outputs:
#     print(tuple(level_out.shape))
