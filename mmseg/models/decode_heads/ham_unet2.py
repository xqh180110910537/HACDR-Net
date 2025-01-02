# Copyright (c) OpenMMLab. All rights reserved.
# Originally from https://github.com/visual-attention-network/segnext
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
import torch
from torch import nn
import torch.nn.functional as F


# Taken from https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/14
class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


class SpaceToDepth(nn.Module):
    # Expects the following shape: Batch, Channel, Height, Width
    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_size, padding=0, kernels_per_layer=1):
        super(DepthwiseSeparableConv, self).__init__()
        # In Tensorflow DepthwiseConv2D has depth_multiplier instead of kernels_per_layer
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * kernels_per_layer,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels * kernels_per_layer, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DoubleDense(nn.Module):
    def __init__(self, in_channels, hidden_neurons, output_channels):
        super(DoubleDense, self).__init__()
        self.dense1 = nn.Linear(in_channels, out_features=hidden_neurons)
        self.dense2 = nn.Linear(in_features=hidden_neurons, out_features=hidden_neurons // 2)
        self.dense3 = nn.Linear(in_features=hidden_neurons // 2, out_features=output_channels)

    def forward(self, x):
        out = F.relu(self.dense1(x.view(x.size(0), -1)))
        out = F.relu(self.dense2(out))
        out = self.dense3(out)
        return out


class DoubleDSConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_ds_conv = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_ds_conv(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttention(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        #  https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
        #  uses Convolutions instead of Linear
        self.MLP = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, input_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_channels // reduction_ratio, input_channels),
        )

    def forward(self, x):
        # Take the input and apply average and max pooling
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP(avg_values) + self.MLP(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        return scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.bn(out)
        scale = x * torch.sigmoid(out)
        return scale


class CBAM(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(input_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        out = self.channel_att(x)
        out = self.spatial_att(out)
        return out

class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x, x_concat=None):
        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x


@HEADS.register_module()
class UnetHamHead2(BaseDecodeHead):
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
        super(UnetHamHead2, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.ham_channels = ham_channels

        self.align = ConvModule(
            self.ham_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # self.decoder1 = DecoderBottleneck(self.in_channels[4] + self.in_channels[3], self.in_channels[3])
        self.decoder2 = DecoderBottleneck(self.in_channels[3] + self.in_channels[2], self.in_channels[2])
        self.decoder3 = DecoderBottleneck(self.in_channels[2] + self.in_channels[1], self.in_channels[1])
        self.decoder4 = DecoderBottleneck(self.in_channels[1] + self.in_channels[0], self.in_channels[0])
        # self.decoder5 = DecoderBottleneck(self.in_channels[3] + self.in_channels[2], self.in_channels[2])
        # self.decoder6 = DecoderBottleneck(self.in_channels[2] + self.in_channels[1], self.in_channels[1])
        # self.decoder7 = DecoderBottleneck(self.in_channels[1] + self.in_channels[0], self.in_channels[0])
        # self.cbam1 = CBAM(self.in_channels[3], reduction_ratio=reduction_ratio)
        # self.cbam2 = CBAM(self.in_channels[2], reduction_ratio=reduction_ratio)
        # self.cbam3 = CBAM(self.in_channels[1], reduction_ratio=reduction_ratio)

    def forward(self, inputs):
        """Forward function."""
        # inputs[7] = resize(inputs[7], size=inputs[6].shape[2:],
        #                    mode='bilinear',
        #                    align_corners=self.align_corners)
        # # inputs[3]=self.cbam1( inputs[3])
        # inputs[6] = self.decoder5(inputs[7], inputs[6])
        # inputs[6] = resize(inputs[6], size=inputs[5].shape[2:],
        #                    mode='bilinear',
        #                    align_corners=self.align_corners)
        # # inputs[2] = self.cbam2(inputs[2])
        # inputs[5] = self.decoder6(inputs[6], inputs[5])
        # # print( inputs[1].shape)
        # inputs[5] = resize(inputs[5], size=inputs[4].shape[2:],
        #                    mode='bilinear',
        #                    align_corners=self.align_corners)
        # # inputs[1] = self.cbam3(inputs[1])
        # inputs[4] = self.decoder7(inputs[5], inputs[4])
        # y=inputs[4].clone()
        # outs = []
        # for i in range(4, 8):
        #     patch1 = self._transform_inputs(inputs[4])
        #     patch1 = self._transform_inputs(patch1)
        #     patch1[3] = resize(patch1[3], size=patch1[2].shape[2:],
        #                        mode='bilinear',
        #                        align_corners=self.align_corners)
        #     # inputs[3]=self.cbam1( inputs[3])
        #     patch1[2] = self.decoder2(patch1[3], patch1[2])
        #     patch1[2] = resize(patch1[2], size=patch1[1].shape[2:],
        #                        mode='bilinear',
        #                        align_corners=self.align_corners)
        #     # inputs[2] = self.cbam2(inputs[2])
        #     patch1[1] = self.decoder3(patch1[2], patch1[1])
        #     # print( inputs[1].shape)
        #     patch1[1] = resize(patch1[1], size=patch1[0].shape[2:],
        #                        mode='bilinear',
        #                        align_corners=self.align_corners)
        #     # inputs[1] = self.cbam3(inputs[1])
        #     patch1[0] = self.decoder4(patch1[1], patch1[0])
        # inputs = torch.cat(inputs, dim=1)
        ## apply a conv block to squeeze feature map
        # x = self.squeeze(inputs)
        # # apply hamburger module
        # x = self.hamburger(x)

        # apply a conv block to align feature map
        #     patch1 = self.align(patch1[0])
        #
        #     outs.append(patch1)
        # temp1 = torch.cat([outs[0], outs[1]], 3)
        # temp2 = torch.cat([outs[2], outs[3]], 3)
        # outs = torch.cat([temp1, temp2], 2)

        inputs = self._transform_inputs(inputs)

        # inputs[3]+=inputs[7]
        # inputs[2] += inputs[6]
        # inputs[1] += inputs[5]
        # inputs[0] += inputs[4]
        inputs[3] = resize(inputs[3], size=inputs[2].shape[2:],
                           mode='bilinear',
                           align_corners=self.align_corners)
        # from tools.vis_cam import draw_feature_map
        # draw_feature_map(inputs[3])
        # inputs[3]=self.cbam1( inputs[3])
        inputs[2] = self.decoder2(inputs[3], inputs[2])
        inputs[2] = resize(inputs[2], size=inputs[1].shape[2:],
                           mode='bilinear',
                           align_corners=self.align_corners)

        # inputs[2] = self.cbam2(inputs[2])
        inputs[1] = self.decoder3(inputs[2], inputs[1])
        # print( inputs[1].shape)
        inputs[1] = resize(inputs[1], size=inputs[0].shape[2:],
                           mode='bilinear',
                           align_corners=self.align_corners)
        # from tools.vis_cam import draw_feature_map

        # inputs[1] = self.cbam3(inputs[1])

        inputs[0] = self.decoder4(inputs[1], inputs[0])

        # inputs = torch.cat(inputs, dim=1)
        ## apply a conv block to squeeze feature map
        # x = self.squeeze(inputs)
        # # apply hamburger module
        # x = self.hamburger(x)

        # apply a conv block to align feature map
        # output = torch.cat([self.align(inputs[0]),outs],dim=3)
        output = self.align(inputs[0])

        # print(output.shape)
        # from tools.vis_cam import Tsne
        # Tsne(output)
        output = self.cls_seg(output)


        return output
