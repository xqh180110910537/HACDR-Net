import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize


@HEADS.register_module()
class UenxtHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

    def forward(self, inputs):
        # inputs = self._transform_inputs(inputs)
        # print(inputs[0].shape)
        # outs = []
        # for idx in range(len(inputs)):
        #     x = inputs[idx]
        #     conv = self.convs[idx]
        #     outs.append(
        #         resize(
        #             input=conv(x),
        #             size=inputs[0].shape[2:],
        #             mode=self.interpolate_mode,
        #             align_corners=self.align_corners))
        #
        # out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(inputs)

        return out
