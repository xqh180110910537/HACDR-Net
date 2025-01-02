from mmcv.cnn import ConvModule
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
import torch
from torch import nn


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
class UnetHead(BaseDecodeHead):
    """
    Args:
        last_channels (int): input channels.
            Defaults: 512.
        kwargs (int): kwagrs for Ham. Defaults: dict().
    """

    def __init__(self, last_channels=512,**kwargs):
        super(UnetHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.last_channels = last_channels

        self.align = ConvModule(
            self.last_channels,
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # self.decoder1 = DecoderBottleneck(self.in_channels[4] + self.in_channels[3], self.in_channels[3])
        self.decoder2 = DecoderBottleneck(self.in_channels[3] + self.in_channels[2], self.in_channels[2])
        self.decoder3 = DecoderBottleneck(self.in_channels[2] + self.in_channels[1], self.in_channels[1])
        self.decoder4 = DecoderBottleneck(self.in_channels[1] + self.in_channels[0], self.in_channels[0])

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)
        inputs[3] = resize(inputs[3], size=inputs[2].shape[2:],
                           mode='bilinear',
                           align_corners=self.align_corners)
        inputs[2] = self.decoder2(inputs[3], inputs[2])
        inputs[2] = resize(inputs[2], size=inputs[1].shape[2:],
                           mode='bilinear',
                           align_corners=self.align_corners)

        inputs[1] = self.decoder3(inputs[2], inputs[1])

        inputs[1] = resize(inputs[1], size=inputs[0].shape[2:],
                           mode='bilinear',
                           align_corners=self.align_corners)
        inputs[0] = self.decoder4(inputs[1], inputs[0])

        output = self.align(inputs[0])

        output = self.cls_seg(output)


        return output
