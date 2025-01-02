import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from ..builder import HEADS
from .decode_head import BaseDecodeHead
# from mmseg.models.builder import HEADS
# from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)

        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)

        x = self.layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))

        # self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        # x = self.conv1(x)

        return x


@HEADS.register_module()
class UnetHead(BaseDecodeHead):

    def __init__(self, interpolate_mode='bilinear',in_channels = [128, 256, 512, 1024], out_channels=128, **kwargs):
        super().__init__(input_transform='multiple_select',in_channels = [128, 256, 512, 1024],**kwargs)


        self.in_channels=in_channels
        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        self.conv11=nn.Conv2d(in_channels=in_channels[3]+in_channels[2],out_channels=in_channels[2],kernel_size=1, stride=1, bias=False)#特征4转接
        assert num_inputs == len(self.in_index)
        self.decoder1 = DecoderBottleneck(768,256)
        self.decoder2 = DecoderBottleneck(256+128, 128)
        self.decoder2 = DecoderBottleneck(64 + 128, 64)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        x0,x,x1,x2,x3=inputs[0],inputs[1],inputs[2],inputs[3],inputs[0]
        x2=torch.cat([x2,x3],dim=1)
        x2=self.conv11(x2)
        x1=self.decoder1(x2,x1)
        x = self.decoder2(x1, x)
        out=self.cls_seg(x)
        return out





