import torch
import torch.nn as nn

from .blocks import darknet_conv

class aspp_decoder(nn.Module):
    def __init__(self, planes, hidden_planes, out_planes):
        super().__init__()
        self.conv0 = darknet_conv(planes, hidden_planes, ksize=1, stride=1)
        self.conv1 = darknet_conv(planes, hidden_planes, ksize=3, stride=1,dilation_rate=6)
        self.conv2 = darknet_conv(planes, hidden_planes, ksize=3, stride=1,dilation_rate=12)
        self.conv3 = darknet_conv(planes, hidden_planes, ksize=3, stride=1,dilation_rate=18)
        self.conv4 = darknet_conv(planes, hidden_planes, ksize=1, stride=1)
        self.pool=nn.AdaptiveAvgPool2d(1)

        self.low_feature = darknet_conv(512, 256, 1)
        self.conv_1 = nn.Conv2d(hidden_planes * 5, 256, 1)
        self.conv_2 = nn.Conv2d(512, 1, 3, padding=1)

        self.upsample_2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample_4 = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, input):
        mid = input[0]
        bot = input[1]

        _, _, h, w = mid.size()
        b0 = self.conv0(mid)
        b1 = self.conv1(mid)
        b2 = self.conv2(mid)
        b3 = self.conv3(mid)
        b4 = self.conv4(self.pool(mid)).repeat(1, 1, h, w)
        mid = torch.cat([b0, b1, b2, b3, b4], 1)

        mid = self.conv_1(mid)
        mid = self.upsample_2(mid)
        low_feature = self.low_feature(bot)
        bot = torch.cat([mid, low_feature], 1)
        bot = self.conv_2(bot)

        return bot