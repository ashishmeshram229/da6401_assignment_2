from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout


# VGG11 input size is fixed to 224x224 as per the original paper
IMAGE_SIZE = 224


def _conv_bn_relu(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11Encoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super(VGG11Encoder, self).__init__()

        # Block 1: 1 conv, out 64ch, 224->112
        self.block1 = nn.Sequential(
            _conv_bn_relu(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 1 conv, out 128ch, 112->56
        self.block2 = nn.Sequential(
            _conv_bn_relu(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 2 convs, out 256ch, 56->28
        self.block3 = nn.Sequential(
            _conv_bn_relu(128, 256),
            _conv_bn_relu(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: 2 convs, out 512ch, 28->14
        self.block4 = nn.Sequential(
            _conv_bn_relu(256, 512),
            _conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 5: 2 convs, out 512ch, 14->7
        self.block5 = nn.Sequential(
            _conv_bn_relu(512, 512),
            _conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        f5 = self.block5(f4)

        if return_features:
            features = {
                "block1": f1,
                "block2": f2,
                "block3": f3,
                "block4": f4,
                "block5": f5,
            }
            return f5, features

        return f5


# Alias so autograder can do: from models.vgg11 import VGG11
VGG11 = VGG11Encoder
