"""import modules for vgg"""
# pylint: disable=W0221, W0613, W0201, E1101, C0301
from typing import Union, List, Dict, cast, Optional
import torch
from torch import nn

# from https://arxiv.org/pdf/1409.1556.pdf
# https://deep-learning-study.tistory.com/521
# VGG type dict
# int : output chnnels after conv layer
# 'M' : max pooling layer
# VGG11 : A / VGG13 : B / VGG16 : D / VGG19 : E
# https://github.com/aaron-xichen/pytorch-playground/blob/5d476295f1c1564c8074eb885bb6138d6bd09889/imagenet/vgg.py#L20
cfgs: Dict[str, List[Union[str, int]]] = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    """VGGNet base"""

    def __init__(
        self,
        model_name: str = "VGG19",
        num_classes: int = 10,
        init_weight: bool = True,
        dropout_rate: float = 0.5,
        batch_norm: Optional[bool] = True,
        in_cfgs=cfgs.copy(),
    ) -> None:
        super().__init__()
        self.in_cfgs = in_cfgs
        self.conv_layers = self._make_layers(
            self.in_cfgs[model_name], batch_norm=batch_norm
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, num_classes),
        )
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, first: torch.Tensor) -> torch.Tensor:
        """
        pytorch-lightning team recommend to use forward for only inference(predict)
        depart last linear to checkpoint deepspeed
        activations are deleted afther use, and re-calculated during the backward pass
        """
        x_1 = self.conv_layers(first)
        x_2 = self.avgpool(x_1)
        x_3 = torch.flatten(x_2, 1)
        out = self.classifier(x_3)
        return out

    def _make_layers(
        self,
        arch,
        batch_norm: bool = False,
    ) -> nn.Sequential:
        """function to make layers according to VGG type
        if batch_norm is True, add batch normalization

        Args:
            arch (List[nn.Module]): list of layers from dict 'cfgs'
        Returns:
            nn.Sequential: sequential layers
        """
        layers: List[nn.Module] = []
        in_channels = 3
        for layer in arch:
            if isinstance(layer, int):
                out_channels = cast(int, layer)
                conv2d = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
                if batch_norm:
                    layers += [
                        conv2d,
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = out_channels
            elif layer == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)
