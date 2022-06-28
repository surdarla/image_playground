"""import modules for vgg"""
# pylint: disable=W0221, W0613, W0201, E1101
from typing import Union, List, Dict, cast, Optional
import torchmetrics
import torch
from torch import nn
from torch.nn import functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl

# from https://arxiv.org/pdf/1409.1556.pdf
# https://deep-learning-study.tistory.com/521
# VGG type dict
# int : output chnnels after conv layer
# 'M' : max pooling layer
# VGG11 : A / VGG13 : B / VGG16 : D / VGG19 : E

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


class VGG(pl.LightningModule):
    """VGGNet"""

    def __init__(
        self,
        model: str = "VGG19",
        num_classes: int = 10,
        init_weight: bool = True,
        dropout: float = 0.5,
        batch_norm: Optional[bool] = True,
        learning_rate: Optional[float] = 0.01,
    ) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.top1 = torchmetrics.Accuracy()
        self.top5 = torchmetrics.Accuracy(top_k=5)

        self.conv_layers = self.make_layers(cfgs[model], batch_norm=batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
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

    def make_layers(
        self,
        arch,
        batch_norm: bool = False,
    ) -> nn.Sequential:
        """_summary_

        Args:
            arch (_type_): _description_
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """pytorch-lightning team recommend to use forward for only inference(predict)"""
        x_1 = self.conv_layers(x)
        x_2 = self.avgpool(x_1)
        x_3 = torch.flatten(x_2, 1)
        out = self.classifier(x_3)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4
        )
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5)
        return {optimizer: optimizer, lr_scheduler: lr_scheduler}
        # return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        loss = self.loss(logits, targets)
        acc1 = self.top1(logits, targets)
        acc5 = self.top5(logits, targets)
        self.log("TRAIN LOSS", loss)
        self.log("TRAIN TOP1 ACC", acc1)
        self.log("TRAIN TOP5 ACC", acc5)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "VALID")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "TEST")

    def _shared_eval(self, batch, batch_idx, prefix):
        images, targets = batch
        logits = self(images)
        loss = self.loss(logits, targets)
        acc1 = self.top1(logits, targets)
        acc5 = self.top5(logits, targets)
        self.log_dict(
            {
                f"{prefix} LOSS": loss,
                f"{prefix} TOP1 ACC": acc1,
                f"{prefix} TOP5 ACC": acc5,
            }
        )
        return loss
