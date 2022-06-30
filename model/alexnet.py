"""Alexnet module import from torchvision.models.alexnet"""
# pylint: disable=E1101, W0221, W0613
import torch
from torch import nn


class AlexNet(nn.Module):
    """Basic AlexNet

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x_input (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        features = self.feature_extractor(x_input)
        avgpooled_features = self.avgpool(features)
        flat = torch.flatten(avgpooled_features, 1)
        out = self.classifier(flat)
        return out
