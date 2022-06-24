"""Alexnet module import from torchvision.models.alexnet"""
import torch
import torchmetrics
from torch import nn
import pytorch_lightning as pl


class AlexNet(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        num_classes: int = 1000,
        drouput_rate: float = 0.5,
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
            nn.Dropout(p=drouput_rate),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drouput_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x_input (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        features = self.feature_extractor(x_input)
        # viewed_features = features.view(features.size(0), -1)
        # out = self.classifier(viewed_features)
        # return out, viewed_features
        avgpooled_features = self.avgpool(features)
        flat = torch.nn.Flatten(avgpooled_features, 1)
        out = self.classifier(flat)
        return out


class AlexNetLit(pl.LightningModule):
    """_summary_

    Args:
        pl (_type_): _description_
    """

    def __init__(self, num_classes: int, lr: float) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model = AlexNet(self.num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, x_input):
        return self.model(x_input)

    def configure_optimizers(self):
        print(self.hparams["lr"])
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

    def training_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, targets)
        acc = self.train_acc(logits, targets)
        self.log("TRAIN LOSS", loss, on_epoch=True)
        self.log("TRAIN ACC", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, targets)
        acc = self.valid_acc(logits, targets)
        self.log("VALID LOSS", loss, on_step=False, on_epoch=True)
        self.log("VALID ACC", acc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, targets)
        acc = self.valid_acc(logits, targets)
        self.log("TEST LOSS", loss, on_step=False, on_epoch=True)
        self.log("TEST ACC", acc, on_step=False, on_epoch=True)
        return loss
