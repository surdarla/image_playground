"""import libraries for model module"""
# pylint: disable=W0221, W0613, W0201, E1101, C0301

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
import torchmetrics
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


class MyModule(pl.LightningModule):
    """lightning module"""

    def __init__(
        self,
        model,
        args: dict,
    ) -> None:
        super().__init__()
        self.model = model
        self.args = args
        self.learning_rate = args.lr
        self.save_hyperparameters()
        self.loss = nn.CrossEntropyLoss()
        self.top1 = torchmetrics.Accuracy()
        self.top5 = torchmetrics.Accuracy(top_k=5)

    def forward(self, first: torch.Tensor) -> torch.Tensor:
        """only for predict and inference"""
        return self.model(first)

    def configure_optimizers(self):
        optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.learning_rate)
        # optimizer = FusedAdam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = ReduceLROnPlateau(
            optimizer, factor=0.1, patience=self.args.patience
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "VALID LOSS",
        }

    def training_step(self, batch, batch_idx):
        images, targets = batch
        logits = self.forward(images)
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
        logits = self.forward(images)
        loss = self.loss(logits, targets)
        self.acc1 = self.top1(logits, targets)
        self.acc5 = self.top5(logits, targets)
        self.log_dict(
            {
                f"{prefix} LOSS": loss,
                f"{prefix} TOP1 ACC": self.acc1,
                f"{prefix} TOP5 ACC": self.acc5,
            }
        )
        return loss

    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(
    #         self.parameters(),
    #         lr=self.hparams.lr,
    #         momentum=0.9,
    #         weight_decay=5e-4,
    #     )
    #     steps_per_epoch = 45000 // self.batch_size
    #     scheduler_dict = {
    #         "scheduler": OneCycleLR(
    #             optimizer,
    #             0.1,
    #             epochs=self.trainer.max_epochs,
    #             steps_per_epoch=steps_per_epoch,
    #         ),
    #         "interval": "step",
    #     }
    #     return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
