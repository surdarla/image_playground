"""modules for pytroch-lighting cifar10"""
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from config import CFG
from utils import mysplit
from data.prepare_data import prepare_imgs_and_targets, MyDataset


class CIFAR10Data(pl.LightningDataModule):
    """_summary_

    Args:
        pl (_type_): _description_ out
    """

    def __init__(
        self,
        root: str,
        batch_size: int,
        this_fold: int,
        transform: Optional[callable] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.this_fold = this_fold
        self.transform = transform

    def prepare_data(self) -> None:
        pass  # no need to download cifar10

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            X, y = prepare_imgs_and_targets(self.root, train=True)
            img_train, img_valid, target_train, target_valid = mysplit(
                X, y, self.this_fold, CFG.n_split, CFG.seed
            )
            self.trainset = MyDataset(
                img_train, target_train, transform=self.transform()
            )
            self.validset = MyDataset(
                img_valid, target_valid, transform=self.transform()
            )
        if stage == "test":
            X, y = prepare_imgs_and_targets(self.root, train=False)
            self.testset = MyDataset(X, y, transform=self.transform())

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validset,
            batch_size=10 * self.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.testset,
            batch_size=10 * self.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            drop_last=False,
        )
