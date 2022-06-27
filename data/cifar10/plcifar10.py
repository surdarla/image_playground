"""modules for pytroch-lighting cifar10"""
import os
import pickle
from typing import Optional
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
import numpy as np

from config import CFG
from utils import mysplit


def prepare_imgs_and_targets(data_dir, train=True):
    """_summary_

    Args:
        data_dir (str): path of data directory
        train (bool, optional): whether to collect data for train or not. Defaults to True.

    Returns:
        imgs, traget labels(10classes for cifar10)
    """
    train_list = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
    ]
    test_list = ["test_batch"]
    imgs = []
    targets = []
    if train:
        downloaded_list = train_list
    else:
        downloaded_list = test_list
    for file_name in downloaded_list:
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, "rb") as file:
            mydict = pickle.load(file, encoding="bytes")
            this_batch_img = (
                mydict[b"data"].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
            )
            this_batch_targets = mydict[b"labels"]
            if file_name == "data_batch_1" or train is False:
                imgs = this_batch_img
                targets = this_batch_targets
            else:
                imgs = np.concatenate([imgs, this_batch_img], axis=0)
                targets = np.concatenate([targets, this_batch_targets], axis=0)

    return imgs, targets


class MyDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, imgs, targets, transform):
        self.imgs = imgs
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.imgs[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        # if self.transform is not None: ## when albumentation
        #     img = self.transform(image=img)["image"]
        return img, target

    def __len__(self):
        return len(self.imgs)


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
        pass  # already downloaded

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            image, target = prepare_imgs_and_targets(self.root, train=True)
            img_train, img_valid, target_train, target_valid = mysplit(
                image, target, self.this_fold, CFG.n_split, CFG.seed
            )
            self.trainset = MyDataset(
                img_train,
                target_train,
                transform=transforms.Compose(
                    [
                        torchvision.transforms.RandomCrop(32, padding=4),
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.ToTensor(),
                        # cifar10_normalization(),
                    ]
                ),
            )
            self.validset = MyDataset(
                img_valid,
                target_valid,
                transform=transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        # cifar10_normalization(),
                    ]
                ),
            )
        if stage == "test":
            image, target = prepare_imgs_and_targets(self.root, train=False)
            self.testset = MyDataset(
                image,
                target,
                transform=transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        # cifar10_normalization(),
                    ]
                ),
            )

    def train_dataloader(self) -> DataLoader:
        cifar_train = DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            drop_last=False,
        )
        return cifar_train

    def val_dataloader(self) -> DataLoader:
        cifar_valid = DataLoader(
            self.validset,
            batch_size=10 * self.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            drop_last=False,
        )
        return cifar_valid

    def test_dataloader(self) -> DataLoader:
        cifar_test = DataLoader(
            self.testset,
            batch_size=10 * self.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            drop_last=False,
        )
        return cifar_test
