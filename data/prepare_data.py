"""docstring for prepare_data"""
import os
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .augmix import RandomAugMix


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


stats = ((0.4914, 0.4822, 0.4465), (0.2022, 0.1994, 0.2009))


def transforms_train(image_size, params):
    """_summary_

    Returns:
        _type_: _description_
    """
    return A.Compose(
        [
            RandomAugMix(severity=params, width=params, alpha=1, p=1.0),
            A.Resize(image_size, image_size),
            A.Normalize(*stats),
            ToTensorV2(p=1.0),
        ]
    )


def transforms_valid(image_size):
    """_summary_

    Returns:
        _type_: _description_
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(*stats),
            ToTensorV2(p=1.0),
        ]
    )


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
