
# !pip install -q -U albumentations --no-binary qudida,albumentations
# !echo "$(pip freeze | grep albumentations) is successfully installed"
import pickle
import numpy as np
from torch.utils .data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


from config import CFG
from augmix import *

def prepare_imgs_and_targets(data_dir, train=True):
    train_list = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
    test_list = ["test_batch"]
    imgs = []
    targets = []
    if train:
        downloaded_list = train_list
    else:
        downloaded_list = test_list
    for file_name in downloaded_list:
        file_path = os.path.join(data_dir,file_name)
        with open(file_path,'rb') as f:
            mydict = pickle.load(f,encoding='bytes')
            this_batch_img = mydict[b'data'].reshape((-1,3,32,32)).transpose(0,2,3,1)
            this_batch_targets = mydict[b'labels']
            if file_name == 'data_batch_1' or train==False:
                imgs = this_batch_img
                targets = this_batch_targets
            else:
                imgs = np.concatenate([imgs,this_batch_img],axis=0)
                targets = np.concatenate([targets,this_batch_targets],axis=0)
            if 'labels' in mydict:
                targets.append(mydict[b'labels'])

    return imgs, targets

stats = ((0.4914, 0.4822, 0.4465), (0.2022, 0.1994, 0.2009))


def transforms_train():
    return A.Compose([
        RandomAugMix(severity=CFG.augmix, width=CFG.augmix, alpha=CFG.augmix, p=1.),
        # A.RandomHorizontalFlip(),
        # ColorAugmentation(),
        A.Resize(CFG.image_size,CFG.image_size),
        A.Normalize(*stats),
        ToTensorV2(p=1.0),
    ])

def transforms_valid():
    return A.Compose([
        A.Resize(CFG.image_size,CFG.image_size),
        A.Normalize(*stats),
        ToTensorV2(p=1.0),
    ])

class MyDataset(Dataset):
    def __init__(self, imgs,targets, transform):
        self.imgs = imgs
        self.targets = targets
        self.transform=transform

    def __getitem__(self,index):
        img,target = self.imgs[index], self.targets[index]
        # img = Image.fromarray(img)
        # img = self.transform(img)
        if self.transform is not None:
            img = self.transform(image = img)['image']
        return img,target

    def __len__(self):
        return len(self.imgs)