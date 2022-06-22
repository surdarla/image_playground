import os, gc

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW

from config import CFG
from utils import *
from prepare_data import (
    prepare_imgs_and_targets,
    MyDataset,
    transforms_train,
    transforms_valid,
)
from epoch_fn import train_one_epoch, valid_one_epoch
from model.myfish import Myfish

seed_everything(CFG.seed)
# device 설정
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

X, y = prepare_imgs_and_targets(CFG.data_dir, train=True)
print("Train starts")
# from pprint import pprint
# pprint(CFG.__dict__)
for this_fold in range(CFG.fold):
    print(f"=================={this_fold+1} fold starting===================")
    img_train, img_valid, target_train, target_valid = mysplit(
        X, y, this_fold, CFG.n_split, CFG.seed
    )
    trainset = MyDataset(img_train, target_train, transform=transforms_train())
    validset = MyDataset(img_valid, target_valid, transform=transforms_valid())
    train_loader = DataLoader(
        trainset,
        batch_size=CFG.batch_size,
        shuffle=True,
        pin_memory=use_cuda,
        num_workers=CFG.num_workers,
        drop_last=False,
    )
    valid_loader = DataLoader(
        validset,
        batch_size=CFG.batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=CFG.num_workers,
        drop_last=False,
    )

    model = Myfish().to(device)
    # model = ResNet9(3,10).to(device)

    scaler = GradScaler(enabled=CFG.amp)
    optimizer = AdamW(model.parameters(), lr=CFG.max_lr, weight_decay=CFG.weight_decay)

    total_steps = len(trainset) // CFG.batch_size * CFG.epochs
    my_scheduler_dict = dict(
        first_cycle_steps=total_steps,
        cycle_mult=0.75,
        max_lr=CFG.max_lr,
        min_lr=CFG.min_lr,
        warmup_steps=len(trainset) // CFG.batch_size * 3,
        gamma=0.75,
    )
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, CFG.max_lr, epochs=CFG.epochs, steps_per_epoch=len(train_loader))
    scheduler = CosineAnnealingWarmupRestarts(optimizer, **my_scheduler_dict)
    loss_fn = nn.CrossEntropyLoss()
    early_stopper = EarlyStopper(CFG.patience)

    for epoch in range(CFG.epochs):
        torch.cuda.empty_cache()
        start_time = time.time()
        # scheduler.step()
        avg_loss = train_one_epoch(
            epoch,
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            scaler,
            scheduler=scheduler,
        )
        with torch.no_grad():
            avg_val_loss, avg_val_score = valid_one_epoch(
                epoch, model, valid_loader, loss_fn, device
            )

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        print(f"Epoch {epoch+1} - Score: {avg_val_score:.4f}")

        early_stopper.check_early_stopping(avg_val_score)
        if early_stopper.save_model == True:
            dic = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(dic, CFG.pth_dir + f"/{CFG.model}_{this_fold}_best_.pth")
            print("save_model")

        if early_stopper.stop:
            break

    os.rename(
        CFG.pth_dir + f"/{CFG.model}_{this_fold}_best_.pth",
        CFG.pth_dir + f"/{CFG.model}_{this_fold}_best_{early_stopper.best_acc:.4f}.pth",
    )

    # del model, optimizer, train_loader, valid_loader, scaler, scheduler
    # torch.cuda.empty_cache()
    # gc.collect()
