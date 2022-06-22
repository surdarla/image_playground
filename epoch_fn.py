"""docstring"""
import time
import numpy as np
from tqdm.auto import tqdm
import torch
from torch.cuda.amp import autocast

from config import CFG
from utils import AverageMeter, timeSince, accuracy2


def train_one_epoch(
    epoch, model, train_loader, criterion, optimizer, device, scaler, scheduler=None
):
    """docstring for train_one_epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    start = time.time()

    for step, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        batch_size = targets.size(0)
        with autocast(enabled=CFG.amp):
            out = model(images)
        loss = criterion(out, targets)
        acc1 = accuracy2(out, targets)

        if CFG.accum_iter > 1:
            loss = loss / CFG.accum_iter
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        scaler.scale(loss).backward()

        if CFG.max_grad_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), CFG.max_grad_norm
            )

        if (step + 1) % CFG.accum_iter == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        scheduler.step()

        if step % CFG.print_freq == 0 or step == (len(train_loader)):
            print(
                "Epoch: [{0}/{1}][{2}/{3}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "Grad: {grad_norm:.4f}  "
                "LR: {lr:.6f}  "
                "top1_avg: {top1.avg:.4f}  ".format(
                    epoch + 1,
                    CFG.epochs,
                    step,
                    len(train_loader),
                    remain=timeSince(start, float(step + 1) / len(train_loader)),
                    loss=losses,
                    grad_norm=grad_norm,
                    lr=scheduler.get_lr()[0],
                    top1=top1,
                )
            )

    return losses.avg


def valid_one_epoch(model, valid_loader, criterion, device):
    """docstirng for valid_one_epoch"""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    start = time.time()
    for step, (images, targets) in enumerate(valid_loader):
        images = images.to(device)
        targets = targets.to(device)
        batch_size = targets.size(0)

        out = model(images)

        loss = criterion(out, targets)
        acc1 = accuracy2(out, targets)

        losses.update(loss, batch_size)
        top1.update(acc1, batch_size)

        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                "EVAL: [{0}/{1}] "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                "top1_avg: {top1.avg:.4f}  ".format(
                    step,
                    len(valid_loader),
                    loss=losses,
                    remain=timeSince(start, float(step + 1) / len(valid_loader)),
                    top1=top1,
                )
            )

    return losses.avg, top1.avg


def inference_one_epoch(model, data_loader, device):
    """docstring for inference_one_epoch"""
    model.eval()
    image_preds_all = []

    for (images, _) in tqdm(enumerate(data_loader), total=len(data_loader)):
        images = images.to(device).float()

        image_preds = model(images)
        image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]

    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all
