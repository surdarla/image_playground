"""import modules and libraries"""
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

from data.cifar10.plcifar10 import CIFAR10Data
from config import CFG
from model.alexnet import AlexNetLit
from model.vgg import VGG

wandb.login(key=CFG.wandb_key)
cifar = CIFAR10Data(CFG.data_dir, CFG.batch_size, 0)
cifar.prepare_data()
cifar.setup(stage="fit")

if CFG.MODEL is "alexnet":
    model = AlexNetLit(10, lr=CFG.min_lr)
elif CFG.MODEL is "vgg":
    model = VGG("VGG13")
wandb_logger = WandbLogger(name="test1", project="image_playground")
pl.seed_everything(43)
trainer = pl.Trainer(
    # max_epochs=100, # max_epochs=-1
    accelerator="auto",
    auto_lr_find=True,
    accumulate_grad_batches={0: 8, 4: 4, 8: 1},
    gradient_clip_val=0.5,
    gradient_clip_algorithm="value",
    detect_anomaly=True,
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    logger=wandb_logger,
    callbacks=[
        EarlyStopping(monitor="VALID LOSS", mode="min", patience=10, verbose=True),
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=50),
    ],
)
# getting new_lr from auto_lr_finder
lr_finder = trainer.tuner.lr_find(
    model,
    cifar.train_dataloader(),
    mode="exponential",
    min_lr=1e-6,
    max_lr=1e-3,
    num_training=100,
)
fig = lr_finder.plot(suggestion=True)
fig.savefig("/")
new_lr = lr_finder.suggestion()
print(f"Suggested learning rate: {new_lr}")
model.hparams.lr = new_lr

trainer.fit(model, cifar)
trainer.test(model, cifar)
trainer.save_checkpoint(CFG.pth_dir, f"{CFG.MODEL}.pth")
wandb.finish()
