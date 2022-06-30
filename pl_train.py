"""import modules and libraries"""
# pylint: disable=C0301, W0621
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# from pytorch_lightning.strategies import DeepSpeedStrategy
import wandb

from data.cifar10.plcifar10 import CIFAR10Data
from model.lightningmodule import MyModule


def main(args):
    """Whole process of training"""
    # setting wandb in image_playground project
    wandb.login(key=args.wandb_key)
    wandb_logger = WandbLogger(
        name=f"{args.model_name}-{args.batch_size}", project="image_playground"
    )

    # getting data
    cifar = CIFAR10Data(args.data_dir, args.batch_size, args.fold)
    cifar.prepare_data()
    cifar.setup(stage="fit")

    model = MyModule(args)

    pl.seed_everything(43)
    trainer = pl.Trainer(
        # max_epochs=100, # max_epochs=-1
        accelerator="auto",
        # auto_lr_find=True,
        # auto_scale_batch_size="binsearch",
        # accumulate_grad_batches={0: 8, 4: 4, 8: 1},
        gradient_clip_val=0.5,
        # gradient_clip_algorithm="value",
        detect_anomaly=True,
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=wandb_logger,
        callbacks=[
            EarlyStopping(monitor="VALID LOSS", mode="min", patience=10, verbose=True),
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=50),
        ],
        # strategy=DeepSpeedStrategy(
        #     stage=3,
        #     offload_optimizer=True,
        #     offload_parameters=True,
        #     # remote_device='nvme',
        #     # offload_params_device='nvme',
        #     # offload_optimizer_device='nvme',
        #     # nvme_path="/mnt/nvme",
        # ),
        strategy="deepspeed_stage_2_offload",
        # precision=16,
    )

    # getting new_lr from auto_lr_finder
    # lr_finder = trainer.tuner.lr_find(
    #     model,
    #     cifar.train_dataloader(),
    #     mode="exponential",
    #     min_lr=1e-6,
    #     max_lr=1e-3,
    #     num_training=100,
    # )
    # fig = lr_finder.plot(suggest=True)
    # fig.savefig("/")
    # new_lr = lr_finder.suggestion()
    # print(f"Suggested learning rate: {new_lr}")
    # model.hparams.lr = new_lr

    # trainer.tune(model, datamodule=cifar)
    trainer.fit(model, cifar)
    cifar.setup(stage="test")
    trainer.test(model, datamodule=cifar)
    trainer.save_checkpoint(args.pth_dir, f"{args.model_name}.pth")
    wandb.finish()


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--data_dir",
        default="./data/cifar10",
        type=str,
        help="path to dataset(default: ./data/cifar10)",
    )
    parser.add_argument(
        "--pth_dir", default="./pth", type=str, help="path to save model pth"
    )
    parser.add_argument(
        "--model_name",
        default="alexnet",
        type=str,
        help="alexnet|vgg|googlenet|resnet(default:alexnet)",
    )
    parser.add_argument(
        "--num_classes", default=10, type=int, help="number of classes(default:10)"
    )
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--batch_norm", default=True, type=bool)
    parser.add_argument("--epochs", default=-1, type=int)
    parser.add_argument("--image_size", default=32, type=int)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--init_weight", default=True, type=bool)
    parser.add_argument("--dropout_rate", default=0.5, type=float)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--patience", default=3, type=int)
    # wandb config
    parser.add_argument(
        "--wandb_key", default="93460ff86561b201141546a407885ba3c8318d81", type=str
    )
    args = parser.parse_args()
    main(args)
