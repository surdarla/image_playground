"""import modules and libraries"""
# pylint: disable=C0301, W0621
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# from pytorch_lightning.strategies import DeepSpeedStrategy
import wandb

from data.cifar10.plcifar10 import CIFAR10Data
from model.lightningmodule import MyModule
from model.alexnet import AlexNet
from model.vgg import VGG
from model.googlenet import GoogLeNet


def main(args):
    """Whole process of training"""
    # setting wandb in image_playground project
    wandb.login(key=args.wandb_key)
    wandb_logger = WandbLogger(
        name=f"{args.model_name}-{args.batch_size}-{args.fp16}",
        project="image_playground",
    )

    # getting data
    cifar = CIFAR10Data(
        root=args.data_dir,
        batch_size=args.batch_size,
        this_fold=args.fold,
        num_workers=args.num_workers,
    )
    cifar.prepare_data()
    cifar.setup(stage="fit")

    if args.model_name == "alexnet":
        model = AlexNet(args.num_classes, args.dropout_rate)
    elif args.model_name.startswith("VGG"):
        model = VGG(
            args.model_name,
            args.num_classes,
            args.init_weight,
            args.dropout_rate,
            args.batch_norm,
        )
    elif args.model_name == "googlenet":
        model = GoogLeNet(
            num_classes=args.num_classes,
        )

    # dict_args = vars(args)
    lit_model = MyModule(model, args)

    pl.seed_everything(43)
    trainer = pl.Trainer(
        # max_epochs=100, # max_epochs=-1
        accelerator="gpu",
        log_every_n_steps=10,
        auto_lr_find=True,
        auto_scale_batch_size="binsearch",  # not compatible with deepspeed
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
        # strategy="deepspeed_stage_2_offload",
        # strategy="deepspeed_stage_3",
        precision=16 if args.fp16 == 16 else 32,
    )
    # scale batch size
    tuner = Tuner(trainer)
    new_batch_size = tuner.scale_batch_size(
        lit_model, mode="binsearch", init_val=128, max_trials=3, datamodule=cifar
    )
    lit_model.hparams.batch_size = new_batch_size
    # learinig rate finding
    lr_finder = tuner.lr_find(lit_model, datamodule=cifar)
    fig = lr_finder.plot(suggest=True)
    fig.show()
    lit_model.hparams.learning_rate = lr_finder.suggestion()

    # training
    trainer.fit(lit_model, cifar)
    cifar.setup(stage="test")
    trainer.test(lit_model, datamodule=cifar)
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
        default="googlenet",
        type=str,
        help="alexnet|VGG11|VGG13|VGG16|VGG19|googlenet|resnet(default:alexnet)",
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
    parser.add_argument("--n_splits", default=5, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--fp16", default=16, type=int)
    # wandb config
    parser.add_argument(
        "--wandb_key", default="93460ff86561b201141546a407885ba3c8318d81", type=str
    )
    args = parser.parse_args()
    main(args)
