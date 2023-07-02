import argparse
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

# from datasets.imagenet import ImageNet1k
from torchvision.datasets import ImageFolder
from models import ResNet50


parser = argparse.ArgumentParser()
data_group = parser.add_argument_group("data config")
data_group.add_argument("--train_dir", type=str, required=True)
data_group.add_argument("--val_dir", type=str, required=True)
trainer_group = parser.add_argument_group("trainer config")
trainer_group.add_argument("--batch_size", type=int, default=256)
trainer_group.add_argument("--num_devices", type=int, default=2)
trainer_group.add_argument("--device", type=str, default="gpu")
trainer_group.add_argument("--num_workers", type=int, default=16)
trainer_group.add_argument("--resume_artifact", type=str)
trainer_group.add_argument("--num_epochs", type=int, default=90)
model_group = parser.add_argument_group("model config")
model_group.add_argument("--lr", type=float, default=0.1)
model_group.add_argument("--num_classes", type=int, default=1000)
model_group.add_argument("--weight_decay", type=float, default=1e-4)
model_group.add_argument("--momentum", type=float, default=0.9)
args = vars(parser.parse_args())

seed_everything(42)


train_transform = Compose(
    [
        RandomResizedCrop(224, antialias=True),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)

test_transform = Compose(
    [
        Resize(size=256, antialias=True),
        CenterCrop(224),
        ToTensor(),
        Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)

train_dataset = ImageFolder(
    root=args.get("train_dir"),
    transform=train_transform,
)
val_dataset = ImageFolder(
    root=args.get("val_dir"),
    transform=test_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.get("batch_size"),
    shuffle=True,
    num_workers=args.get("num_workers"),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=args.get("batch_size"),
    shuffle=False,
    num_workers=args.get("num_workers"),
)


model = ResNet50(
    num_classes=args.get("num_classes"),
    lr=args.get("lr"),
    weight_decay=args.get("weight_decay"),
    momentum=args.get("momentum"),
)

pl_trainer = Trainer(
    accelerator=args.get("device"),
    devices=args.get("num_devices"),
    max_epochs=args.get("num_epochs"),
    enable_progress_bar=False,
    callbacks=[
        ModelCheckpoint(
            "chkpt",
            save_last=True,
            save_top_k=5,
            monitor=ResNet50.VAL_TOP1_ACC_KEY,
            mode="max",
        ),
        LearningRateMonitor(),
    ],
    logger=TensorBoardLogger(save_dir="logs"),
)

pl_trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)
