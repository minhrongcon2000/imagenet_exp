import argparse
import os

import cv2
import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
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

# https://github.com/pytorch/pytorch/issues/11201
torch.multiprocessing.set_sharing_strategy("file_system")

parser = argparse.ArgumentParser()
data_group = parser.add_argument_group("data config")
data_group.add_argument("--train_dir", type=str, required=True)
data_group.add_argument("--val_dir", type=str, required=True)
auth_group = parser.add_argument_group("wandb authentication")
auth_group.add_argument("--wandb_api_key", type=str, required=True)
trainer_group = parser.add_argument_group("trainer config")
trainer_group.add_argument("--batch_size", type=int, default=256)
trainer_group.add_argument("--num_devices", type=int, default=2)
trainer_group.add_argument("--device", type=str, default="gpu")
trainer_group.add_argument("--num_workers", type=int, default=4)
trainer_group.add_argument("--resume_artifact", type=str)
trainer_group.add_argument("--num_epochs", type=int, default=90)
model_group = parser.add_argument_group("model config")
model_group.add_argument("--lr", type=float, default=0.1)
model_group.add_argument("--num_classes", type=int, default=1000)
model_group.add_argument("--weight_decay", type=float, default=1e-4)
model_group.add_argument("--momentum", type=float, default=0.9)
args = vars(parser.parse_args())

os.environ["WANDB_API_KEY"] = args.get("wandb_api_key")

seed_everything(42)


def image_loader(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return img


train_transform = Compose(
    [
        ToTensor(),
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        Normalize(
            mean=[0.49139968, 0.48215841, 0.44653091],
            std=[0.24703223, 0.24348513, 0.26158784],
        ),
    ]
)

test_transform = Compose(
    [
        ToTensor(),
        Resize(size=256),
        CenterCrop(224),
        Normalize(
            mean=[0.49139968, 0.48215841, 0.44653091],
            std=[0.24703223, 0.24348513, 0.26158784],
        ),
    ]
)

train_dataset = ImageFolder(
    root=args.get("train_dir"),
    transform=train_transform,
    loader=image_loader,
)
val_dataset = ImageFolder(
    root=args.get("val_dir"),
    transform=test_transform,
    loader=image_loader,
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

if args.get("resume_artifact"):
    artifact_dir = WandbLogger.download_artifact(args.get("resume_artifact"))
else:
    artifact_dir = None

logger = WandbLogger(project="ImageNet1k", name="ImageNet1k_ResNet50", log_model=True)

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
    logger=logger,
)

try:
    pl_trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=os.path.join(artifact_dir, "model.ckpt")
        if artifact_dir is not None
        else None,
    )

except Exception as e:
    logger._experiment.alert(
        title="Run crashes",
        text=str(e.with_traceback()),
    )
    raise (e)
