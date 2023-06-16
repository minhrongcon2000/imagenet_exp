import argparse
import os

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

parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str, required=True)
parser.add_argument("--val_dir", type=str, required=True)
parser.add_argument("--wandb_api_key", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--num_devices", type=int, default=2)
parser.add_argument("--resume_artifact", type=str)
parser.add_argument("--device", type=str, default="gpu")
args = vars(parser.parse_args())

os.environ["WANDB_API_KEY"] = args.get("wandb_api_key")

seed_everything(42)

train_transform = Compose(
    [
        ToTensor(),
        RandomResizedCrop(224),
        RandomHorizontalFlip(),
        Normalize(
            [0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]
        ),
    ]
)

test_transform = Compose(
    [
        ToTensor(),
        Resize(size=256),
        CenterCrop(224),
        Normalize(
            [0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784]
        ),
    ]
)

train_dataset = ImageFolder(root=args["train_dir"], transform=train_transform)
val_dataset = ImageFolder(root=args["val_dir"], transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)


model = ResNet50(num_classes=1000)

if args.get("resume_artifact"):
    artifact_dir = WandbLogger.download_artifact(args.get("resume_artifact"))
else:
    artifact_dir = None

pl_trainer = Trainer(
    accelerator=args.get("device"),
    devices=args.get("num_devices"),
    strategy="auto",
    max_epochs=100,
    enable_progress_bar=False,
    callbacks=[
        ModelCheckpoint("chkpt", save_last=True, save_top_k=5),
        LearningRateMonitor(),
    ],
    logger=WandbLogger(
        project="ImageNet1k", name="ImageNet1k_ResNet50", log_model=True
    ),
)

pl_trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
    ckpt_path=os.path.join(artifact_dir, "model.ckpt")
    if artifact_dir is not None
    else None,
)
