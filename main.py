import argparse
import os

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import disable_beta_transforms_warning
from torchvision.transforms import (CenterCrop, Compose, Normalize,
                                    RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor)

from datasets.imagenet import ImageNet1k
from models import ResNet50

parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str, required=True)
parser.add_argument("--val_dir", type=str, required=True)
parser.add_argument("--wandb_api_key", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--num_gpu", type=int, default=2)
args = vars(parser.parse_args())

os.environ["WANDB_API_KEY"] = args.get("wandb_api_key")

seed_everything(42)
disable_beta_transforms_warning()

train_transform = Compose([
    ToTensor(),
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
    Normalize([0.49139968, 0.48215841, 0.44653091],
              [0.24703223, 0.24348513, 0.26158784]),
])

test_transform = Compose([
    ToTensor(),
    Resize(size=256),
    CenterCrop(224),
    Normalize([0.49139968, 0.48215841, 0.44653091],
              [0.24703223, 0.24348513, 0.26158784]),
])

train_dataset = ImageNet1k(label_files=args["train_dir"], transform=train_transform)
val_dataset = ImageNet1k(label_files=args["val_dir"], transform=test_transform)

train_loader = DataLoader(train_dataset,
                          batch_size=64,
                          shuffle=True,
                          num_workers=4)
val_loader = DataLoader(val_dataset,
                        batch_size=64,
                        shuffle=False,
                        num_workers=4)

model = ResNet50(num_classes=1000)

pl_trainer = Trainer(accelerator="gpu",
                     devices=args.get("num_gpu"),
                     strategy="ddp" if args.get("num_gpu") > 1 else "auto",
                     max_epochs=100,
                     enable_progress_bar=False,
                     callbacks=[
                         ModelCheckpoint("chkpt",
                                         monitor=ResNet50.VAL_TOP1_ACC_KEY,
                                         mode="max"),
                     ],
                     logger=WandbLogger(project="ImageNet1k",
                                        name="ImageNet1k_ResNet50"))
pl_trainer.fit(model=model,
               train_dataloaders=train_loader,
               val_dataloaders=val_loader)
