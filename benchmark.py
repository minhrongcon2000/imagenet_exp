import argparse
import os

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
from torchvision.datasets import ImageFolder


parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str, required=True)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=256)
args = vars(parser.parse_args())


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

train_dataset = ImageFolder(
    root=args.get("train_dir"),
    transform=train_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.get("batch_size"),
    shuffle=True,
    num_workers=args.get("num_workers"),
)

for i, (batchX, batchY) in enumerate(train_loader):
    print(i, batchX.shape, batchY.shape)
