import argparse

import lightning as L

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
parser.add_argument("--accelerator", type=str, default="cuda")
parser.add_argument("--num_devices", type=int, default=1)
args = vars(parser.parse_args())


L.seed_everything(42)

fabric = L.Fabric(
    accelerator=args.get("accelerator", "cuda"), devices=args.get("num_devices", 1)
)
fabric.launch()

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
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.get("batch_size"),
    shuffle=True,
    num_workers=args.get("num_workers"),
)

train_loader = fabric.setup_dataloaders(train_loader)

for epoch in range(args.get("num_epochs")):
    for step, (input, target) in enumerate(train_loader):
        print(step)
