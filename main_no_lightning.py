import argparse
import os

import lightning as L
import torch
import wandb

from torch.nn import Linear
from torch.nn.functional import cross_entropy
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
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
from torchvision.models import resnet50


parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str, required=True)
parser.add_argument("--wandb_api_key", type=str, required=True)
parser.add_argument("--val_dir", type=str, required=True)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--accelerator", type=str, default="cuda")
parser.add_argument("--num_devices", type=int, default=1)
parser.add_argument("--num_classes", type=int, default=1000)
parser.add_argument("--num_epochs", type=int, default=90)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--log_every_n_step", type=int, default=50)
parser.add_argument("--log_model_every_n_step", type=int, default=100)
args = vars(parser.parse_args())


L.seed_everything(42)

os.environ["WANDB_API_KEY"] = args.get("wandb_api_key")

wandb.init(project="ImageNet1k", name="ImageNet1k_ResNet50")

fabric = L.Fabric(
    accelerator=args.get("accelerator", "cuda"), devices=args.get("num_devices", 1)
)
fabric.launch()

train_top1_acc = Accuracy(
    num_classes=args.get("num_classes", 1000), top_k=1, task="multiclass"
).to(args.get("accelerator"))
train_top5_acc = Accuracy(
    num_classes=args.get("num_classes", 1000), top_k=5, task="multiclass"
).to(args.get("accelerator"))
val_top1_acc = Accuracy(
    num_classes=args.get("num_classes", 1000), top_k=1, task="multiclass"
).to(args.get("accelerator"))
val_top5_acc = Accuracy(
    num_classes=args.get("num_classes", 1000), top_k=5, task="multiclass"
).to(args.get("accelerator"))

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

model = resnet50()
model.fc = Linear(2048, args.get("num_classes"))

optimizer = SGD(
    model.parameters(),
    lr=args.get("lr", 0.1),
    weight_decay=args.get("weight_decay", 1e-4),
    momentum=args.get("momentum", 0.9),
)

lr_scheduler = StepLR(
    optimizer=optimizer,
    step_size=30,
    gamma=0.1,
)

model, optimizer = fabric.setup(model, optimizer)


train_loader = fabric.setup_dataloaders(train_loader)
val_loader = fabric.setup_dataloaders(val_loader)

for epoch in range(args.get("num_epochs")):
    print("Training...")

    model.train()
    for step, (input, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(input)

        loss = cross_entropy(output, target)
        fabric.backward(loss)

        batch_top1_acc = train_top1_acc(output, target)
        batch_top5_acc = train_top5_acc(output, target)

        optimizer.step()
        lr_scheduler.step()

        if step % args.get("log_every_n_step") == 0:
            print(
                f"Step {step}, "
                + f"train_loss: {loss}, "
                + f"train_top1_acc: {batch_top1_acc}, "
                + f"train_top5_acc: {batch_top5_acc}"
            )
            wandb.log(
                dict(
                    train_loss=loss,
                    train_top1_acc=batch_top1_acc,
                    train_top5_acc=batch_top5_acc,
                )
            )

        if step % args.get("log_model_every_n_step") == 0:
            torch.save(
                model.state_dict(), os.path.join(wandb.run.dir, "resnet50.chkpt")
            )
            wandb.save(os.path.join(wandb.run.dir, "resnet50.chkpt"))

    print("Evaluation...")

    model.eval()

    avg_val_loss = 0

    for step, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            output = model(input)
            loss = cross_entropy(output, target)

            avg_val_loss += loss.item()

            val_top1_acc(output, target)
            val_top5_acc(output, target)

    epoch_train_top1_acc = train_top1_acc.compute()
    epoch_train_top5_acc = train_top5_acc.compute()
    epoch_val_top1_acc = val_top1_acc.compute()
    epoch_val_top5_acc = val_top5_acc.compute()

    print(
        f"Epoch {epoch}, "
        + f"val_loss: {avg_val_loss / (step + 1)}, "
        + f"val_top1_acc: {epoch_val_top1_acc}, "
        + f"val_top5_acc: {epoch_val_top5_acc}, "
        + f"train_top1_acc: {epoch_train_top1_acc}, "
        + f"train_top5_acc: {epoch_train_top5_acc}"
    )

    wandb.log(
        dict(
            val_loss=avg_val_loss / (step + 1),
            val_top1_acc=epoch_val_top1_acc,
            val_top5_acc=epoch_val_top5_acc,
        )
    )

    train_top1_acc.reset()
    train_top5_acc.reset()
    val_top1_acc.reset()
    val_top5_acc.reset()
