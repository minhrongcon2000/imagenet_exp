import argparse

import torchvision.transforms as transforms
from datasets.imagenet import ImageNet1k
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", type=str, required=True)
# parser.add_argument("--val_dir", type=str, required=True)
args = vars(parser.parse_args())

dataset = ImageNet1k(label_files=args["train_dir"])
print(len(dataset))

dataset = ImageNet1k(label_files=args["train_dir"], transform=transforms.Compose([
    transforms.RandomCrop(size=224),
    transforms.ToTensor(),
]))
loader = DataLoader(dataset=dataset, batch_size=32)
for batch in loader:
    print(batch)
    break
