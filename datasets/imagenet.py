from typing import List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class ImageNet1k(Dataset):
    def __init__(self, label_files: str, transform=None) -> None:
        super().__init__()
        self.label_files = label_files
        self.label_df = pd.read_csv(self.label_files)
        self.transform=transform
        
    def __len__(self) -> int:
        return len(self.label_df)
    
    def __getitem__(self, index: Union[int, List[int], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(index):
            index = index.tolist()
        
        img_dir = self.label_df.loc[index, "image_dir"]
        labels = self.label_df.loc[index, "label"]
        
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        
        if self.transform:
            img = self.transform(img)
        
        return img, labels