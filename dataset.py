from random import sample
from tensorflow_hub import image_embedding_column
import torch 
import numpy as np
import os
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

class dataset(Dataset):
    def __init__(self, root_Mask, root_X_ray, sample_len = None):
        super().__init__()
        self.transformation = transforms.Compose([
                            transforms.Resize((128, 128)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(degrees = 10),
                            transforms.ToTensor(),
                        ])

        self.root_mask = root_Mask
        self.root_x_ray = root_X_ray
        
        self.folder_mask = os.listdir(root_Mask) if sample_len is None else os.listdir(root_Mask)[:sample_len]
        self.folder_x_ray = os.listdir(root_X_ray) if sample_len is None else os.listdir(root_X_ray)[:sample_len]


    def __len__(self):
        return min(len(self.folder_mask), len(self.folder_x_ray))
        
    def __getitem__(self, index):
        mask_file = os.path.join(self.root_mask, self.folder_mask[index])
        x_ray_file = os.path.join(self.root_x_ray, self.folder_x_ray[index])
        
        return {
            "mask": self.transformation(Image.open(mask_file).convert("L")),
            "x_ray": self.transformation(Image.open(x_ray_file).convert("L"))
        }
