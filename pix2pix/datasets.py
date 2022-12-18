import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch

class ImageDataset(Dataset):
    def __init__(self, path, transforms_a=None,transforms_b=None):
        self.transform_a = transforms.Compose(transforms_a)
        self.transform_b = transforms.Compose(transforms_b)
        f = open(path, 'r')
        self.files = f.readlines()
        f.close()

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)].strip()
        img_path = img_path.split(' ')

        img_a = Image.open(img_path[0])
        img_b = Image.open(img_path[1])
        if np.random.random() < 0.7:
            if np.random.random()<0.5:
                img_a = Image.fromarray(np.fliplr(np.array(img_a)),'RGB')
                img_b = Image.fromarray(np.fliplr(np.array(img_b)), 'L')
            else:
                img_a = Image.fromarray(np.flipud(np.array(img_a)),'RGB')
                img_b = Image.fromarray(np.flipud(np.array(img_b)), 'L')

        img_a = self.transform_a(img_a)
        img_b = self.transform_b(img_b)
        return {'A': img_a, 'B':img_b}

    def __len__(self):
        return len(self.files)


class ImageTestDataset(Dataset):
    def __init__(self, path, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        f = open(path, 'r')
        self.files = f.readlines()
        f.close()

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)].strip()
        img_path = img_path.split(' ')

        img_a = Image.open(img_path[0])
        if np.random.random() < 0.7:
            if np.random.random()<0.5:
                img_a = Image.fromarray(np.fliplr(np.array(img_a)),'RGB')
            else:
                img_a = Image.fromarray(np.flipud(np.array(img_a)),'RGB')

        img_a = self.transform(img_a)
        return {'A': img_a}

    def __len__(self):
        return len(self.files)