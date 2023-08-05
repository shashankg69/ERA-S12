from .dataset import Dataset
import cv2
import torch
import numpy as np
from torchvision import datasets
import albumentations as A

class AlbCIFAR10(datasets.CIFAR10):
    def __init__(self, root, alb_transform=None, **kwargs):
        super(AlbCIFAR10, self).__init__(root, **kwargs)
        self.alb_transform = alb_transform

    def __getitem__(self, index):
        image, label = super(AlbCIFAR10, self).__getitem__(index)
        if self.alb_transform is not None:
            image = self.alb_transform(image=np.array(image))['image']
        return image, label
    
class CIFAR10(Dataset):
    DataSet = AlbCIFAR10
    mean = (0.49139968, 0.48215827, 0.44653124)
    std = (0.24703233, 0.24348505, 0.26158768)
    classes = None
    default_alb_transforms = [
        A.HorizontalFlip(p=0.5),
        A.PadIfNeeded(40, 40, p=1),
        A.RandomCrop(32, 32, p=1),
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, fill_value=0, p=1),
        A.PadIfNeeded(64, 64, border_mode=cv2.BORDER_CONSTANT, value=0, p=1),
        A.CenterCrop(32, 32, p=1) 
    ]

