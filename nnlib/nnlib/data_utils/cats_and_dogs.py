import os
import logging
logging.basicConfig(level=logging.INFO)

from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch

from .base import log_call_parameters
from .abstract import StandardVisionDataset


class CatsAndDogs(StandardVisionDataset):
    @log_call_parameters
    def __init__(self, data_augmentation: bool = False, **kwargs):
        super(CatsAndDogs, self).__init__(**kwargs)
        self.data_augmentation = data_augmentation

    @property
    def dataset_name(self) -> str:
        return "cats-and-dogs"

    @property
    def means(self):
        return torch.tensor([0.485, 0.456, 0.406])

    @property
    def stds(self):
        return torch.tensor([0.229, 0.224, 0.225])

    @property
    def train_transforms(self):
        if not self.data_augmentation:
            return self.test_transforms

        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def test_transforms(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    def raw_dataset(self, data_dir: str, download: bool, split: str, transform):
        assert split in ['train', 'val', 'test']
        if split == 'train':
            return ImageFolder(os.path.join(data_dir, 'PetImages'), transform=transform)
        if split == 'test':
            logging.warning("The cats and dogs dataset has only training set. "
                            "Instead of a testing set the training set will be returned.")
            return ImageFolder(os.path.join(data_dir, 'PetImages'), transform=transform)
        return None  # no predetermined validation set
