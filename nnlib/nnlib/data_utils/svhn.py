from torchvision import transforms, datasets
import torch

from .base import log_call_parameters
from .abstract import StandardVisionDataset


class SVHN(StandardVisionDataset):
    @log_call_parameters
    def __init__(self, data_augmentation: bool = False, **kwargs):
        super(SVHN, self).__init__(**kwargs)
        self.data_augmentation = data_augmentation

    @property
    def dataset_name(self) -> str:
        return "svhn"

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
        return transforms.Compose([transforms.RandomCrop(32, 4),
                                   transforms.ToTensor(),
                                   self.normalize_transform])

    @property
    def test_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            self.normalize_transform
        ])

    def raw_dataset(self, data_dir: str, download: bool, split: str, transform):
        assert split in ['train', 'val', 'test']
        if split == 'val':
            return None  # no predetermined validation set
        return datasets.SVHN(data_dir, download=download, split=split, transform=transform)
