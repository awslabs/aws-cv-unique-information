from torchvision import transforms, datasets
import torch

from .base import log_call_parameters
from .abstract import StandardVisionDataset


class EMNIST(StandardVisionDataset):
    @log_call_parameters
    def __init__(self, split: str = 'letters', data_augmentation: bool = False, **kwargs):
        super(EMNIST, self).__init__(**kwargs)
        self.split = split
        self.data_augmentation = data_augmentation

    @property
    def dataset_name(self) -> str:
        return "emnist"

    @property
    def means(self):
        return torch.tensor([0.456])

    @property
    def stds(self):
        return torch.tensor([0.224])

    @property
    def train_transforms(self):
        if not self.data_augmentation:
            return self.test_transforms
        return transforms.Compose([transforms.RandomCrop(28, 4),
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
        if self.split == 'letters':
            target_transform = (lambda x: x - 1)
        else:
            target_transform = None
        if split == 'val':
            return None  # no predetermined validation set
        return datasets.EMNIST(data_dir, split=self.split, download=download, train=(split == 'train'),
                               transform=transform, target_transform=target_transform)
