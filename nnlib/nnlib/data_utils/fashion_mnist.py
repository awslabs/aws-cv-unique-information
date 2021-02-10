from torchvision import transforms, datasets
import torch


from .base import log_call_parameters
from .abstract import StandardVisionDataset


class FashionMNIST(StandardVisionDataset):
    @log_call_parameters
    def __init__(self, data_augmentation: bool = False, **kwargs):
        super(FashionMNIST, self).__init__(**kwargs)
        self.data_augmentation = data_augmentation

    @property
    def dataset_name(self) -> str:
        return "fashion-mnist"

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
                                   transforms.RandomHorizontalFlip(),
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
        return datasets.FashionMNIST(data_dir, download=download, train=(split == 'train'),
                                     transform=transform)
