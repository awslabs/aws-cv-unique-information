from abc import abstractmethod

from torchvision import transforms, datasets
import torch
import numpy as np

from .base import log_call_parameters
from .abstract import StandardVisionDataset
from .noise_tools import get_uniform_error_corruption_fn, get_corruption_function_from_confusion_matrix

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class CIFAR(StandardVisionDataset):
    @log_call_parameters
    def __init__(self, n_classes: int = 10, data_augmentation: bool = False, **kwargs):
        super(CIFAR, self).__init__(**kwargs)
        assert n_classes in [10, 100]
        self.n_classes = n_classes
        self.data_augmentation = data_augmentation

    @property
    def dataset_name(self) -> str:
        if self.n_classes == 10:
            return "cifar10"
        if self.n_classes == 100:
            return "cifar100"
        raise ValueError("num_classes should be 10 or 100")

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
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.RandomCrop(32, 4),
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
        if self.n_classes == 10:
            return datasets.CIFAR10(data_dir, download=download, train=(split == 'train'),
                                    transform=transform)
        if self.n_classes == 100:
            return datasets.CIFAR100(data_dir, download=download, train=(split == 'train'),
                                     transform=transform)
        raise ValueError("num_classes should be 10 or 100")


def cifar10_custom_confusion_matrix(n_classes, error_prob):
    assert n_classes == 10
    cf = np.eye(n_classes)
    cf[9][1] = error_prob
    cf[9][9] = 1 - error_prob
    cf[2][0] = error_prob
    cf[2][2] = 1 - error_prob
    cf[4][7] = error_prob
    cf[4][4] = 1 - error_prob
    cf[3][5] = error_prob
    cf[3][3] = 1 - error_prob
    assert np.allclose(cf.sum(axis=1), 1)
    return cf


class NoisyCIFAR(CIFAR):
    def __init__(self, n_classes: int = 10, data_augmentation: bool = False, clean_validation: bool = False, **kwargs):
        super(NoisyCIFAR, self).__init__(n_classes=n_classes, data_augmentation=data_augmentation, **kwargs)
        self.clean_validation = clean_validation

    @abstractmethod
    def corruption_fn(self, dataset):
        raise NotImplementedError('corruption_fn is not implemented')

    def post_process_datasets(self, train_data, val_data, test_data, info=None):
        is_corrupted = self.corruption_fn(train_data)
        if not self.clean_validation:
            _ = self.corruption_fn(val_data)
        return train_data, val_data, test_data, is_corrupted


class UniformNoiseCIFAR(NoisyCIFAR):
    @log_call_parameters
    def __init__(self, error_prob: float, n_classes: int = 10, data_augmentation: bool = False,
                 clean_validation: bool = False, **kwargs):
        super(UniformNoiseCIFAR, self).__init__(n_classes=n_classes,
                                                data_augmentation=data_augmentation,
                                                clean_validation=clean_validation,
                                                **kwargs)
        self._corruption_fn = get_uniform_error_corruption_fn(error_prob=error_prob, n_classes=self.n_classes)

    def corruption_fn(self, dataset):
        return self._corruption_fn(dataset)


class PairNoiseCIFAR10(NoisyCIFAR):
    @log_call_parameters
    def __init__(self, error_prob: float, data_augmentation: bool = False, clean_validation: bool = False, **kwargs):
        super(PairNoiseCIFAR10, self).__init__(n_classes=10,
                                               data_augmentation=data_augmentation,
                                               clean_validation=clean_validation,
                                               **kwargs)
        cf = cifar10_custom_confusion_matrix(n_classes=10, error_prob=error_prob)
        self._corruption_fn = get_corruption_function_from_confusion_matrix(cf)

    def corruption_fn(self, dataset):
        return self._corruption_fn(dataset)
