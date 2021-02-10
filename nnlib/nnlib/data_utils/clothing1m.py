from PIL import Image
import os
import logging
logging.basicConfig(level=logging.INFO)

from torchvision import transforms
from torch.utils.data import Subset, Dataset
import numpy as np
import torch

from .base import log_call_parameters, print_loaded_dataset_shapes
from .abstract import StandardVisionDataset


class Clothing1MRaw(Dataset):
    def __init__(self, root, img_transform, split: str):
        self.root = root
        if split == 'train':
            flist = os.path.join(root, "dmi_annotations/noisy_train.txt")
        elif split == 'val':
            flist = os.path.join(root, "dmi_annotations/clean_val.txt")
        elif split == 'test':
            flist = os.path.join(root, "dmi_annotations/clean_test.txt")
        else:
            raise ValueError(f"Split should be 'train', 'val', or 'test'.")
        self.imlist = self.flist_reader(flist)
        self.transform = img_transform

    def __getitem__(self, index):
        path, target = self.imlist[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imlist)

    def flist_reader(self, flist):
        imlist = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                row = line.split(" ")
                impath = os.path.join(self.root, row[0])
                imlabel = row[1]
                imlist.append((impath, int(imlabel)))
        return imlist


class Clothing1M(StandardVisionDataset):
    @log_call_parameters
    def __init__(self, data_augmentation: bool = False, **kwargs):
        super(Clothing1M, self).__init__(**kwargs)
        self.data_augmentation = data_augmentation

    @property
    def dataset_name(self) -> str:
        return "clothing1m"

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
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def test_transforms(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    def raw_dataset(self, data_dir: str, download: bool, train: bool, transform):
        pass

    @print_loaded_dataset_shapes
    @log_call_parameters
    def build_datasets(self, data_dir: str = None, val_ratio: float = 0.2, num_train_examples: int = None,
                       seed: int = 42, **kwargs):
        """ Builds train, validation, and test datasets. """
        logging.info(f"val_ratio is ignored as {self.dataset_name} does not require splitting")
        logging.info(f"Building datasets of {self.dataset_name} with data_dir={data_dir}, "
                     f"num_train_examples={num_train_examples}, seed={seed}")

        if data_dir is None:
            data_dir = os.path.join(os.environ['DATA_DIR'], self.dataset_name)

        train_data = Clothing1MRaw(root=data_dir, split='train', img_transform=self.train_transforms)
        val_data = Clothing1MRaw(root=data_dir, split='val', img_transform=self.test_transforms)
        test_data = Clothing1MRaw(root=data_dir, split='test', img_transform=self.test_transforms)

        np.random.seed(seed)
        if num_train_examples is not None:
            subset = np.random.choice(len(train_data), num_train_examples, replace=False)
            train_data = Subset(train_data, subset)

        # name datasets and save statistics
        for dataset in [train_data, val_data, test_data]:
            dataset.dataset_name = self.dataset_name
            dataset.statistics = (self.means, self.stds)

        # general way of returning extra information
        info = None

        return train_data, val_data, test_data, info
