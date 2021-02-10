# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import copy
import logging

import numpy as np
import torch.utils.data
import torch.nn.functional as F
from torchvision import transforms

import nnlib.nnlib.data_utils.abstract
import nnlib.nnlib.data_utils.base


class LabelSubsetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, which_labels=(0, 1)):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        :param which_labels: which labels to use
        """
        super(LabelSubsetWrapper, self).__init__()
        self.dataset = dataset
        self.which_labels = which_labels
        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics
        self.valid_indices = [idx for idx, (x, y) in enumerate(dataset) if y in which_labels]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.valid_indices[idx]]
        assert y in self.which_labels
        new_y = self.which_labels.index(y)
        return x, torch.tensor(new_y, dtype=torch.long)


BinaryDatasetWrapper = LabelSubsetWrapper  # shortcut


class LabelSelectorWrapper(torch.utils.data.Dataset):
    """ Select a subset of label in case it is an array. """

    def __init__(self, dataset, which_labels):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        :param which_labels: which coordinates of label array to select
        """
        super(LabelSelectorWrapper, self).__init__()
        self.dataset = dataset
        self.which_labels = which_labels
        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y[self.which_labels]


class OneHotLabelWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, num_classes):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        :param num_classes: number of classes
        """
        super(OneHotLabelWrapper, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes
        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        y = torch.tensor(y, dtype=torch.long)
        y = F.one_hot(y, num_classes=self.num_classes)
        return x, y


class ReturnSampleIndexWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        """
        super(ReturnSampleIndexWrapper, self).__init__()
        self.dataset = dataset
        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return (x, idx), y


class SubsetDataWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, exclude_indices=None, include_indices=None):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        """
        super(SubsetDataWrapper, self).__init__()

        if exclude_indices is None:
            assert include_indices is not None
        if include_indices is None:
            assert exclude_indices is not None

        self.dataset = dataset

        if include_indices is not None:
            self.include_indices = include_indices
        else:
            S = set(exclude_indices)
            self.include_indices = [idx for idx in range(len(dataset)) if idx not in S]

        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics

    def __len__(self):
        return len(self.include_indices)

    def __getitem__(self, idx):
        real_idx = self.include_indices[idx]
        return self.dataset[real_idx]


class ResizeImagesWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, size=(224, 224)):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        """
        super(ResizeImagesWrapper, self).__init__()
        self.dataset = dataset
        self.size = size

        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = F.interpolate(x.unsqueeze(dim=0), size=self.size, mode='bilinear')[0]
        return x, y


class MetisDataset(nnlib.nnlib.data_utils.abstract.StandardVisionDataset):
    @nnlib.nnlib.data_utils.base.log_call_parameters
    def __init__(self, dataset: str, data_augmentation: bool = False, **kwargs):
        super(MetisDataset, self).__init__(**kwargs)
        self._dataset_name = dataset
        self.data_augmentation = data_augmentation

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

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

    def raw_dataset(self, data_dir: str, download: bool, train: bool, transform):
        from metis import Metis
        metis = Metis()
        return metis.pytorch_dataset(self.dataset_name, split=('train' if train else 'test'),
                                     target_to_number=True, transform=transform)


class CacheDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        """
        super(CacheDatasetWrapper, self).__init__()
        self.dataset = dataset

        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics

        # create cache
        logging.warning(f"Caching dataset {dataset.dataset_name}. Assuming data augmentation is disabled.")
        self._cached_dataset = [p for p in dataset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self._cached_dataset[idx]


class MergeDatasetsWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset1, dataset2):
        """
        :params dataset1 and dataset2: StandardVisionDataset or derivative class instance
        """
        super(MergeDatasetsWrapper, self).__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2

        # record important attributes
        self.dataset_name = f'merge {dataset1.dataset_name} and {dataset2.dataset_name}'
        self.statistics = dataset1.statistics

    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)

    def __getitem__(self, idx):
        if idx < len(self.dataset1):
            return self.dataset1[idx]
        return self.dataset2[idx - len(self.dataset1)]


class GrayscaleToColoredWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        """
        :params dataset: StandardVisionDataset or derivative class instance
        """
        super(GrayscaleToColoredWrapper, self).__init__()
        self.dataset = dataset

        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = (dataset.statistics[0].repeat(3), dataset.statistics[1].repeat(3))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x.repeat(3, 1, 1), y


class UniformNoiseWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, error_prob, num_classes, seed=42):
        """
        :params dataset: StandardVisionDataset or derivative class instance
        :params error_prob: the probability that a label is switched

        Assumes that the labels are integers
        """
        super(UniformNoiseWrapper, self).__init__()
        self.dataset = dataset
        self.error_prob = error_prob
        self.num_classes = num_classes
        self.seed = seed

        # prepare samples
        self.ys = []
        self.is_corrupted = np.zeros(len(dataset), dtype=np.bool)
        np.random.seed(seed)
        for idx in range(len(dataset)):
            y = torch.tensor(dataset[idx][1]).item()
            if np.random.rand() < error_prob:
                self.is_corrupted[idx] = True
                while True:
                    new_y = np.random.choice(num_classes)
                    if new_y != y:
                        y = new_y
                        break
            self.ys.append(y)

        # record important attributes
        self.dataset_name = dataset.dataset_name
        self.statistics = dataset.statistics

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, self.ys[idx]


def get_synthetic_data(seed=42, extra_dim=100, num_examples=600):
    np.random.seed(seed)
    mu_1 = np.array([0.0, 1.0])
    mu_2 = np.array([1.0, 0.0])
    sigma_small = 0.25 ** 2
    sigma_large = 0.5 ** 2
    cov_1 = np.array([
        [sigma_small, 0.0],
        [0.0, sigma_large]
    ])

    cov_2 = np.array([
        [sigma_large, 0.0],
        [0.0, sigma_small]
    ])

    np.random.seed(seed)
    A = np.random.multivariate_normal(mu_1, cov_1, size=(num_examples,))
    B = np.random.multivariate_normal(mu_2, cov_2, size=(num_examples,))
    if extra_dim > 0:
        A_extra = np.random.randn(A.shape[0], extra_dim)
        B_extra = np.random.randn(B.shape[0], extra_dim)
        A = np.concatenate([A, A_extra], axis=1)
        B = np.concatenate([B, B_extra], axis=1)

    data_X = np.concatenate([A, B], axis=0)
    data_Y = np.concatenate([np.zeros(A.shape[0]), np.ones(B.shape[0])], axis=0)
    shuffle_order = np.random.permutation(data_X.shape[0])
    data_X = data_X[shuffle_order]
    data_Y = data_Y[shuffle_order]

    info = {
        'mu_1': mu_1,
        'mu_2': mu_2,
        'cov_1': cov_1,
        'cov_2': cov_2
    }

    return data_X, data_Y, info


class DataSelector(nnlib.nnlib.data_utils.base.DataSelector):
    """ Helper class for loading data from arguments. """
    _parsers = nnlib.nnlib.data_utils.base.DataSelector._parsers.copy()  # copying to keep the parent class intact

    def __init__(self):
        super(DataSelector, self).__init__()

    @nnlib.nnlib.data_utils.base.register_parser(_parsers, 'mnist4vs9')
    def _parse_mnist(self, args, build_loaders=True):
        args = copy.deepcopy(args)
        num_train_examples = args.pop('num_train_examples', None)

        from nnlib.nnlib.data_utils.mnist import MNIST
        data_builder = MNIST(**args)

        train_data, val_data, test_data, info = data_builder.build_datasets(**args)
        train_data = BinaryDatasetWrapper(train_data, which_labels=(4, 9))
        val_data = BinaryDatasetWrapper(val_data, which_labels=(4, 9))
        test_data = BinaryDatasetWrapper(test_data, which_labels=(4, 9))

        # trim down validation and training sets to num_train_examples
        if num_train_examples is not None:
            np.random.seed(args.get('seed', 42))
            if len(train_data) > num_train_examples:
                train_indices = np.random.choice(len(train_data), size=num_train_examples, replace=False)
                train_data = SubsetDataWrapper(train_data, include_indices=train_indices)

            if len(val_data) > num_train_examples:
                val_indices = np.random.choice(len(val_data), size=num_train_examples, replace=False)
                val_data = SubsetDataWrapper(val_data, include_indices=val_indices)

        # add label noise
        seed = args.get('seed', 42)
        error_prob = args.get('error_prob', 0.0)
        if error_prob > 0.0:
            train_data = UniformNoiseWrapper(train_data, error_prob=error_prob, num_classes=2, seed=seed)
            info = train_data.is_corrupted
            clean_validation = args.get('clean_validation', True)
            if not clean_validation:
                val_data = UniformNoiseWrapper(val_data, error_prob=error_prob, num_classes=2, seed=seed)

        if not build_loaders:
            return train_data, val_data, test_data, info

        train_loader, val_loader, test_loader = nnlib.nnlib.data_utils.base.get_loaders_from_datasets(
            train_data=train_data, val_data=val_data, test_data=test_data, **args)

        return train_loader, val_loader, test_loader, info

    @nnlib.nnlib.data_utils.base.register_parser(_parsers, 'cifar10-cat-vs-dog')
    def _parse_cifar10_cat_vs_dog(self, args, build_loaders=True):
        args = copy.deepcopy(args)
        num_train_examples = args.pop('num_train_examples', None)

        from nnlib.nnlib.data_utils.cifar import CIFAR
        args['num_classes'] = 10
        data_builder = CIFAR(**args)

        train_data, val_data, test_data, info = data_builder.build_datasets(**args)
        train_data = BinaryDatasetWrapper(train_data, which_labels=(3, 5))
        val_data = BinaryDatasetWrapper(val_data, which_labels=(3, 5))
        test_data = BinaryDatasetWrapper(test_data, which_labels=(3, 5))

        if args.pop('resize_to_imagenet', False):
            train_data = ResizeImagesWrapper(train_data)
            val_data = ResizeImagesWrapper(val_data)
            test_data = ResizeImagesWrapper(test_data)

        # trim down validation and training sets to num_train_examples
        if num_train_examples is not None:
            np.random.seed(args.get('seed', 42))
            if len(train_data) > num_train_examples:
                train_indices = np.random.choice(len(train_data), size=num_train_examples, replace=False)
                train_data = SubsetDataWrapper(train_data, include_indices=train_indices)

            if len(val_data) > num_train_examples:
                val_indices = np.random.choice(len(val_data), size=num_train_examples, replace=False)
                val_data = SubsetDataWrapper(val_data, include_indices=val_indices)

        # add label noise
        seed = args.get('seed', 42)
        error_prob = args.get('error_prob', 0.0)
        if error_prob > 0.0:
            train_data = UniformNoiseWrapper(train_data, error_prob=error_prob, num_classes=2, seed=seed)
            info = train_data.is_corrupted
            clean_validation = args.get('clean_validation', True)
            if not clean_validation:
                val_data = UniformNoiseWrapper(val_data, error_prob=error_prob, num_classes=2, seed=seed)

        if not build_loaders:
            return train_data, val_data, test_data, info

        train_loader, val_loader, test_loader = nnlib.nnlib.data_utils.base.get_loaders_from_datasets(
            train_data=train_data, val_data=val_data, test_data=test_data, **args)

        return train_loader, val_loader, test_loader, info

    # NOTE: here we override the nnlib's parser of cats-and-dogs
    @nnlib.nnlib.data_utils.base.register_parser(_parsers, 'cats-and-dogs')
    def _parse_cats_and_dogs(self, args, build_loaders=True):
        args = copy.deepcopy(args)
        num_train_examples = args.pop('num_train_examples', None)

        from nnlib.nnlib.data_utils.cats_and_dogs import CatsAndDogs
        data_builder = CatsAndDogs(**args)

        train_data, val_data, test_data, info = data_builder.build_datasets(**args)

        # trim down validation and training sets to num_train_examples
        if num_train_examples is not None:
            np.random.seed(args.get('seed', 42))
            if len(train_data) > num_train_examples:
                train_indices = np.random.choice(len(train_data), size=num_train_examples, replace=False)
                train_data = SubsetDataWrapper(train_data, include_indices=train_indices)

            if len(val_data) > num_train_examples:
                val_indices = np.random.choice(len(val_data), size=num_train_examples, replace=False)
                val_data = SubsetDataWrapper(val_data, include_indices=val_indices)

        # set testing equal to validation set, because there is no test set
        test_data = val_data

        # add label noise
        seed = args.get('seed', 42)
        error_prob = args.get('error_prob', 0.0)
        if error_prob > 0.0:
            train_data = UniformNoiseWrapper(train_data, error_prob=error_prob, num_classes=2, seed=seed)
            info = train_data.is_corrupted
            clean_validation = args.get('clean_validation', True)
            if not clean_validation:
                val_data = UniformNoiseWrapper(val_data, error_prob=error_prob, num_classes=2, seed=seed)

        if not build_loaders:
            return train_data, val_data, test_data, info

        train_loader, val_loader, test_loader = nnlib.nnlib.data_utils.base.get_loaders_from_datasets(
            train_data=train_data, val_data=val_data, test_data=test_data, **args)

        return train_loader, val_loader, test_loader, info

    # NOTE: here we override the nnlib's parser of cassava
    @nnlib.nnlib.data_utils.base.register_parser(_parsers, 'cassava')
    def _parse_cassava(self, args, build_loaders=True):
        args = copy.deepcopy(args)

        from nnlib.nnlib.data_utils.cassava import Cassava
        data_builder = Cassava(**args)

        train_data, val_data, test_data, info = data_builder.build_datasets(**args)

        # add label noise
        seed = args.get('seed', 42)
        error_prob = args.get('error_prob', 0.0)
        if error_prob > 0.0:
            train_data = UniformNoiseWrapper(train_data, error_prob=error_prob, num_classes=5, seed=seed)
            info = train_data.is_corrupted
            clean_validation = args.get('clean_validation', True)
            if not clean_validation:
                val_data = UniformNoiseWrapper(val_data, error_prob=error_prob, num_classes=5, seed=seed)

        train_data = OneHotLabelWrapper(train_data, num_classes=5)
        val_data = OneHotLabelWrapper(val_data, num_classes=5)
        test_data = OneHotLabelWrapper(test_data, num_classes=5)

        if not build_loaders:
            return train_data, val_data, test_data, info

        train_loader, val_loader, test_loader = nnlib.nnlib.data_utils.base.get_loaders_from_datasets(
            train_data=train_data, val_data=val_data, test_data=test_data, **args)

        return train_loader, val_loader, test_loader, info

    @nnlib.nnlib.data_utils.base.register_parser(_parsers, 'synthetic')
    def _parse_synthetic(self, args, build_loaders=True):
        extra_dim = args.get('extra_dim', 100)
        num_examples = args.get('num_examples', 600)
        data_X, data_Y, info = get_synthetic_data(args['seed'], extra_dim=extra_dim,
                                                  num_examples=num_examples)
        half = len(data_X) // 2
        train_data = torch.utils.data.TensorDataset(torch.tensor(data_X[:half]).float(),
                                                    torch.tensor(data_Y[:half]).long().reshape((-1, 1)))
        val_data = torch.utils.data.TensorDataset(torch.tensor(data_X[half:]).float(),
                                                  torch.tensor(data_Y[half:]).long().reshape((-1, 1)))
        test_data = None

        train_data.dataset_name = None
        train_data.statistics = None
        val_data.dataset_name = None
        val_data.statistics = None

        if not build_loaders:
            return train_data, val_data, test_data, info

        train_loader, val_loader, test_loader = nnlib.nnlib.data_utils.base.get_loaders_from_datasets(
            train_data=train_data, val_data=val_data, test_data=test_data, **args)

        return train_loader, val_loader, test_loader, info

    @nnlib.nnlib.data_utils.base.register_parser(_parsers, 'nike')
    def _parse_nike(self, args, build_loaders=True):
        data_builder = MetisDataset(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)

    @nnlib.nnlib.data_utils.base.register_parser(_parsers, 'beverages')
    def _parse_beverages(self, args, build_loaders=True):
        data_builder = MetisDataset(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)


def load_data_from_arguments(args, build_loaders=True):
    return DataSelector().parse(args, build_loaders=build_loaders)
