import os

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

from .base import print_loaded_dataset_shapes, log_call_parameters


class DSpritesDataset(Dataset):
    def __init__(self, indices, classification=False, colored=False, data_file=None):
        super(DSpritesDataset, self).__init__()

        if data_file is None:
            data_file = os.path.join(os.environ['DATA_DIR'], 'dsprites-dataset',
                                     'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(data_file, encoding='latin1', allow_pickle=True)

        self.indices = indices

        # color related stuff
        self.colored = colored
        self.colors = None
        self.n_colors = 1
        indices_without_color = indices
        if colored:
            color_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'resources/rainbow-7.npy')
            self.colors = np.load(color_file)
            self.n_colors = len(self.colors)
            indices_without_color = [idx // self.n_colors for idx in indices]

        # factor_names and factor_sizes
        meta = data['metadata'].item()
        self.factor_names = list(meta['latents_names'][1:])
        self.factor_sizes = list(meta['latents_sizes'][1:])
        if colored:
            self.factor_names.append('color')
            self.factor_sizes.append(self.n_colors)
        self.n_factors = len(self.factor_names)

        # save relevant part of the grid
        self.imgs = data['imgs'][indices_without_color]

        # factor values, classes, possible_values
        self.factor_values = data['latents_values'][indices_without_color]
        self.factor_values = [arr[1:] for arr in self.factor_values]

        self.factor_classes = data['latents_classes'][indices_without_color]
        self.factor_classes = [arr[1:] for arr in self.factor_classes]

        self.possible_values = []
        for name in ['shape', 'scale', 'orientation', 'posX', 'posY']:
            self.possible_values.append(meta['latents_possible_values'][name])

        if colored:
            for i, idx in enumerate(self.indices):
                color_class = idx % self.n_colors
                color_value = color_class / (self.n_colors - 1.0)
                self.factor_classes[i] = np.append(self.factor_classes[i], color_class)
                self.factor_values[i] = np.append(self.factor_values[i], color_value)
            self.possible_values.append(list(np.arange(0, self.n_colors) / (self.n_colors - 1.0)))

        self.classification = classification

        # dataset name
        self.dataset_name = 'dsprites'
        if self.colored:
            self.dataset_name += '-colored'

        # factor types
        self.is_categorical = [True, False, False, False, False]
        if self.colored:
            self.is_categorical.append(True)

        # normalization values
        means = torch.tensor([0.456])
        stds = torch.tensor([0.224])
        if self.colored:
            means = torch.tensor([0.485, 0.456, 0.406])
            stds = torch.tensor([0.229, 0.224, 0.225])
        self.statistics = (means, stds)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        sample = self.imgs[idx]
        sample = torch.tensor(sample, dtype=torch.float).unsqueeze(dim=0)  # (1, H, W)

        factors = (self.factor_classes[idx] if self.classification else self.factor_values[idx])
        factors = torch.tensor(factors, dtype=torch.float)

        if self.colored:
            color_class = self.factor_classes[idx][-1]
            color = self.colors[color_class]
            sample = torch.cat([sample * color[0], sample * color[1], sample * color[2]], dim=0)  # (3, H, W)

        means, stds = self.statistics
        sample = ((sample - means.unsqueeze(dim=-1).unsqueeze(dim=-1)) /
                  stds.unsqueeze(dim=-1).unsqueeze(dim=-1))

        return sample, factors


@print_loaded_dataset_shapes
@log_call_parameters
def load_dsprites_datasets(val_ratio=0.2, test_ratio=0.2, seed=42,
                           classification=False, colored=False, data_file=None, **kwargs):
    N = 737280
    if colored:
        N *= 7
    val_cnt = int(val_ratio * N)
    test_cnt = int(test_ratio * N)
    train_cnt = N - val_cnt - test_cnt

    np.random.seed(seed)
    perm = np.random.permutation(N)
    train_indices = perm[:train_cnt]
    val_indices = perm[train_cnt:train_cnt + val_cnt]
    test_indices = perm[train_cnt + val_cnt:]

    train_dataset = DSpritesDataset(indices=train_indices, colored=colored,
                                    classification=classification, data_file=data_file)
    val_dataset = DSpritesDataset(indices=val_indices, colored=colored,
                                  classification=classification, data_file=data_file)
    test_dataset = DSpritesDataset(indices=test_indices, colored=colored,
                                   classification=classification, data_file=data_file)

    return train_dataset, val_dataset, test_dataset, None


@log_call_parameters
def load_dsprites_loaders(val_ratio=0.2, test_ratio=0.2, batch_size=128,
                          seed=42, classification=False, colored=False,
                          drop_last=False, **kwargs):

    train_dataset, val_dataset, test_dataset, info = load_dsprites_datasets(
        val_ratio=val_ratio, test_ratio=test_ratio, seed=seed,
        classification=classification, colored=colored, **kwargs)

    train_loader = None
    val_loader = None
    test_loader = None

    if len(train_dataset) > 0:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=4, drop_last=drop_last)
    if len(val_dataset) > 0:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4, drop_last=drop_last)
    if len(test_dataset) > 0:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=4, drop_last=drop_last)

    return train_loader, val_loader, test_loader, info
