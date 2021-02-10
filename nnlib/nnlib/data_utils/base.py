from argparse import Namespace
import inspect

from torch.utils.data import DataLoader
import numpy as np


def get_split_indices(n_samples, val_ratio, seed):
    np.random.seed(seed)
    train_cnt = int((1 - val_ratio) * n_samples)
    perm = np.random.permutation(n_samples)
    train_indices = perm[:train_cnt]
    val_indices = perm[train_cnt:]
    return train_indices, val_indices


def revert_normalization(samples, dataset):
    """ Reverts normalization of images.
    :param samples: 3D or 4D tensor of images.
    :param dataset: should have attribute `statistics`, which is a pair (means, stds)
    :return:
    """
    means, stds = dataset.statistics
    means = means.to(samples.device)
    stds = stds.to(samples.device)
    if len(samples.shape) == 3:
        samples = samples.unsqueeze(dim=0)
    return (samples * stds.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3) +
            means.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3))


def get_loaders_from_datasets(train_data, val_data=None, test_data=None, batch_size=128,
                              num_workers=4, drop_last=False, shuffle_train=True, **kwargs):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle_train,
                              num_workers=num_workers, drop_last=drop_last)
    if val_data is not None:
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, drop_last=drop_last)
    else:
        val_loader = None

    if test_data is not None:
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, drop_last=drop_last)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader


def get_input_shape(train_data):
    example_inputs = train_data[0][0]
    if isinstance(example_inputs, (list, tuple)):
        return example_inputs[0].shape
    return example_inputs.shape


def print_loaded_dataset_shapes(build_datasets_fn):
    def wrapper(*args, **kwargs):
        train_data, val_data, test_data, info = build_datasets_fn(*args, **kwargs)
        print(f"Dataset {train_data.dataset_name} is loaded")
        print(f"\ttrain_samples: {len(train_data)}")
        if val_data is not None:
            print(f"\tval_samples: {len(val_data)}")
        if test_data is not None:
            print(f"\ttest_samples: {len(test_data)}")
        print(f"\texample_shape: {get_input_shape(train_data)}")
        return train_data, val_data, test_data, info
    return wrapper


def log_call_parameters(fn):
    """ A decorator for logging arguments of a function call. """
    def wrapper(*args, **kwargs):
        # get the signature, bind arguments, apply defaults, and convert to dictionary
        signature = inspect.signature(fn)
        bind_result = signature.bind(*args, **kwargs)
        bind_result.apply_defaults()
        argument_dict = bind_result.arguments

        # if it has self, we can get the class name
        class_name = None
        if 'self' in argument_dict:
            self = argument_dict.pop('self')
            class_name = self.__class__.__name__

        # get the name of kwargs and remove it
        kwargs_name = inspect.getfullargspec(fn).varkw
        if kwargs_name is not None:
            argument_dict.pop(kwargs_name)

        # log the call
        fn_name = fn.__name__
        if class_name is not None:
            print(f"calling function {fn_name} of {class_name} with the following parameters:")
        else:
            print(f"calling function {fn_name} with the following parameters:")

        for k, v in argument_dict.items():
            print(f"\t{k}: {v}")

        return fn(*args, **kwargs)

    return wrapper


def register_parser(registry, parser_name):
    def decorator(parser_fn):
        registry[parser_name] = parser_fn

        def wrapper(*args, **kwargs):
            return parser_fn(*args, **kwargs)

        return wrapper

    return decorator


class DataSelector:
    """ Helper class for loading data from arguments. """

    _parsers = {}  # register_parsers decorator will fill this

    def __init__(self):
        pass

    @register_parser(_parsers, 'mnist')
    def _parse_mnist(self, args, build_loaders=True):
        from .mnist import MNIST
        data_builder = MNIST(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)

    @register_parser(_parsers, 'uniform-noise-mnist')
    def _parse_uniform_noise_mnist(self, args, build_loaders=True):
        from .mnist import UniformNoiseMNIST
        data_builder = UniformNoiseMNIST(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)

    @register_parser(_parsers, 'fashion-mnist')
    def _parse_fashion_mnist(self, args, build_loaders=True):
        from .fashion_mnist import FashionMNIST
        data_builder = FashionMNIST(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)

    @register_parser(_parsers, 'cifar10')
    def _parse_cifar10(self, args, build_loaders=True):
        from .cifar import CIFAR
        args['n_classes'] = 10
        data_builder = CIFAR(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)

    @register_parser(_parsers, 'uniform-noise-cifar10')
    def _parse_uniform_noise_cifar10(self, args, build_loaders=True):
        from .cifar import UniformNoiseCIFAR
        args['n_classes'] = 10
        data_builder = UniformNoiseCIFAR(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)

    @register_parser(_parsers, 'pair-noise-cifar10')
    def _parse_pair_noise_cifar10(self, args, build_loaders=True):
        from .cifar import PairNoiseCIFAR10
        data_builder = PairNoiseCIFAR10(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)

    @register_parser(_parsers, 'cifar100')
    def _parse_cifar100(self, args, build_loaders=True):
        from .cifar import CIFAR
        args['n_classes'] = 100
        data_builder = CIFAR(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)

    @register_parser(_parsers, 'uniform-noise-cifar100')
    def _parse_uniform_noise_cifar100(self, args, build_loaders=True):
        from .cifar import UniformNoiseCIFAR
        args['n_classes'] = 100
        data_builder = UniformNoiseCIFAR(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)

    @register_parser(_parsers, 'imagenet')
    def _parse_imagenet(self, args, build_loaders=True):
        from .imagenet import ImageNet
        if 'num_workers' not in args:
            args['num_workers'] = 10
        if 'download' not in args:
            args['download'] = False  # ImageNet cannot be downloaded automatically
        data_builder = ImageNet(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)

    @register_parser(_parsers, 'dsprites')
    def _parse_dsprites(self, args, build_loaders=True):
        from .dsprites import load_dsprites_loaders, load_dsprites_datasets
        if build_loaders:
            return load_dsprites_loaders(**args)
        else:
            return load_dsprites_datasets(**args)

    @register_parser(_parsers, 'clothing1m')
    def _parse_clothing1m(self, args, build_loaders=True):
        from .clothing1m import Clothing1M
        if 'num_workers' not in args:
            args['num_workers'] = 10
        data_builder = Clothing1M(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)

    @register_parser(_parsers, 'cars')
    def _parse_cars(self, args, build_loaders=True):
        from .cars import Cars
        data_builder = Cars(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)

    @register_parser(_parsers, 'birds')
    def _parse_birds(self, args, build_loaders=True):
        from .birds import Birds
        data_builder = Birds(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)

    @register_parser(_parsers, 'celeba')
    def _parse_celeba(self, args, build_loaders=True):
        from .celeba import CelebA
        data_builder = CelebA(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)

    @register_parser(_parsers, 'svhn')
    def _parse_svhn(self, args, build_loaders=True):
        from .svhn import SVHN
        data_builder = SVHN(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)

    @register_parser(_parsers, 'emnist')
    def _parse_emnist(self, args, build_loaders=True):
        from .emnist import EMNIST
        data_builder = EMNIST(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)

    @register_parser(_parsers, 'cats-and-dogs')
    def _parse_cats_and_dogs(self, args, build_loaders=True):
        from .cats_and_dogs import CatsAndDogs
        data_builder = CatsAndDogs(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)

    @register_parser(_parsers, 'cassava')
    def _parse_icassava(self, args, build_loaders=True):
        from .cassava import Cassava
        data_builder = Cassava(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)

    @register_parser(_parsers, 'oxford-flowers-102')
    def _parse_oxford_flowers_102(self, args, build_loaders=True):
        from .oxford_flowers_102 import OxfordFlowers102
        data_builder = OxfordFlowers102(**args)
        if build_loaders:
            return data_builder.build_loaders(**args)
        else:
            return data_builder.build_datasets(**args)

    def can_parse(self, dataset_name):
        return dataset_name in self._parsers

    def parse(self, args, build_loaders=True):
        """ Loads a dataset from a Namespace or dict. """
        if isinstance(args, Namespace):
            args = vars(args).copy()
        elif isinstance(args, dict):
            args = args.copy()
        else:
            raise ValueError(f"Type of args: {type(args)} is not supported")

        if 'dataset' not in args:
            raise ValueError('Variable "dataset" is missing')

        dataset = args['dataset']
        if not self.can_parse(dataset):
            raise ValueError(f"Value {dataset} for dataset is not recognized")

        parser = self._parsers[dataset]
        return parser(self, args, build_loaders=build_loaders)


def load_data_from_arguments(args, build_loaders=True):
    return DataSelector().parse(args, build_loaders=build_loaders)
