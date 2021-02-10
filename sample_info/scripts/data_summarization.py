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

import argparse
import json
import pickle
import os

import numpy as np

from nnlib.nnlib import training, metrics, callbacks, utils
from sample_info.modules.data_utils import load_data_from_arguments, SubsetDataWrapper, CacheDatasetWrapper
from nnlib.nnlib.data_utils.base import get_loaders_from_datasets
from sample_info import methods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--device', '-d', default='cuda', help='specifies the main device')
    parser.add_argument('--all_device_ids', nargs='+', type=str, default=None,
                        help="If not None, this list specifies devices for multiple GPU training. "
                             "The first device should match with the main device (args.device).")
    parser.add_argument('--batch_size', '-b', type=int, default=256)
    parser.add_argument('--epochs', '-e', type=int, default=400)
    parser.add_argument('--stopping_param', type=int, default=2**30)
    parser.add_argument('--save_iter', '-s', type=int, default=2**30)
    parser.add_argument('--vis_iter', '-v', type=int, default=2**30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_accumulation_steps', default=1, type=int,
                        help='Number of training steps to accumulate before updating weights')

    # data parameters
    parser.add_argument('--dataset', '-D', type=str, default='mnist')
    parser.add_argument('--data_augmentation', '-A', action='store_true', dest='data_augmentation')
    parser.set_defaults(data_augmentation=False)
    parser.add_argument('--error_prob', '-n', type=float, default=0.0)
    parser.add_argument('--num_train_examples', type=int, default=None)
    parser.add_argument('--clean_validation', action='store_true', default=False)
    parser.add_argument('--resize_to_imagenet', action='store_true', dest='resize_to_imagenet')
    parser.set_defaults(resize_to_imagenet=False)
    parser.add_argument('--cache_dataset', action='store_true', dest='cache_dataset')
    parser.set_defaults(cache_dataset=False)
    parser.add_argument('--sample_ranking_file', type=str, default=None,
                        help='Points to a pickle file that stores an ordering of examples from least to '
                             'most important. The most important args.exclude_ratio number of samples '
                             'will be excluded from training.')
    parser.add_argument('--exclude_ratio', type=float, default=0.0,
                        help='Fraction of examples to exclude.')
    parser.add_argument('--exclude_side', type=str, default='top', choices=['top', 'bottom'],
                        help='from which side of the order to remove')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers in data loaders')

    # hyper-parameters
    parser.add_argument('--model_class', '-m', type=str, default='ClassifierL2')

    parser.add_argument('--l2_reg_coef', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--random_baseline_seed', type=int, default=42)

    parser.add_argument('--output_dir', '-o', type=str, default='sample_info/results/data-summarization/')
    parser.add_argument('--baseline_name', '-B', type=str, required=True)
    parser.add_argument('--exp_name', '-E', type=str, required=True)

    args = parser.parse_args()
    print(args)

    # set tensorboard log directory
    args.log_dir = os.path.join(args.output_dir, args.baseline_name, args.exp_name, 'logs')
    utils.make_path(args.log_dir)

    # Load data
    train_data, val_data, test_data, _ = load_data_from_arguments(args, build_loaders=False)

    # exclude samples
    np.random.seed(args.random_baseline_seed)
    order = np.random.permutation(len(train_data))

    # if sample ranking file is given, take the order from there
    if args.sample_ranking_file is not None:
        with open(args.sample_ranking_file, 'rb') as f:
            order = pickle.load(f)

    exclude_count = int(args.exclude_ratio * len(train_data))
    if exclude_count == 0:
        exclude_indices = []
    else:
        if args.exclude_side == 'top':
            exclude_indices = order[-exclude_count:]
        else:
            exclude_indices = order[:exclude_count]

    train_data = SubsetDataWrapper(dataset=train_data, exclude_indices=exclude_indices)

    if args.cache_dataset:
        train_data = CacheDatasetWrapper(train_data)
        val_data = CacheDatasetWrapper(val_data)
        test_data = CacheDatasetWrapper(test_data)

    shuffle_train = (args.batch_size * args.num_accumulation_steps < len(train_data))
    train_loader, val_loader, test_loader = get_loaders_from_datasets(train_data, val_data, test_data,
                                                                      batch_size=args.batch_size,
                                                                      num_workers=args.num_workers,
                                                                      shuffle_train=shuffle_train)

    # Options
    optimization_args = {
        'optimizer': {
            'name': args.optimizer,
            'lr': args.lr,
        }
    }

    with open(args.config, 'r') as f:
        architecture_args = json.load(f)

    model_class = getattr(methods, args.model_class)

    model = model_class(input_shape=train_loader.dataset[0][0].shape,
                        architecture_args=architecture_args,
                        l2_reg_coef=args.l2_reg_coef,
                        device=args.device,
                        seed=args.seed)

    # put the model in always eval mode. This makes sure that in case the network has pretrained BatchNorm
    # layers, their running average is fixed.
    utils.put_always_eval_mode(model)

    metrics_list = [metrics.Accuracy(output_key='pred', one_hot=(train_data[0][1].ndim > 0))]
    if args.dataset == 'imagenet':
        metrics_list.append(metrics.TopKAccuracy(k=5, output_key='pred'))

    stopper = callbacks.EarlyStoppingWithMetric(metric=metrics_list[0], stopping_param=args.stopping_param,
                                                partition='val', direction='max')

    training.train(model=model,
                   train_loader=train_loader,
                   val_loader=val_loader,
                   epochs=args.epochs,
                   save_iter=args.save_iter,
                   vis_iter=args.vis_iter,
                   optimization_args=optimization_args,
                   log_dir=args.log_dir,
                   args_to_log=args,
                   stopper=stopper,
                   metrics=metrics_list,
                   device_ids=args.all_device_ids,
                   num_accumulation_steps=args.num_accumulation_steps)

    val_preds = utils.apply_on_dataset(model=model, dataset=val_data, cpu=True,
                                       partition='val', batch_size=args.batch_size)['pred']
    val_acc = metrics_list[0].value(epoch=args.epochs-1, partition='val')

    file_name = f'results-{args.exclude_ratio:.4f}'
    if args.baseline_name == 'random':
        file_name += f'-{args.random_baseline_seed}'
    file_name += '.pkl'
    file_path = os.path.join(args.output_dir, args.baseline_name, args.exp_name, file_name)
    utils.make_path(os.path.dirname(file_path))
    with open(file_path, 'wb') as f:
        pickle.dump({
            'val_preds': val_preds,
            'val_acc': val_acc,
            'args': args
        }, f)


if __name__ == '__main__':
    main()
