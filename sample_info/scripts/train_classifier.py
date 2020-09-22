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

from nnlib.nnlib import training, metrics, callbacks
from sample_info.modules.data_utils import load_data_from_arguments, CacheDatasetWrapper
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
    parser.add_argument('--save_iter', '-s', type=int, default=10)
    parser.add_argument('--vis_iter', '-v', type=int, default=10)
    parser.add_argument('--log_dir', '-l', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)

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
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers in data loaders')

    # hyper-parameters
    parser.add_argument('--model_class', '-m', type=str, default='ClassifierL2')

    parser.add_argument('--l2_reg_coef', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])

    args = parser.parse_args()
    print(args)

    # Load data
    train_data, val_data, test_data, _ = load_data_from_arguments(args, build_loaders=False)

    if args.cache_dataset:
        train_data = CacheDatasetWrapper(train_data)
        val_data = CacheDatasetWrapper(val_data)
        test_data = CacheDatasetWrapper(test_data)

    train_loader, val_loader, test_loader = get_loaders_from_datasets(train_data, val_data, test_data,
                                                                      batch_size=args.batch_size,
                                                                      num_workers=args.num_workers)

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

    metrics_list = [metrics.Accuracy(output_key='pred')]
    if args.dataset == 'imagenet':
        metrics_list.append(metrics.TopKAccuracy(k=5, output_key='pred'))

    callbacks_list = [callbacks.SaveBestWithMetric(metric=metrics_list[0], partition='val', direction='max')]

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
                   callbacks=callbacks_list,
                   device_ids=args.all_device_ids)


if __name__ == '__main__':
    main()
