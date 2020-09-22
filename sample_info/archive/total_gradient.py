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

import torch

from nnlib.nnlib import training, metrics
from nnlib.nnlib.data_utils.base import get_loaders_from_datasets
from sample_info.modules.data_utils import load_data_from_arguments, BinaryDatasetWrapper, ReturnSampleIndexWrapper
from sample_info.modules.misc import process_results
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
    parser.add_argument('--dataset', '-D', type=str, default='corrupt4_mnist')
    parser.add_argument('--data_augmentation', '-A', action='store_true', dest='data_augmentation')
    parser.set_defaults(data_augmentation=False)
    parser.add_argument('--error_prob', '-n', type=float, default=0.0)
    parser.add_argument('--num_train_examples', type=int, default=None)
    parser.add_argument('--clean_validation', action='store_true', default=False)

    # hyper-parameters
    parser.add_argument('--model_class', '-m', type=str, default='ClassifierL2WithGradCollector')

    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])

    parser.add_argument('--output_dir', '-o', type=str, default='results/stability/mnist-4vs9-1000-samples/')
    args = parser.parse_args()
    print(args)

    # Load data
    # TODO: remove hard coding
    train_data, val_data, test_data, _ = load_data_from_arguments({'dataset': 'mnist',
                                                                   'num_train_examples': 10 * 500},
                                                                  build_loaders=False)
    train_data = BinaryDatasetWrapper(train_data, which_labels=(4, 9))
    val_data = BinaryDatasetWrapper(val_data, which_labels=(4, 9))
    test_data = BinaryDatasetWrapper(test_data, which_labels=(4, 9))

    train_data = ReturnSampleIndexWrapper(train_data)
    val_data = ReturnSampleIndexWrapper(val_data)
    test_data = ReturnSampleIndexWrapper(test_data)

    train_loader, val_loader, test_loader = get_loaders_from_datasets(train_data, val_data, test_data,
                                                                      batch_size=2 ** 30,
                                                                      shuffle_train=False, num_workers=0)

    # Options
    optimization_args = {
        'optimizer': {
            'name': args.optimizer,
            'lr': args.lr,
            'weight_decay': args.weight_decay
        }
    }

    with open(args.config, 'r') as f:
        architecture_args = json.load(f)

    ts = range(100, 401, 100)

    for t in ts:
        model_class = getattr(methods, args.model_class)

        model = model_class(input_shape=train_loader.dataset[0][0][0].shape,
                            architecture_args=architecture_args,
                            device=args.device,
                            seed=args.seed)

        metrics_list = [metrics.Accuracy(output_key='pred')]

        training.train(model=model,
                       train_loader=train_loader,
                       val_loader=val_loader,
                       epochs=t,
                       save_iter=args.save_iter,
                       vis_iter=2**30,  # NOTE: never visualize
                       optimization_args=optimization_args,
                       log_dir=args.log_dir,
                       args_to_log=args,
                       metrics=metrics_list,
                       device_ids=args.all_device_ids)

        vectors = model._grad_updates

        norms = []
        for i in range(len(train_data)):
            grad_dict = vectors[i]
            norm = 0.0
            for k, v in grad_dict.items():
                norm += torch.norm(v.flatten())
            norms.append(norm)

        quantities = norms

        meta = {
            'description': 'Total gradient update per example. The measures are the norm of total gradient update.',
            'time': t,
            'continuous': False,
            'args': args
        }

        process_results(vectors=vectors, quantities=quantities, meta=meta,
                        exp_name=f'total-grad-t{t}', output_dir=args.output_dir, train_data=train_data.dataset)


if __name__ == '__main__':
    main()
