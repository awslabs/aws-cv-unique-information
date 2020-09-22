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


from nnlib.nnlib import training, metrics, utils
from sample_info.modules.data_utils import load_data_from_arguments, SubsetDataWrapper, CacheDatasetWrapper
from nnlib.nnlib.data_utils.base import get_loaders_from_datasets
from sample_info import methods
from sample_info.methods import LinearizedModelV2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--device', '-d', default='cuda', help='specifies the main device')
    parser.add_argument('--all_device_ids', nargs='+', type=str, default=None,
                        help="If not None, this list specifies devices for multiple GPU training. "
                             "The first device should match with the main device (args.device).")
    parser.add_argument('--batch_size', '-b', type=int, default=2 ** 20)
    parser.add_argument('--epochs', '-e', type=int, default=2000)
    parser.add_argument('--stopping_param', type=int, default=2 ** 20)
    parser.add_argument('--save_iter', '-s', type=int, default=2 ** 20)
    parser.add_argument('--vis_iter', '-v', type=int, default=2 ** 20)
    parser.add_argument('--log_dir', '-l', type=str, default='sample_info/logs/junk')
    parser.add_argument('--seed', type=int, default=42)

    # data parameters
    parser.add_argument('--dataset', '-D', type=str, default='mnist4vs9',
                        choices=['mnist4vs9', 'synthetic', 'cifar10-cat-vs-dog'],
                        help='Which dataset to use. One can add more choices if needed.')
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
    parser.add_argument('--exclude_index', type=int, default=None, help='Index of an example to remove.')

    # hyper-parameters
    parser.add_argument('--model_class', '-m', type=str, default='ClassifierL2')
    parser.add_argument('--linearized', dest='linearized', action='store_true')
    parser.set_defaults(linearized=False)

    parser.add_argument('--l2_reg_coef', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'sgd'])

    parser.add_argument('--output_dir', '-o', type=str, default='sample_info/results/ground-truth/ground-truth/')
    parser.add_argument('--exp_name', '-E', type=str, required=True)
    args = parser.parse_args()
    print(args)

    # Build data
    train_data, val_data, test_data, _ = load_data_from_arguments(args, build_loaders=False)

    # exclude the example
    if args.exclude_index is not None:
        train_data = SubsetDataWrapper(dataset=train_data, exclude_indices=[args.exclude_index])

    if args.cache_dataset:
        train_data = CacheDatasetWrapper(train_data)
        val_data = CacheDatasetWrapper(val_data)
        test_data = CacheDatasetWrapper(test_data)

    shuffle_train = (args.batch_size < len(train_data))
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
                        seed=args.seed,
                        device=args.device)

    # put the model in always eval mode. This makes sure that in case the network has pretrained BatchNorm
    # layers, their running average is fixed.
    utils.put_always_eval_mode(model)

    if args.linearized:
        print("Using a linearized model")
        model = LinearizedModelV2(model=model,
                                  train_data=train_data,
                                  val_data=val_data,
                                  l2_reg_coef=args.l2_reg_coef)

    if args.dataset == 'synthetic':
        model.visualize = (lambda *args, **kwargs: {})  # no visualization is needed

    metrics_list = [metrics.Accuracy(output_key='pred')]

    training.train(model=model,
                   train_loader=train_loader,
                   val_loader=val_loader,
                   epochs=args.epochs+1,
                   save_iter=args.save_iter,
                   vis_iter=args.vis_iter,
                   optimization_args=optimization_args,
                   log_dir=args.log_dir,
                   args_to_log=args,
                   metrics=metrics_list,
                   device_ids=args.all_device_ids)

    params = dict(model.named_parameters())
    for k in params.keys():
        params[k] = utils.to_cpu(params[k])
    val_preds = utils.apply_on_dataset(model=model, dataset=val_data, cpu=True,
                                       partition='val', batch_size=2**30)['pred']
    val_acc = metrics_list[0].value(epoch=args.epochs, partition='val')

    exp_dir = os.path.join(args.output_dir, args.exp_name)

    # if it the the full dataset save params and val_preds, otherwise compare to the saved weights/predictions
    if args.exclude_index is None:
        file_path = os.path.join(exp_dir, 'full-data-training.pkl')
    else:
        file_path = os.path.join(exp_dir, f'{args.exclude_index}.pkl')

    utils.make_path(os.path.dirname(file_path))
    with open(file_path, 'wb') as f:
        pickle.dump({
            'weights': params,
            'val_preds': val_preds,
            'val_acc': val_acc,
            'args': args
        }, f)


if __name__ == '__main__':
    main()
