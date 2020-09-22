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
import os

from nnlib.nnlib import utils
from sample_info.modules.data_utils import load_data_from_arguments, CacheDatasetWrapper
from sample_info.modules.ntk import prepare_needed_items
from sample_info.modules.stability import test_pred_stability, weight_stability
from sample_info.modules.misc import process_results
from sample_info import methods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--device', '-d', default='cuda', help='specifies the main device')
    parser.add_argument('--seed', type=int, default=42)

    # data parameters
    parser.add_argument('--dataset', '-D', type=str, default='mnist4vs9')
    parser.add_argument('--data_augmentation', '-A', action='store_true', dest='data_augmentation')
    parser.set_defaults(data_augmentation=False)
    parser.add_argument('--error_prob', '-n', type=float, default=0.0)
    parser.add_argument('--num_train_examples', type=int, default=None)
    parser.add_argument('--clean_validation', action='store_true', default=False)
    parser.add_argument('--resize_to_imagenet', action='store_true', dest='resize_to_imagenet')
    parser.set_defaults(resize_to_imagenet=False)
    parser.add_argument('--cache_dataset', action='store_true', dest='cache_dataset')
    parser.set_defaults(cache_dataset=False)

    # hyper-parameters
    parser.add_argument('--model_class', '-m', type=str, default='ClassifierL2')

    parser.add_argument('--l2_reg_coef', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')

    parser.add_argument('--output_dir', '-o', type=str, default='sample_info/results/ground-truth/informativeness')
    parser.add_argument('--exp_name', '-E', type=str, required=True)

    # which measures to compute
    parser.add_argument('--which_measures', '-w', type=str, nargs='+', required=True,
                        help="Options are 'weights-full', 'weights-plain', and 'predictions'")

    # NTK arguments
    parser.add_argument('--t', '-t', type=int, default=None)
    parser.add_argument('--projection', type=str, default='none', choices=['none', 'random-subset', 'very-sparse'])
    parser.add_argument('--cpu', dest='cpu', action='store_true')
    parser.set_defaults(cpu=False)
    parser.add_argument('--large_model_regime', dest='large_model_regime', action='store_true')
    parser.set_defaults(large_model_regime=False)
    parser.add_argument('--return_change_vectors', dest='return_change_vectors', action='store_true')
    parser.set_defaults(return_change_vectors=False)

    args = parser.parse_args()
    print(args)

    # Load data
    train_data, val_data, test_data, _ = load_data_from_arguments(args, build_loaders=False)
    if args.cache_dataset:
        train_data = CacheDatasetWrapper(train_data)
        val_data = CacheDatasetWrapper(val_data)
        test_data = CacheDatasetWrapper(test_data)

    with open(args.config, 'r') as f:
        architecture_args = json.load(f)

    model_class = getattr(methods, args.model_class)

    model = model_class(input_shape=train_data[0][0].shape,
                        architecture_args=architecture_args,
                        l2_reg_coef=args.l2_reg_coef,
                        device=args.device,
                        seed=args.seed)
    model.eval()
    print("Number of parameters: ", utils.get_num_parameters(model))

    # Prepare the needed terms
    ret = prepare_needed_items(model=model, train_data=train_data, test_data=val_data,
                               projection=args.projection, cpu=args.cpu)
    n = len(train_data)

    exp_dir = os.path.join(args.output_dir, args.exp_name)

    # weights with SGD
    if 'weights-full' in args.which_measures:
        vectors, quantities = weight_stability(t=args.t, n=n, eta=args.lr / n, init_params=ret['init_params'],
                                               jacobians=ret['train_jacobians'], ntk=ret['ntk'],
                                               init_preds=ret['train_init_preds'], Y=ret['train_Y'],
                                               l2_reg_coef=n * args.l2_reg_coef, continuous=False,
                                               without_sgd=False, model=model, dataset=train_data,
                                               large_model_regime=args.large_model_regime,
                                               return_change_vectors=False)

        meta = {
            'description': f'weights (full) at epoch {args.t}',
            'continuous': False,
            'args': args
        }

        process_results(vectors=vectors, quantities=quantities, meta=meta,
                        exp_name=f'weight-full-t{args.t}', output_dir=exp_dir, train_data=train_data)

    # weights without SGD
    if 'weights-plain' in args.which_measures:
        vectors, quantities = weight_stability(t=args.t, n=n, eta=args.lr / n, init_params=ret['init_params'],
                                               jacobians=ret['train_jacobians'], ntk=ret['ntk'],
                                               init_preds=ret['train_init_preds'], Y=ret['train_Y'],
                                               l2_reg_coef=n * args.l2_reg_coef, continuous=False,
                                               without_sgd=True, model=model, dataset=train_data,
                                               large_model_regime=args.large_model_regime,
                                               return_change_vectors=args.return_change_vectors)

        meta = {
            'description': f'weights (plain) at epoch {args.t}',
            'continuous': False,
            'args': args
        }

        process_results(vectors=vectors, quantities=quantities, meta=meta,
                        exp_name=f'weight-plain-t{args.t}', output_dir=exp_dir, train_data=train_data)

    # test prediction
    if 'predictions' in args.which_measures:
        vectors, quantities = test_pred_stability(t=args.t, n=n, eta=args.lr / n, ntk=ret['ntk'],
                                                  test_train_ntk=ret['test_train_ntk'],
                                                  train_init_preds=ret['train_init_preds'],
                                                  test_init_preds=ret['test_init_preds'],
                                                  train_Y=ret['train_Y'],
                                                  l2_reg_coef=n * args.l2_reg_coef,
                                                  continuous=False)
        meta = {
            'description': f'validation predictions at epoch {args.t}',
            'continuous': False,
            'args': args
        }

        process_results(vectors=vectors, quantities=quantities, meta=meta,
                        exp_name=f'predictions-t{args.t}', output_dir=exp_dir, train_data=train_data)


if __name__ == '__main__':
    main()
