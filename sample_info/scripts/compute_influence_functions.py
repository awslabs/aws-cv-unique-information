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

from tqdm import tqdm
import torch

from nnlib.nnlib import gradients, utils
from sample_info.modules.data_utils import load_data_from_arguments, CacheDatasetWrapper
from sample_info import methods
from sample_info.modules.misc import process_results
from sample_info.modules.influence_functions import inverse_hvp_lissa
from sample_info.modules.ntk import JacobianEstimator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--device', '-d', default='cuda', help='specifies the main device')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', dest='cpu', action='store_true')
    parser.set_defaults(cpu=False)

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

    # hyper-parameters
    parser.add_argument('--model_class', '-m', type=str, default='ClassifierL2')

    parser.add_argument('--l2_reg_coef', type=float, default=0.0)
    parser.add_argument('--damping', type=float, default=1e-10)
    parser.add_argument('--scale', type=float, default=10.0)
    parser.add_argument('--recursion_depth', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--output_dir', '-o', type=str, default='sample_info/results/ground-truth/')
    parser.add_argument('--exp_name', '-E', type=str, required=True)
    args = parser.parse_args()
    print(args)

    # Build data
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
                        seed=args.seed,
                        device=args.device)

    # load the final parameters
    saved_file_path = os.path.join(args.output_dir, 'ground-truth', args.exp_name, 'full-data-training.pkl')
    with open(saved_file_path, 'rb') as f:
        saved_data = pickle.load(f)

    params = dict(model.named_parameters())
    for k, v in saved_data['weights'].items():
        params[k].data = v.to(args.device)

    # compute per example gradients (d loss / d weights for train and d pred / d weights for validation)
    train_grads = gradients.get_weight_gradients(model=model, dataset=train_data, cpu=args.cpu,
                                                 description='computing per example gradients on train data')

    jacobian_estimator = JacobianEstimator()
    val_grads = jacobian_estimator.compute_jacobian(model=model, dataset=val_data, cpu=args.cpu,
                                                    description='computing jacobian on validation data')

    # compute weight and prediction influences
    weight_vectors = []
    weight_quantities = []

    pred_vectors = []
    pred_quantities = []

    for sample_idx in tqdm(range(len(train_data)), desc='computing influences'):
        # compute weights
        v = []
        for k in dict(model.named_parameters()).keys():
            v.append(train_grads[k][sample_idx].to(model.device))
        inv_hvp = inverse_hvp_lissa(model, dataset=train_data, v=v, batch_size=args.batch_size,
                                    recursion_depth=args.recursion_depth, damping=args.damping,
                                    scale=args.scale)
        if args.cpu:
            inv_hvp = [utils.to_cpu(a) for a in inv_hvp]

        for a in inv_hvp:
            if torch.isnan(a).any():
                raise ValueError("Inverse hessian vector product contains NaNs. Increase the scale.")

        cur_weight_influence = 1.0 / len(train_data) * torch.cat([a.flatten() for a in inv_hvp])
        weight_vectors.append(cur_weight_influence)
        weight_quantities.append(torch.sum(cur_weight_influence ** 2))

        # compute for predictions
        cur_pred_influences = []
        for val_sample_idx in range(len(val_data)):
            val_grad_flat = []
            for k, v in dict(model.named_parameters()).items():
                val_grad_flat.append(val_grads[k][val_sample_idx].flatten())
            val_grad_flat = torch.cat(val_grad_flat, dim=0)
            cur_pred_influences.append(torch.dot(cur_weight_influence, val_grad_flat))

        cur_pred_influences = torch.stack(cur_pred_influences)
        pred_vectors.append(cur_pred_influences)
        pred_quantities.append(torch.sum(cur_pred_influences ** 2))

    # save weights
    meta = {
        'description': f'weight influence functions',
        'args': args
    }

    exp_dir = os.path.join(args.output_dir, 'influence-functions', args.exp_name)
    process_results(vectors=weight_vectors, quantities=weight_quantities, meta=meta,
                    exp_name='weights', output_dir=exp_dir, train_data=train_data)

    # save preds
    meta = {
        'description': f'pred influence functions',
        'args': args
    }

    exp_dir = os.path.join(args.output_dir, 'influence-functions', args.exp_name)
    process_results(vectors=pred_vectors, quantities=pred_quantities, meta=meta,
                    exp_name='pred', output_dir=exp_dir, train_data=train_data)


if __name__ == '__main__':
    main()
