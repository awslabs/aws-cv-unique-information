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

from nnlib.nnlib import utils, gradients
from sample_info.modules.data_utils import load_data_from_arguments, CacheDatasetWrapper
from nnlib.nnlib.data_utils.base import get_loaders_from_datasets
from sample_info import methods
from sample_info.modules.misc import process_results
from sample_info.modules.influence_functions import hessian
from sample_info.modules.ntk import JacobianEstimator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--device', '-d', default='cuda', help='specifies the main device')
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

    # hyper-parameters
    parser.add_argument('--model_class', '-m', type=str, default='ClassifierL2')

    parser.add_argument('--l2_reg_coef', type=float, default=0.0)

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

    train_loader, val_loader, test_loader = get_loaders_from_datasets(train_data, val_data, test_data,
                                                                      batch_size=2 ** 30,
                                                                      shuffle_train=False)

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

    # brute force compute hessian and its inverse
    total_loss = 0.0
    for x, y in train_loader:
        out = model.forward(inputs=[x], labels=[y])
        losses, _ = model.compute_loss(inputs=[x], labels=[y], outputs=out)
        total_loss = total_loss + sum([v for k, v in losses.items()])

    with utils.Timing(description='Computing the Hessian'):
        H = hessian(ys=[total_loss], xs=tuple(model.parameters()))

    params = tuple(model.parameters())
    for i in range(len(H)):
        for j in range(len(H[i])):
            ni = params[i].nelement()
            nj = params[j].nelement()
            H[i][j] = H[i][j].reshape((ni, nj))
        H[i] = torch.cat(H[i], dim=1)
    H = torch.cat(H, dim=0)
    # add extra eps to the diagonal to make it invertible
    if args.l2_reg_coef < 1e-10:
        H += 1e-10 * torch.eye(H.shape[0], dtype=torch.float, device=H.device)
    print(f"Hessian shape: {H.shape}")
    H_inv = torch.inverse(H)

    # compute per example gradients (d loss / d weights for train and d pred / d weights for validation)
    train_grads = gradients.get_weight_gradients(model=model, dataset=train_data, cpu=False,
                                                 description='computing per example gradients on train data')

    jacobian_estimator = JacobianEstimator()
    val_grads = jacobian_estimator.compute_jacobian(model=model, dataset=val_data, cpu=False,
                                                    description='computing jacobian on validation data')

    # compute weight and prediction influences
    weight_vectors = []
    weight_quantities = []

    pred_vectors = []
    pred_quantities = []

    for sample_idx in tqdm(range(len(train_data)), desc='computing influences'):
        # compute for weights
        train_grad_flat = []
        for k, v in dict(model.named_parameters()).items():
            train_grad_flat.append(train_grads[k][sample_idx].flatten())
        train_grad_flat = torch.cat(train_grad_flat, dim=0)

        cur_weight_influence = 1.0 / len(train_data) * torch.mm(H_inv, train_grad_flat.view((-1, 1)))
        cur_weight_influence = cur_weight_influence.view((-1,))
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

    exp_dir = os.path.join(args.output_dir, 'influence-functions-brute-force', args.exp_name)
    process_results(vectors=weight_vectors, quantities=weight_quantities, meta=meta,
                    exp_name='weights', output_dir=exp_dir, train_data=train_data)

    # save preds
    meta = {
        'description': f'pred influence functions',
        'args': args
    }

    exp_dir = os.path.join(args.output_dir, 'influence-functions-brute-force', args.exp_name)
    process_results(vectors=pred_vectors, quantities=pred_quantities, meta=meta,
                    exp_name='pred', output_dir=exp_dir, train_data=train_data)


if __name__ == '__main__':
    main()
