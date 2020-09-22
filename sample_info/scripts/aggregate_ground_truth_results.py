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

from collections import defaultdict
import argparse
import os
import pickle

from tqdm import tqdm
import numpy as np
import torch

from nnlib.nnlib import utils
from nnlib.nnlib.visualizations import savefig
from nnlib.nnlib.matplotlib_utils import import_matplotlib
matplotlib, plt = import_matplotlib(agg=True, use_style=True)


def flatten_weight_diff(weight_diff):
    ret = []
    for example_weight_diff in weight_diff:
        flat = []
        for k, v in example_weight_diff.items():
            flat.append(v.flatten())
        flat = torch.cat(flat, dim=0)
        ret.append(flat)
    return ret


def read_ground_truth(args, results):
    exp_dir = os.path.join(args.root_dir, 'ground-truth', args.exp_name)
    assert os.path.exists(exp_dir)

    file_path = os.path.join(exp_dir, 'full-data-training.pkl')
    with open(file_path, 'rb') as f:
        full_data_training = pickle.load(f)

    print(f"Reading ground truth, found {len(os.listdir(exp_dir)) - 1} / {args.num_examples} examples")

    results['ground-truth']['weights_diff'] = [None] * args.num_examples
    results['ground-truth']['pred_diff'] = [None] * args.num_examples
    mask = np.zeros(args.num_examples, dtype=np.bool)
    for file_name in tqdm(os.listdir(exp_dir), desc='Reading ground truth'):
        if file_name == 'full-data-training.pkl':
            continue

        sample_idx = int(file_name[:-4])
        file_path = os.path.join(exp_dir, file_name)
        with open(file_path, 'rb') as f:
            cur_data = pickle.load(f)

        weights_diff_flat = []
        for k in cur_data['weights'].keys():
            weights_diff_flat.append((cur_data['weights'][k] - full_data_training['weights'][k]).flatten())
        weights_diff_flat = torch.cat(weights_diff_flat, dim=0)

        mask[sample_idx] = True
        results['ground-truth']['weights_diff'][sample_idx] = weights_diff_flat
        results['ground-truth']['pred_diff'][sample_idx] = cur_data['val_preds'] - full_data_training['val_preds']

    return mask


def read_proposed(args, results):
    print("Reading proposed")
    exp_dir = os.path.join(args.root_dir, 'informativeness', args.exp_name)
    assert os.path.exists(exp_dir)

    dirs = os.listdir(exp_dir)
    assert len(dirs) == 2

    weights_dir = list(filter(lambda x: x.find('weight') != -1, dirs))
    assert len(weights_dir) == 1
    weights_dir = weights_dir[0]
    file_path = os.path.join(exp_dir, weights_dir, 'results.pkl')
    with open(file_path, 'rb') as f:
        proposed_saved = pickle.load(f)
        results['proposed']['weights_diff'] = flatten_weight_diff(proposed_saved['importance_vectors'])

    pred_dir = list(filter(lambda x: x.find('pred') != -1, dirs))
    assert len(pred_dir) == 1
    pred_dir = pred_dir[0]
    file_path = os.path.join(exp_dir, pred_dir, 'results.pkl')
    with open(file_path, 'rb') as f:
        proposed_saved = pickle.load(f)
        results['proposed']['pred_diff'] = proposed_saved['importance_vectors']

    return True


def read_influence_functions(args, results):
    exp_dir = os.path.join(args.root_dir, 'influence-functions', args.exp_name)
    if not os.path.exists(exp_dir):
        return False

    print("Reading influence functions")
    dirs = os.listdir(exp_dir)
    assert len(dirs) == 2

    weights_dir = list(filter(lambda x: x.find('weight') != -1, dirs))
    assert len(weights_dir) == 1
    weights_dir = weights_dir[0]
    file_path = os.path.join(exp_dir, weights_dir, 'results.pkl')
    with open(file_path, 'rb') as f:
        proposed_saved = pickle.load(f)
        results['influence-functions']['weights_diff'] = proposed_saved['importance_vectors']

    pred_dir = list(filter(lambda x: x.find('pred') != -1, dirs))
    assert len(pred_dir) == 1
    pred_dir = pred_dir[0]
    file_path = os.path.join(exp_dir, pred_dir, 'results.pkl')
    with open(file_path, 'rb') as f:
        proposed_saved = pickle.load(f)
        results['influence-functions']['pred_diff'] = proposed_saved['importance_vectors']

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-E', type=str, required=True)
    parser.add_argument('--root_dir', '-r', type=str,
                        default='sample_info/results/ground-truth/')
    parser.add_argument('--num_examples', '-n', type=int, required=True)
    args = parser.parse_args()
    print(args)

    # storage for all methods
    results = defaultdict(lambda: defaultdict(list))

    # read ground truths
    mask = read_ground_truth(args, results)

    # read proposed
    read_proposed(args, results)

    # read influence functions
    influence_functions = read_influence_functions(args, results)

    # plot
    keys = ['weights_diff', 'pred_diff']
    for key in keys:
        norms = dict()

        for method, D in results.items():
            vectors = D[key]
            cur_norms = [torch.sum(x**2) for idx, x in enumerate(vectors) if mask[idx]]
            cur_norms = torch.stack(cur_norms).flatten()
            norms[method] = utils.to_numpy(cur_norms)

        fig, ax = plt.subplots()
        vmin = np.min(norms['ground-truth'])
        vmax = np.max(norms['ground-truth'])
        ax.set_title(key)
        ax.scatter(norms['ground-truth'], norms['proposed'], label='gt-vs-proposed', s=5)
        ax.set_xlim(left=vmin, right=vmax)
        ax.set_ylim(bottom=vmin, top=vmax)
        # ax.scatter(norms['ground-truth'], norms['influence-functions'], label='gt-vs-influence', s=5)
        ax.set_xlabel('ground_truth')
        ax.legend()
        fig.tight_layout()
        save_path = os.path.join(args.root_dir, 'aggregated', args.exp_name, f'{key}-norm-scatter.pdf')
        savefig(fig, save_path)

        print("Correlations of proposed:")
        print(np.corrcoef(norms['ground-truth'], norms['proposed']))

        if influence_functions:
            print("Correlations of influence functions:")
            print(np.corrcoef(norms['ground-truth'], norms['influence-functions']))


if __name__ == '__main__':
    main()
