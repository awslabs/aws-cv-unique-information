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
import re

import numpy as np

from nnlib.nnlib.visualizations import savefig
from nnlib.nnlib.matplotlib_utils import import_matplotlib
matplotlib, plt = import_matplotlib(agg=True, use_style=True)


def read_random(results, args):
    print("Reading the 'random' baseline")
    exp_dir = os.path.join(args.root_dir, 'random', args.exp_name)
    assert os.path.exists(exp_dir)

    cur_results = defaultdict(list)

    files = os.listdir(exp_dir)
    for file_name in files:
        regex = re.compile("results-(\d*\.\d+|\d+)-(\d+).pkl")
        ret = regex.match(file_name)
        if ret is None:
            print(f"Skipping {file_name}")
            continue
        exclude_ratio = ret.group(1)

        with open(os.path.join(exp_dir, file_name), 'rb') as f:
            acc = pickle.load(f)['val_acc']
        cur_results[exclude_ratio].append(acc)

    cur_results = list(cur_results.items())
    cur_results = [(float(x), y) for x, y in cur_results]
    cur_results = sorted(cur_results)
    results['random']['xs'] = [x for x, y in cur_results]
    results['random']['ys'] = [y for x, y in cur_results]


def read_proposed(results, args, baseline_name):
    print(f"Reading the '{baseline_name}' baseline")
    exp_dir = os.path.join(args.root_dir, baseline_name, args.exp_name)
    assert os.path.exists(exp_dir)

    cur_results = []
    files = os.listdir(exp_dir)
    for file_name in files:
        regex = re.compile("results-(\d*\.\d+|\d+).pkl")
        ret = regex.match(file_name)
        if ret is None:
            print(f"Skipping {file_name}")
            continue
        exclude_ratio = float(ret.group(1))

        with open(os.path.join(exp_dir, file_name), 'rb') as f:
            acc = pickle.load(f)['val_acc']

        cur_results.append((exclude_ratio, acc))

    cur_results = sorted(cur_results)
    results[baseline_name]['xs'] = [x for x, y in cur_results]
    results[baseline_name]['ys'] = [y for x, y in cur_results]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-E', type=str, required=True)
    parser.add_argument('--root_dir', '-r', type=str,
                        default='sample_info/results/data-summarization/')
    parser.add_argument('--baselines', '-b', type=str, nargs='+', required=True)
    parser.add_argument('--num_examples', '-n', type=int, required=True)
    args = parser.parse_args()
    print(args)

    # storage for all methods
    results = defaultdict(dict)

    # random baseline
    if 'random' in args.baselines:
        read_random(results, args)

    # predictions top
    if 'predictions-top' in args.baselines:
        read_proposed(results, args, 'predictions-top')

    # predictions bottom
    if 'predictions-bottom' in args.baselines:
        read_proposed(results, args, 'predictions-bottom')

    # weights top
    if 'weights-plain-top' in args.baselines:
        read_proposed(results, args, 'weights-plain-top')

    # weights bottom
    if 'weights-plain-bottom' in args.baselines:
        read_proposed(results, args, 'weights-plain-bottom')

    # predictions iterative
    if 'predictions-iterative' in args.baselines:
        read_proposed(results, args, 'predictions-iterative')

    # weights iterative
    if 'weights-plain-iterative' in args.baselines:
        read_proposed(results, args, 'weights-plain-iterative')

    # plot
    fig, ax = plt.subplots(figsize=(7, 5))

    if 'random' in results:
        cur_results = results.pop('random')
        means = np.array([np.mean(y) for y in cur_results['ys']])
        stds = np.array([np.std(y) for y in cur_results['ys']])
        ax.plot(cur_results['xs'], means, label='random')
        ax.fill_between(cur_results['xs'], means - stds, means + stds, alpha=0.2)

    rename_dict = {
        'predictions-iterative': 'bottom (iterative)',
        'predictions-top': 'top',
        'predictions-bottom': 'bottom'
    }

    for baseline_name, cur_results in results.items():
        ax.plot(cur_results['xs'], cur_results['ys'], label=rename_dict.get(baseline_name, baseline_name))

    ax.set_xlabel('Ratio of removed examples')
    ax.set_ylabel('Test accuracy')
    ax.legend()
    fig.tight_layout()
    save_path = os.path.join(args.root_dir, 'aggregated', args.exp_name, 'plot.pdf')
    savefig(fig, save_path)


if __name__ == '__main__':
    main()
