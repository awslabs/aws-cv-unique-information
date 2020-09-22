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

from torch.utils.data import TensorDataset
import torch
import numpy as np
from tqdm import tqdm

from nnlib.nnlib import utils
from nnlib.nnlib.matplotlib_utils import import_matplotlib
matplotlib, plt = import_matplotlib(agg=True, use_style=True)

from sample_info.modules.data_utils import get_synthetic_data
from sample_info.modules.ntk import JacobianEstimator, compute_ntk
from sample_info.modules.stability import weight_stability
from sample_info import methods


def plot(quantities, data_X, data_Y, half, t):
    q = utils.to_numpy(torch.stack(quantities).flatten())
    order = np.argsort(q)
    top_percent = 10
    top_cnt = int(top_percent / 100.0 * len(quantities))
    indices = order[-top_cnt:]

    fig, ax = plt.subplots()
    color_pallet = ['gray', 'red']
    colors = [color_pallet[1] if i in indices else color_pallet[0] for i in range(len(quantities))]
    colors = np.array(colors)
    markers_list = ['o', '*']
    for class_idx in range(2):
        mask = (data_Y[:half] == class_idx)
        ax.scatter(data_X[:half][mask][:, 0], data_X[:half][mask][:, 1],
                   alpha=0.6, s=10, c=colors[mask], marker=markers_list[class_idx])
    ax.set_title(f"top {top_percent}% most important samples at time {t}")

    return fig, ax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str,
                        default='sample_info/configs/1hidden-mlp-n1024-binary-mnist.json')
    parser.add_argument('--device', '-d', default='cuda', help='specifies the main device')
    parser.add_argument('--seed', type=int, default=42)

    # hyper-parameters
    parser.add_argument('--model_class', '-m', type=str, default='ClassifierL2')

    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    args = parser.parse_args()
    print(args)

    # Build data
    data_X, data_Y = get_synthetic_data(args.seed)
    half = len(data_X) // 2
    train_data = TensorDataset(torch.tensor(data_X[:half]).float(), torch.tensor(data_Y[:half]).long().reshape((-1, 1)))
    val_data = TensorDataset(torch.tensor(data_X[half:]).float(), torch.tensor(data_Y[half:]).long().reshape((-1, 1)))

    with open(args.config, 'r') as f:
        architecture_args = json.load(f)

    model_class = getattr(methods, args.model_class)

    model = model_class(input_shape=train_data[0][0].shape,
                        architecture_args=architecture_args,
                        device=args.device)

    jacobian_estimator = JacobianEstimator(projection='none')
    jacobians = jacobian_estimator.compute_jacobian(model=model, dataset=train_data, output_key='pred', cpu=False)
    # val_jacobians = get_jacobians(model=model, dataset=val_data, output_key='pred', cpu=False)
    init_preds = utils.apply_on_dataset(model=model, dataset=train_data, cpu=False)['pred']
    # val_init_preds = utils.apply_on_dataset(model=model, dataset=val_data, cpu=False)['pred']
    init_params = dict(model.named_parameters())
    ntk = compute_ntk(jacobians=jacobians)

    Y = [torch.tensor([y]) for (x, y) in train_data]
    Y = torch.stack(Y).float().to(ntk.device)

    ts = range(0, 1001, 20)
    for idx, t in tqdm(enumerate(ts), desc='main loop', total=len(ts)):
        _, q = weight_stability(t=t,
                                n=len(train_data),
                                eta=args.lr / len(train_data),
                                init_params=init_params,
                                jacobians=jacobians,
                                ntk=ntk,
                                init_preds=init_preds,
                                Y=Y,
                                continuous=False,
                                return_change_vectors=False,
                                scale_by_hessian=False)

        fig, ax = plot(q, data_X=data_X, data_Y=data_Y, half=half, t=t)
        file_path = f'sample_info/plots/synthetic-data/weight-{idx:04d}.png'
        utils.make_path(os.path.dirname(file_path))
        fig.savefig(file_path)
        plt.close()

    # save video
    cur_dir = os.path.abspath(os.curdir)
    os.chdir('sample_info/plots/synthetic-data')
    os.system("ffmpeg -r 2 -i weight-%04d.png movie.webm")
    os.chdir(cur_dir)


if __name__ == '__main__':
    main()
