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

import numpy as np
import torch

from nnlib.nnlib import utils
from nnlib.nnlib.data_utils.base import revert_normalization
from nnlib.nnlib.matplotlib_utils import import_matplotlib
from nnlib.nnlib.visualizations import get_image, savefig


def plot_histogram_of_informativeness(informativeness_scores, groups=None, bins=50, plt=None, save_name=None,
                                      density=False, **kwargs):
    """
    :param informativeness_scores: np.ndarray of informativeness scores
    :param groups: list of strings describing the group of each sample
    """
    if plt is None:
        _, plt = import_matplotlib(agg=True, use_style=False)

    with_groups = True
    if groups is None:
        groups = ['dummy'] * len(informativeness_scores)
        with_groups = False

    different_groups = sorted(list(set(groups)))
    groups = np.array(groups)

    fig, ax = plt.subplots(figsize=(7, 5))
    for g in different_groups:
        _, bins, _ = ax.hist(informativeness_scores[groups == g], bins=bins, alpha=0.5, label=g,
                             density=density)

    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.set_xlabel('Informativeness of an example')
    if density:
        ax.ticklabel_format(axis="y", style='sci', scilimits=(0, 0))
        ax.set_ylabel('Density')
    else:
        ax.set_ylabel('Count')
    if with_groups:
        ax.legend()

    fig.tight_layout()
    if save_name:
        savefig(fig, save_name)

    return fig, ax


def plot_summary_of_informativeness(train_data, informativeness_scores, label_names=None, save_name=None, plt=None,
                                    is_label_one_hot=False):
    """
    :param informativeness_scores: np.ndarray of informativeness scores
    """
    if plt is None:
        _, plt = import_matplotlib(agg=True, use_style=False)

    fig = plt.figure(constrained_layout=True, figsize=(24, 5))
    gs = fig.add_gridspec(22, 13)
    ax_left = fig.add_subplot(gs[1:22, :3])

    ys = [torch.tensor(y) for x, y in train_data]
    if is_label_one_hot:
        ys = [torch.argmax(y) for y in ys]
    ys = np.array([y.item() for y in ys])

    set_ys = sorted(list(set(ys)))

    for y in set_ys:
        mask = (ys == y)
        label = str(y)
        if label_names is not None:
            label = label_names[y]
        ax_left.hist(informativeness_scores[mask], bins=30, label=label, alpha=0.6)
    ax_left.legend()
    ax_left.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax_left.set_xlabel('Informativeness of an example')
    ax_left.set_ylabel('Count')
    ax_left.legend()
    x_pos = ax_left.get_position().x0
    y_pos = ax_left.get_position().y1
    fig.text(x_pos - 0.05, y_pos + 0.065, 'A', size=28, weight='bold')

    order = np.argsort(informativeness_scores)
    least_informative = order[:10]
    most_informative = order[-10:]

    for i in range(10):
        ax = fig.add_subplot(gs[1:11, 3 + i])
        if i == 0:
            x_pos = ax.get_position().x0
            y_pos = ax.get_position().y1
            fig.text(x_pos - 0.05, y_pos + 0.065, 'B', size=28, weight='bold')

        x, y = train_data[least_informative[i]]
        x = revert_normalization(x, train_data)[0]
        x = utils.to_numpy(x)
        x = get_image(x)
        ax.imshow(x, vmin=0, vmax=1)
        ax.set_axis_off()

    for i in range(10):
        ax = fig.add_subplot(gs[12:22, 3 + i])
        if i == 0:
            x_pos = ax.get_position().x0
            y_pos = ax.get_position().y1
            fig.text(x_pos - 0.05, y_pos + 0.042, 'C', size=28, weight='bold')

        x, y = train_data[most_informative[i]]
        x = revert_normalization(x, train_data)[0]
        x = utils.to_numpy(x)
        x = get_image(x)
        ax.imshow(x, vmin=0, vmax=1)
        ax.set_axis_off()

    if save_name is not None:
        savefig(fig, save_name)

    return fig, None
