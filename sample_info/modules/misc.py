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

import pickle
import os
import time

import numpy as np
import torch

from nnlib.nnlib import utils
from nnlib.nnlib.visualizations import plot_examples_from_dataset
from nnlib.nnlib.matplotlib_utils import import_matplotlib
from sample_info.modules.visualizations import plot_histogram_of_informativeness


def unique(arr):
    return sorted(list(set(arr)))


def invert_permutation(p):
    n = len(p)
    inv_p = [0] * n
    for i in range(n):
        inv_p[p[i]] = i
    return inv_p


def compute_order_similarity(order1, order2):
    order1 = list(order1)
    order2 = list(order2)
    pos2 = invert_permutation(order2)
    n = len(order1)
    agreement = 0
    for i in range(n):
        for j in range(i):
            index_i = pos2[order1[i]]
            index_j = pos2[order1[j]]
            if index_j < index_i:
                agreement += 1
    agreement /= (n * (n-1)) / 2
    return agreement


def make_importance_result_dict(importance_vectors, importance_measures, meta):
    """
    :param importance_vectors: per example list of [vectors | dictionary of vectors]
    :param importance_measures: per example list of [vector | dictionary of vectors]
    :param meta: dictionary of tags or other values
    :return: dictionary
    """
    ret = {}
    if importance_vectors is not None:
        ret['importance_vectors'] = importance_vectors
    if importance_measures is not None:
        ret['importance_measures'] = importance_measures
    ret['meta'] = meta
    return ret


def save_results_dict(results_dict, file_path):
    utils.make_path(os.path.dirname(file_path))
    with open(file_path, 'wb') as f:
        pickle.dump(results_dict, f)


@utils.with_no_grad
def process_results(vectors, quantities, meta, exp_name, output_dir, train_data, plt=None, **kwargs):
    if plt is None:
        matplotlib, plt = import_matplotlib(agg=True)
    np.random.seed(int(time.time()))
    random_string = str(np.random.randint(1000000))
    exp_name += '-h' + random_string

    results_dict = make_importance_result_dict(importance_vectors=vectors,
                                               importance_measures=quantities,
                                               meta=meta)

    # save the results
    if output_dir is not None:
        file_path = os.path.join(output_dir, exp_name, 'results.pkl')
        save_results_dict(results_dict=results_dict, file_path=file_path)

    # save the histogram image and extreme examples
    quantities = torch.stack(quantities).flatten()
    quantities = utils.to_numpy(quantities)

    if output_dir is None:
        save_name = None
    else:
        save_name = os.path.join(output_dir, exp_name, 'histogram.pdf')
    fig, ax = plot_histogram_of_informativeness(informativeness_scores=quantities, plt=plt,
                                                save_name=save_name, **kwargs)

    order = np.argsort(quantities)
    fig, ax = plot_examples_from_dataset(data=train_data, indices=order[:10], plt=plt, n_rows=1, **kwargs)
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, exp_name, 'least-important.pdf'))

    fig, ax = plot_examples_from_dataset(data=train_data, indices=order[-10:], plt=plt, n_rows=1, **kwargs)
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, exp_name, 'most-important.pdf'))


@utils.with_no_grad
def update_ntk_inv_slower(ntk, ntk_inv, keep_indices):
    """ Slightly slower but more general implementation. """
    S = set(keep_indices)
    remove_indices = [i for i in range(ntk.shape[0]) if i not in S]

    A12 = ntk[keep_indices][:, remove_indices]
    A22 = ntk[remove_indices][:, remove_indices]
    A21 = ntk[remove_indices][:, keep_indices]

    F11_inv = ntk_inv[keep_indices][:, keep_indices]
    mid_inv = torch.inverse(A22 + torch.mm(torch.mm(A21, F11_inv), A12))
    A11_inv = F11_inv - torch.mm(torch.mm(torch.mm(F11_inv, A12), mid_inv), torch.mm(A21, F11_inv))

    return A11_inv


@utils.with_no_grad
def update_ntk_inv(ntk, ntk_inv, keep_indices):
    """ A little bit faster implementation. Uses the fact that ntk is a symmetric matrix. """
    S = set(keep_indices)
    remove_indices = [i for i in range(ntk.shape[0]) if i not in S]

    A12 = ntk[keep_indices][:, remove_indices]
    A22 = ntk[remove_indices][:, remove_indices]
    F11_inv = ntk_inv[keep_indices][:, keep_indices]
    C = torch.mm(F11_inv, A12)
    mid_inv = torch.inverse(A22 + torch.mm(C.T, A12))
    A11_inv = F11_inv - torch.mm(torch.mm(C, mid_inv), C.T)

    return A11_inv
