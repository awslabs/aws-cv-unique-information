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

from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
import torch
import numpy as np

from nnlib.nnlib import utils


@utils.with_no_grad
def get_sgd_covariance_diagonal(model, dataset, cpu=True, max_num_examples=2**30, num_workers=0, seed=42, **kwargs):
    """ Returns the diagonal of the per-sample SGD noise covariance matrix.
    The formula is \Sigma = \frac{1}{n} \sum_{i=1}^n g_i g_i^T - \bar{g} \bar{g}^T, where g_i is the gradient
    corresponding to the ith example and \bar{g} is the total gradient. Note that we can ignore weight decay here,
    as adding weight decay doesn't change the SGD noise covariance matrix.
    """
    np.random.seed(seed)
    model.eval()

    if num_workers > 0:
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.multiprocessing.set_start_method('spawn', force=True)

    n_examples = min(len(dataset), max_num_examples)
    loader = DataLoader(dataset=Subset(dataset, range(n_examples)),
                        batch_size=1, shuffle=False, num_workers=num_workers)

    grad_sum = defaultdict(lambda: None)
    grad_squared_sum = defaultdict(lambda: None)

    # loop over the dataset
    for inputs_batch, labels_batch in tqdm(loader, desc='Computing sgd noise covariance...'):
        if isinstance(inputs_batch, torch.Tensor):
            inputs_batch = [inputs_batch]
        if not isinstance(labels_batch, list):
            labels_batch = [labels_batch]

        with torch.set_grad_enabled(True):
            outputs = model.forward(inputs=inputs_batch, labels=labels_batch, loader=loader, **kwargs)
            batch_losses, outputs = model.compute_loss(inputs=inputs_batch, labels=labels_batch, outputs=outputs,
                                                       loader=loader, dataset=loader.dataset)
            batch_total_loss = sum([loss for name, loss in batch_losses.items()])

        grad = torch.autograd.grad(batch_total_loss, model.parameters())
        if cpu:
            grad = [utils.to_cpu(v) for v in grad]

        for (k, _), v in zip(model.named_parameters(), grad):
            if grad_sum[k] is None:
                grad_sum[k] = v
            else:
                grad_sum[k] += v

            if grad_squared_sum[k] is None:
                grad_squared_sum[k] = v**2
            else:
                grad_squared_sum[k] += v**2

    out = dict()
    for k in grad_sum.keys():
        out[k] = grad_squared_sum[k] / n_examples - (grad_sum[k] / n_examples) ** 2

    return out


@utils.with_no_grad
def get_sgd_covariance_full(model, dataset, cpu=True, max_num_examples=2**30, num_workers=0, seed=42, **kwargs):
    """ Returns the per-sample SGD noise covariance matrix. Works when number of parameters is not large.
    The formula is \Sigma = \frac{1}{n} \sum_{i=1}^n g_i g_i^T - \bar{g} \bar{g}^T, where g_i is the gradient
    corresponding to the ith example and \bar{g} is the total gradient. Note that we can ignore weight decay here,
    as adding weight decay doesn't change the SGD noise covariance matrix.
    """
    np.random.seed(seed)
    model.eval()

    if num_workers > 0:
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.multiprocessing.set_start_method('spawn', force=True)

    n_examples = min(len(dataset), max_num_examples)
    loader = DataLoader(dataset=Subset(dataset, range(n_examples)),
                        batch_size=1, shuffle=False, num_workers=num_workers)

    n_params = utils.get_num_parameters(model)
    avg_grad = torch.zeros((n_params,), dtype=torch.float, device=('cpu' if cpu else model.device))
    sigma = torch.zeros((n_params, n_params), dtype=torch.float, device=('cpu' if cpu else model.device))

    # loop over the dataset
    for inputs_batch, labels_batch in tqdm(loader, desc='Computing sgd noise covariance...'):
        if isinstance(inputs_batch, torch.Tensor):
            inputs_batch = [inputs_batch]
        if not isinstance(labels_batch, list):
            labels_batch = [labels_batch]

        with torch.set_grad_enabled(True):
            outputs = model.forward(inputs=inputs_batch, labels=labels_batch, loader=loader, **kwargs)
            batch_losses, outputs = model.compute_loss(inputs=inputs_batch, labels=labels_batch, outputs=outputs,
                                                       loader=loader, dataset=loader.dataset)
            batch_total_loss = sum([loss for name, loss in batch_losses.items()])

        grad = torch.autograd.grad(batch_total_loss, model.parameters())
        if cpu:
            grad = [utils.to_cpu(v) for v in grad]

        grad_flat = [v.flatten() for v in grad]
        grad_flat = torch.cat(grad_flat, dim=0)

        avg_grad += 1.0 / n_examples * grad_flat
        sigma += 1.0 / n_examples * torch.mm(grad_flat.reshape((-1, 1)), grad_flat.reshape((1, -1)))

    sigma = sigma - torch.mm(avg_grad.reshape((-1, 1)), avg_grad.reshape((1, -1)))

    return sigma
