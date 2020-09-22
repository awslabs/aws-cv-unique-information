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
import scipy
import numpy as np

from nnlib.nnlib import utils


class JacobianEstimator:
    def __init__(self, projection='none', random_subset_n_select=2000, very_sparse_count=10000):
        if projection not in ['none', 'random-subset', 'very-sparse']:
            raise ValueError(f"Projection '{projection}' is not implemented.")
        self.projection = projection
        self.random_subset_n_select = random_subset_n_select
        self.very_sparse_count = very_sparse_count
        self._random_subset_proj_indices = None  # used when projection == 'random-subset'
        self._very_sparse_proj_matrix = None  # used when projection == 'very-sparse'

    def _prepare_random_subset_proj_indices(self, named_params):
        if self._random_subset_proj_indices is not None:
            return
        # build the projection if it is the first time
        self._random_subset_proj_indices = dict()
        total_selected = 0
        for k, v in dict(named_params).items():
            v = v.flatten()
            n_select = min(v.shape[0], self.random_subset_n_select)
            total_selected += n_select
            self._random_subset_proj_indices[k] = np.random.choice(v.shape[0], n_select, replace=False)
        print(f"JacobianEstimator: projection dimensionality = {total_selected}")

    def _prepare_very_sparse_proj_matrix(self, named_params):
        if self._very_sparse_proj_matrix is not None:
            return
        # build the projection matrix if it is the first time
        self._very_sparse_proj_matrix = dict()
        count = self.very_sparse_count
        print(f"JacobianEstimator: projection dimensionality = {count}")

        for k, v in dict(named_params).items():
            v = v.flatten()

            print(f"Building the projection matrix for {k}...", end='')
            if v.shape[0] <= 1000:
                n_select = v.shape[0]
            else:
                n_select = int(np.sqrt(v.shape[0]))
            select_prob = n_select / v.shape[0]
            sparse_indices = []
            sparse_values = []
            for idx in range(count):
                select_indices = np.zeros((n_select, 2), dtype=np.int)
                select_indices[:, 0] = np.random.choice(v.shape[0], n_select)
                select_indices[:, 1] = idx
                values = 2 * np.random.randint(2, size=(n_select,)) - 1.0
                sparse_indices.append(select_indices)
                sparse_values.append(values)

            sparse_indices = np.concatenate(sparse_indices, axis=0)
            sparse_values = np.concatenate(sparse_values, axis=0)
            sparse_values *= np.sqrt(1.0 / select_prob) / np.sqrt(count)

            self._very_sparse_proj_matrix[k] = scipy.sparse.coo_matrix((sparse_values, sparse_indices.T),
                                                                       shape=(v.shape[0], count))
            print("\tDone")

    @utils.with_no_grad
    def compute_jacobian(self, model, dataset, cpu=True, description="", output_key='pred',
                         max_num_examples=2 ** 30, num_workers=0, seed=42, **kwargs):
        np.random.seed(seed)
        model.eval()

        if num_workers > 0:
            torch.multiprocessing.set_sharing_strategy('file_system')
            torch.multiprocessing.set_start_method('spawn', force=True)

        n_examples = min(len(dataset), max_num_examples)
        loader = DataLoader(dataset=Subset(dataset, range(n_examples)),
                            batch_size=1, shuffle=False, num_workers=num_workers)

        jacobians = defaultdict(list)

        # loop over the dataset
        n_outputs = None
        for inputs_batch, labels_batch in tqdm(loader, desc=description):
            if isinstance(inputs_batch, torch.Tensor):
                inputs_batch = [inputs_batch]
            if not isinstance(labels_batch, list):
                labels_batch = [labels_batch]

            with torch.set_grad_enabled(True):
                outputs = model.forward(inputs=inputs_batch, labels=labels_batch, loader=loader, **kwargs)
                preds = outputs[output_key][0]
            n_outputs = preds.shape[-1]

            for output_idx in range(n_outputs):
                retain_graph = (output_idx != n_outputs - 1)
                with torch.set_grad_enabled(True):
                    cur_jacobians = torch.autograd.grad(preds[output_idx], model.parameters(),
                                                        retain_graph=retain_graph)
                if cpu:
                    cur_jacobians = [utils.to_cpu(v) for v in cur_jacobians]

                if self.projection == 'none':
                    for (k, _), v in zip(model.named_parameters(), cur_jacobians):
                        jacobians[k].append(v)

                if self.projection == 'random-subset':
                    self._prepare_random_subset_proj_indices(model.named_parameters())

                    for (k, _), v in zip(model.named_parameters(), cur_jacobians):
                        v = v.flatten()
                        n_select = len(self._random_subset_proj_indices[k])
                        v_proj = v[self._random_subset_proj_indices[k]] * np.sqrt(v.shape[0] / n_select)
                        jacobians[k].append(v_proj)

                if self.projection == 'very-sparse':
                    self._prepare_very_sparse_proj_matrix(model.named_parameters())
                    for (k, _), v in zip(model.named_parameters(), cur_jacobians):
                        # now that the projection matrix is ready, we can project v into the smaller subspace
                        v = v.flatten()
                        v_proj = self._very_sparse_proj_matrix[k].T.dot(utils.to_numpy(v))
                        v_proj = torch.tensor(v_proj, dtype=v.dtype, device=v.device)
                        jacobians[k].append(v_proj)

        for k in jacobians:
            jacobians[k] = torch.stack(jacobians[k])  # n_samples * n_outputs x n_params
            assert len(jacobians[k]) == n_outputs * n_examples

        return jacobians


@utils.with_no_grad
def compute_test_train_ntk(train_jacobians, test_jacobians):
    """
    :param train_jacobians: dictionary of train gradients of parameters, each of
        shape (n_train_samples * n_outputs, n_dimensions).
    :param test_jacobians: dictionary of test gradients of parameters, each of
        shape (n_test_samples * n_outputs, n_dimensions).
    :return: the NTK matrix of shape (n_test_samples * n_outputs, n_train_samples * n_outputs)
    """
    ntk = None
    for k, v_train in train_jacobians.items():
        v_test = test_jacobians[k]
        n_train = v_train.shape[0]
        n_test = v_test.shape[0]
        v_train_flat = v_train.reshape((n_train, -1))
        v_test_flat = v_test.reshape((n_test, -1))
        if ntk is None:
            ntk = torch.zeros((n_test, n_train), dtype=torch.float, device=v_train.device)
        ntk += torch.mm(v_test_flat, v_train_flat.T)
    return ntk


@utils.with_no_grad
def compute_ntk(jacobians):
    """
    :param jacobians: dictionary of gradients of parameters, each of shape (n_samples * n_outputs, n_dimensions).
    :return the NTK matrix of shape (n_samples * n_outputs, n_samples * n_outputs)
    """
    return compute_test_train_ntk(jacobians, jacobians)


@utils.with_no_grad
def compute_exp_matrix(t, eta, ntk, continuous):
    """ computes exp{-t eta ntk} or its discrete counterpart"
    :param t: if time is set to None, then t=infinity (exp_matrix = 0) will be returned (assuming ntk is invertible).
    """
    if t is None:
        return torch.zeros_like(ntk)
    n = ntk.shape[0]
    if continuous:
        exp_matrix = scipy.linalg.expm(-eta * t * utils.to_numpy(ntk))
        exp_matrix = torch.tensor(exp_matrix, device=ntk.device, dtype=torch.float)
    else:
        identity_matrix = torch.eye(n, dtype=torch.float, device=ntk.device)
        exp_matrix = torch.matrix_power(identity_matrix - eta * ntk, t)
    return exp_matrix


@utils.with_no_grad
def get_predictions_at_time_t(t, eta, ntk, init_preds, Y, continuous=False):
    """
    :param t: time/iteration.
    :param eta: NOTE that if network's loss is a mean over examples (instead of sum) then eta=learning_rate / n_samples.
    :param ntk: the neural tangent kernel.
    :param init_preds: predictions at initialization  # (n_samples, n_outputs)
    :param Y: labels  # (n_samples, n_outputs)
    :param continuous: True => continuous GD, False => discrete GD
    :return predictions  # (n_samples, n_outputs)
    """
    n_outputs = init_preds.shape[-1]
    init_preds = init_preds.reshape((-1, 1))  # (n_samples * n_outputs, 1)
    Y = Y.reshape((-1, 1))  # (n_samples * n_outputs, 1)
    n = ntk.shape[0]
    exp_matrix = compute_exp_matrix(t=t, eta=eta, ntk=ntk, continuous=continuous)
    identity_matrix = torch.eye(n, dtype=torch.float, device=ntk.device)
    pred = torch.mm(identity_matrix - exp_matrix, Y) + torch.mm(exp_matrix, init_preds)
    return pred.reshape((-1, n_outputs))


@utils.with_no_grad
def get_test_predictions_at_time_t(t, eta, train_init_preds, test_init_preds, train_Y,
                                   ntk, test_train_ntk, ntk_inv=None, continuous=False):
    if ntk_inv is None:
        ntk_inv = torch.inverse(ntk)
    exp_matrix = compute_exp_matrix(t=t, eta=eta, ntk=ntk, continuous=continuous)

    n_outputs = train_init_preds.shape[-1]
    train_init_preds = train_init_preds.reshape((-1, 1))  # (n_samples * n_outputs, 1)
    test_init_preds = test_init_preds.reshape((-1, 1))  # (n_samples * n_outputs, 1)
    train_Y = train_Y.reshape((-1, 1))  # (n_samples * n_outputs, 1)

    identity_matrix = torch.eye(ntk.shape[0], dtype=torch.float, device=ntk.device)

    test_preds = test_init_preds - torch.mm(test_train_ntk, torch.mm(ntk_inv,
                                                                     torch.mm(identity_matrix - exp_matrix,
                                                                              train_init_preds - train_Y)))
    return test_preds.reshape((-1, n_outputs))


@utils.with_no_grad
def compute_training_loss_at_time_t(t, eta, ntk, init_preds, Y, continuous=False):
    pred_t = get_predictions_at_time_t(t=t, eta=eta, ntk=ntk, init_preds=init_preds,
                                       Y=Y, continuous=continuous)
    n_outputs = pred_t.shape[-1]
    Y = Y.reshape((-1, n_outputs))
    loss_t = 0.5 * torch.sum((pred_t - Y) ** 2, dim=1).mean(dim=0)
    return loss_t


@utils.with_no_grad
def get_weights_at_time_t(t, eta, init_params, ntk, init_preds, Y, jacobians=None, continuous=False,
                          ntk_inv=None, large_model_regime=False, model=None, dataset=None, batch_size=256):
    """ Computes the at time t. Since storing the full Jacobian can be impossible for large networks, the code
    can work in two regimes (given by large_model_regime):
        1. [small model regime] full Jacobian should be passed using the `jacobians` argument
        2. [large model regime] `model`, `dataset`, ['example_indices'] should be specified, so we can go
        over and compute all Jacobians one by one.

    :param t: time/iteration. If t=None, then final weights are going to be returned (assuming ntk is invertible).
    :param eta: the learning rate. NOTE: if network's loss is a mean over examples (instead of sum) then
                eta=learning_rate / n_samples.
    :param init_params: a dictionary containing parameters at initialziation.
    :param jacobians: dictionary of Jacobians at initialization.
    :param ntk: the neural tangent kernel.
    :param init_preds: predictions at initialization  # (n_samples, n_outputs)
    :param Y: labels  # (n_samples, n_outputs)
    :param continuous: True => continuous GD, False => discrete GD
    :param ntk_inv: Inverse of the ntk matrix. This argument is optional.
    :param large_model_regime: which regime is it

    # the following parameters are used in the large data regime only.
    :param model: the model.
    :param dataset: the training dataset for which ntk was computed for.
    :param batch_size: the batch_size argument to pass to the DataLoader.
    """

    # check that the needed arguments are not None
    if large_model_regime:
        assert (model is not None) and (dataset is not None)
    else:
        assert (jacobians is not None)

    exp_matrix = compute_exp_matrix(t=t, eta=eta, ntk=ntk, continuous=continuous)

    # compute the ntk inverse and multiply it with f(X) - Y
    if ntk_inv is None:
        ntk_inv = torch.inverse(ntk)
    init_preds = init_preds.reshape((-1, 1))  # (n_samples * n_outputs, 1)
    Y = Y.reshape((-1, 1))  # (n_samples * n_outputs, 1)
    identity_matrix = torch.eye(ntk.shape[0], dtype=torch.float, device=ntk.device)
    rhs_vector = -torch.mm(ntk_inv, torch.mm(identity_matrix - exp_matrix, init_preds - Y))

    # Now we need to compute jacobians * rhs_vector. This corresponds to the sum of all output gradients weighted
    # with rhs_vector coefficients. Therefore, we can go over (examples, output) pairs and sum their gradients.
    out = defaultdict(lambda: None)

    # is the Jacobian is already computed we can just use it
    if not large_model_regime:
        for k, v in jacobians.items():
            v = v.reshape((v.shape[0], -1))  # (n_samples * n_outputs, n_dim)
            out[k] = torch.mm(v.T, rhs_vector)[:, 0]   # (n_dim,)
    else:  # we need to go over examples and compute the Jacobian
        model.eval()
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

        # We want to compute \frac{\partial preds}{\partial W} and take a linear combination of its columns
        # with coefficients given with rhs_vector. To not go over all examples/outputs one by one, we can compute
        # compute #\frac{\partial <preds, coefficients>}{\partial W}.

        # loop over the dataset
        example_output_index = 0
        for inputs_batch, labels_batch in tqdm(loader):
            if isinstance(inputs_batch, torch.Tensor):
                inputs_batch = [inputs_batch]
            if not isinstance(labels_batch, list):
                labels_batch = [labels_batch]

            with torch.set_grad_enabled(True):
                outputs = model.forward(inputs=inputs_batch, labels=labels_batch, loader=loader)
                preds = outputs['pred']
                preds = preds.reshape((-1, 1))  # (n_examples * n_outputs, 1)
                coefficients = rhs_vector[example_output_index:example_output_index + preds.shape[0]].to(preds.device)
                loss = torch.sum(preds * coefficients)

            cur_jacobians = torch.autograd.grad(loss, model.parameters())
            for (k, _), v in zip(model.named_parameters(), cur_jacobians):
                v = v.detach().to(ntk.device).flatten()
                if out[k] is None:
                    out[k] = v
                else:
                    out[k] += v

            example_output_index += preds.shape[0]

        assert example_output_index == rhs_vector.shape[0]

    # add the initialized values
    for k, v in init_params.items():
        out[k] += v.flatten()
        out[k] = out[k].reshape(v.shape)

    return out


def prepare_needed_items(model, train_data, test_data=None, projection='none', cpu=False, **kwargs):
    jacobian_estimator = JacobianEstimator(projection=projection, **kwargs)
    train_jacobians = jacobian_estimator.compute_jacobian(model=model, dataset=train_data,
                                                          output_key='pred', cpu=cpu)
    test_jacobians = None
    if test_data is not None:
        test_jacobians = jacobian_estimator.compute_jacobian(model=model, dataset=test_data,
                                                             output_key='pred', cpu=cpu)

    train_init_preds = utils.apply_on_dataset(model=model, dataset=train_data, cpu=cpu)['pred']
    test_init_preds = None
    if test_data is not None:
        test_init_preds = utils.apply_on_dataset(model=model, dataset=test_data, cpu=cpu)['pred']

    init_params = dict(model.named_parameters())
    if cpu:
        for k, v in init_params.items():
            init_params[k] = v.to('cpu')

    ntk = compute_ntk(jacobians=train_jacobians)
    test_train_ntk = None
    if test_data is not None:
        test_train_ntk = compute_test_train_ntk(train_jacobians=train_jacobians,
                                                test_jacobians=test_jacobians)

    def extract_labels(data):
        ys = [utils.to_tensor(y, device=ntk.device).view((-1,)) for x, y in data]
        return torch.stack(ys).float()

    train_Y = extract_labels(train_data)
    test_Y = None
    if test_data is not None:
        test_Y = extract_labels(test_data)

    return {
        'jacobian_estimator': jacobian_estimator,
        'train_jacobians': train_jacobians,
        'test_jacobians': test_jacobians,
        'train_init_preds': train_init_preds,
        'test_init_preds': test_init_preds,
        'init_params': init_params,
        'ntk': ntk,
        'test_train_ntk': test_train_ntk,
        'train_Y': train_Y,
        'test_Y': test_Y
    }
