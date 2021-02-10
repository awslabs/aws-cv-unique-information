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

"""
TODO: write description and notes.

Remark1: eta here corresponds to the eta in math. Note that in math total loss = sum_i Loss_i + lamb * ||w - w_0||. In
the codes however, the loss = 1/n * sum_i Loss_i + lamb' * ||w - w_0||.
Therefore, we want lamb' = lamb * n and eta' (in math) = 1/n * eta.
"""
from scipy.linalg import solve_continuous_lyapunov
from tqdm import tqdm
from torch.utils.data import Subset
import torch

from nnlib.nnlib import utils
from .ntk import compute_training_loss_at_time_t, get_predictions_at_time_t, get_weights_at_time_t, \
    get_test_predictions_at_time_t
from . import misc
from .sgd import get_sgd_covariance_full


@utils.with_no_grad
def training_loss_stability(ts, n, eta, ntk, init_preds, Y, l2_reg_coef=0.0, continuous=False):
    if l2_reg_coef > 0:
        ntk = ntk + l2_reg_coef * torch.eye(ntk.shape[0], dtype=torch.float, device=ntk.device)

    losses_without_excluding = [compute_training_loss_at_time_t(t=t,
                                                                eta=eta,
                                                                ntk=ntk,
                                                                init_preds=init_preds,
                                                                Y=Y,
                                                                continuous=continuous) for t in ts]
    losses_without_excluding = torch.stack(losses_without_excluding)

    n_outputs = init_preds.shape[-1]
    change_quantities = []
    change_vectors = []
    for sample_idx in tqdm(range(n)):
        example_indices = [i for i in range(n) if i != sample_idx]
        example_output_indices = []
        for i in example_indices:
            example_output_indices.extend(range(i * n_outputs, (i + 1) * n_outputs))

        new_ntk = ntk.clone()[example_output_indices]
        new_ntk = new_ntk[:, example_output_indices]
        new_init_preds = init_preds[example_indices]
        new_Y = Y[example_indices]

        losses = [compute_training_loss_at_time_t(t=t,
                                                  eta=eta * n / (n - 1),
                                                  ntk=new_ntk,
                                                  init_preds=new_init_preds,
                                                  Y=new_Y,
                                                  continuous=continuous) for t in ts]
        losses = torch.stack(losses)

        change_vectors.append(losses - losses_without_excluding)
        change_quantities.append(torch.mean((losses - losses_without_excluding) ** 2))

    return change_vectors, change_quantities


@utils.with_no_grad
def training_pred_stability(t, n, eta, ntk, init_preds, Y, l2_reg_coef=0.0, continuous=False):
    if l2_reg_coef > 0:
        ntk = ntk + l2_reg_coef * torch.eye(ntk.shape[0], dtype=torch.float, device=ntk.device)

    old_preds = get_predictions_at_time_t(t=t, eta=eta, ntk=ntk, init_preds=init_preds,
                                          Y=Y, continuous=continuous)

    n_outputs = init_preds.shape[-1]
    change_vectors = []
    change_quantities = []
    for sample_idx in tqdm(range(n)):
        example_indices = [i for i in range(n) if i != sample_idx]
        example_output_indices = []
        for i in example_indices:
            example_output_indices.extend(range(i * n_outputs, (i + 1) * n_outputs))

        new_ntk = ntk.clone()[example_output_indices]
        new_ntk = new_ntk[:, example_output_indices]
        new_init_preds = init_preds[example_indices]
        new_Y = Y[example_indices]

        new_preds = get_predictions_at_time_t(t=t,
                                              eta=eta * n / (n - 1),
                                              ntk=new_ntk,
                                              init_preds=new_init_preds,
                                              Y=new_Y,
                                              continuous=continuous)

        change_quantities.append(torch.sum((old_preds[example_indices] - new_preds) ** 2, dim=1).mean(dim=0))
        change_vectors.append(new_preds - old_preds[example_indices])

    return change_vectors, change_quantities


@utils.with_no_grad
def weight_stability(t, n, eta, init_params, jacobians, ntk, init_preds, Y, continuous=False, without_sgd=True,
                     l2_reg_coef=0.0, large_model_regime=False, model=None, dataset=None, return_change_vectors=True,
                     **kwargs):
    """
    # TODO: fill the description of arguments
    :param without_sgd: if without_sgd = True, then only ||w1-w2|| will be returned,
                        otherwise (w1-w2)^T H Sigma^{-1} (w1-w2).
    """
    if l2_reg_coef > 0:
        ntk = ntk + l2_reg_coef * torch.eye(ntk.shape[0], dtype=torch.float, device=ntk.device)

    ntk_inv = torch.inverse(ntk)
    old_weights = get_weights_at_time_t(t=t, eta=eta, init_params=init_params, jacobians=jacobians,
                                        ntk=ntk, ntk_inv=ntk_inv, init_preds=init_preds, Y=Y, continuous=continuous,
                                        large_model_regime=large_model_regime, model=model, dataset=dataset, **kwargs)

    steady_state_inv_cov = None
    if not without_sgd:
        if large_model_regime:
            raise ValueError("SGD formula works only for small models")

        # compute the SGD noise covariance matrix at the end
        assert (model is not None) and (dataset is not None)
        with utils.SetTemporaryParams(model=model, params=old_weights):
            sgd_cov = get_sgd_covariance_full(model=model, dataset=dataset, cpu=False, **kwargs)
            # add small amount of isotropic Gaussian noise to make sgd_cov invertible
            sgd_cov += 1e-10 * torch.eye(sgd_cov.shape[0], device=sgd_cov.device, dtype=torch.float)

        # now we compute H Sigma^{-1}
        jacobians_cat = [v.view((v.shape[0], -1)) for k, v in jacobians.items()]
        jacobians_cat = torch.cat(jacobians_cat, dim=1)  # (n_samples * n_outputs, n_params)
        H = torch.mm(jacobians_cat.T, jacobians_cat) + l2_reg_coef * torch.eye(jacobians_cat.shape[1],
                                                                               device=ntk.device, dtype=torch.float)
        # steady_state_inv_cov = torch.mm(H, torch.inverse(sgd_cov))
        with utils.Timing(description="Solving the Lyapunov equation"):
            steady_state_cov = solve_continuous_lyapunov(a=utils.to_numpy(H), q=utils.to_numpy(sgd_cov))
        steady_state_cov = torch.tensor(steady_state_cov, dtype=torch.float, device=ntk.device)
        # add small amount of isotropic Gaussian noise to make steady_state_cov invertible
        steady_state_cov += 1e-10 * torch.eye(steady_state_cov.shape[0], device=steady_state_cov.device,
                                              dtype=torch.float)
        steady_state_inv_cov = torch.inverse(steady_state_cov)

    change_vectors = []
    change_quantities = []
    n_outputs = init_preds.shape[-1]
    for sample_idx in tqdm(range(n)):
        example_indices = [i for i in range(n) if i != sample_idx]
        example_output_indices = []
        for i in example_indices:
            example_output_indices.extend(range(i * n_outputs, (i + 1) * n_outputs))

        new_ntk = ntk.clone()[example_output_indices]
        new_ntk = new_ntk[:, example_output_indices]

        new_ntk_inv = misc.update_ntk_inv(ntk=ntk, ntk_inv=ntk_inv, keep_indices=example_output_indices)

        new_init_preds = init_preds[example_indices]
        new_Y = Y[example_indices]

        if not large_model_regime:
            new_jacobians = dict()
            for k, v in jacobians.items():
                new_jacobians[k] = v[example_output_indices]
        else:
            new_jacobians = None

        new_dataset = Subset(dataset, example_indices)

        new_weights = get_weights_at_time_t(t=t, eta=eta * n / (n - 1), init_params=init_params,
                                            jacobians=new_jacobians, ntk=new_ntk, ntk_inv=new_ntk_inv,
                                            init_preds=new_init_preds, Y=new_Y, continuous=continuous,
                                            large_model_regime=large_model_regime, model=model, dataset=new_dataset,
                                            **kwargs)

        total_change = 0.0

        param_changes = dict()
        for k in old_weights.keys():
            param_changes[k] = (new_weights[k] - old_weights[k]).cpu()  # to save GPU memory

        if return_change_vectors:
            change_vectors.append(param_changes)

        if without_sgd:
            for k in old_weights.keys():
                total_change += torch.sum(param_changes[k] ** 2)
        else:
            param_changes = [v.flatten() for k, v in param_changes.items()]
            param_changes = torch.cat(param_changes, dim=0)

            total_change = torch.mm(param_changes.view((1, -1)),
                                    torch.mm(steady_state_inv_cov.cpu(), param_changes.view(-1, 1)))

        change_quantities.append(total_change)

    return change_vectors, change_quantities


@utils.with_no_grad
def test_pred_stability(t, n, eta, ntk, test_train_ntk, train_init_preds, test_init_preds,
                        train_Y, l2_reg_coef=0.0, continuous=False):
    if l2_reg_coef > 0:
        ntk = ntk + l2_reg_coef * torch.eye(ntk.shape[0], dtype=torch.float, device=ntk.device)

    ntk_inv = torch.inverse(ntk)

    old_preds = get_test_predictions_at_time_t(t=t, eta=eta,
                                               ntk=ntk,
                                               test_train_ntk=test_train_ntk,
                                               train_Y=train_Y,
                                               train_init_preds=train_init_preds,
                                               test_init_preds=test_init_preds,
                                               continuous=continuous,
                                               ntk_inv=ntk_inv)

    n_outputs = train_init_preds.shape[-1]
    change_vectors = []
    change_quantities = []
    for sample_idx in tqdm(range(n)):
        example_indices = [i for i in range(n) if i != sample_idx]
        example_output_indices = []
        for i in example_indices:
            example_output_indices.extend(range(i * n_outputs, (i + 1) * n_outputs))

        new_ntk = ntk.clone()[example_output_indices]
        new_ntk = new_ntk[:, example_output_indices]
        new_test_train_ntk = test_train_ntk[:, example_output_indices]
        new_ntk_inv = misc.update_ntk_inv(ntk=ntk, ntk_inv=ntk_inv, keep_indices=example_output_indices)

        new_train_init_preds = train_init_preds[example_indices]
        new_train_Y = train_Y[example_indices]

        new_preds = get_test_predictions_at_time_t(
            t=t,
            eta=eta * n / (n-1),
            train_Y=new_train_Y,
            train_init_preds=new_train_init_preds,
            test_init_preds=test_init_preds,
            continuous=continuous,
            ntk=new_ntk,
            ntk_inv=new_ntk_inv,
            test_train_ntk=new_test_train_ntk)

        change_vectors.append(new_preds - old_preds)
        change_quantities.append(torch.sum((new_preds - old_preds) ** 2, dim=1).mean(dim=0))

    return change_vectors, change_quantities
