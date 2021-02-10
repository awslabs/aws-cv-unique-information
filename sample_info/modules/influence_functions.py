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

from tqdm import tqdm

from torch.utils.data import DataLoader
import torch

from nnlib.nnlib import utils


def jacobian(ys, xs, create_graph=False):
    jac = [[[] for i in range(len(xs))] for j in range(len(ys))]
    for y_idx, y in enumerate(ys):
        flat_y = y.reshape(-1)
        grad_y = torch.zeros_like(flat_y)

        for i in range(len(flat_y)):
            grad_y[i] = 1.
            grad_xs = torch.autograd.grad(flat_y, xs, grad_y, retain_graph=True, create_graph=create_graph)
            for x_idx, grad_x in enumerate(grad_xs):
                jac[y_idx][x_idx].append(grad_x)
            grad_y[i] = 0.

        for x_idx, x in enumerate(xs):
            jac[y_idx][x_idx] = torch.stack(jac[y_idx][x_idx]).reshape(y.shape + x.shape)

    return jac


def hessian(ys, xs):
    assert len(ys) == 1
    jac = jacobian(ys, xs, create_graph=True)
    jac = jac[0]
    return jacobian(ys=jac, xs=xs)


def hessian_vector_product(loss, params, v):
    """
    :param loss: the final loss, whose Hessian is of interest
    :param params: list of parameters
    :param v: list of parameters, denoting the vector in the Hessian-vector product
    reference: https://github.com/kohpangwei/influence-release/blob/master/influence/hessians.py#L10
    """
    assert len(params) == len(v)
    for a, b in zip(params, v):
        assert a.shape == b.shape
    grads = torch.autograd.grad(loss, params, create_graph=True)
    dot_product = 0.0
    for grad_elem, v_elem in zip(grads, v):
        dot_product = dot_product + torch.sum(grad_elem * v_elem.detach())
    return torch.autograd.grad(dot_product, params)


@utils.with_no_grad
def inverse_hvp_lissa(model, dataset, v, batch_size=128, scale=10,
                      damping=0.0, num_samples=1, recursion_depth=1000, num_workers=0):
    """
    reference: https://github.com/kohpangwei/influence-release/blob/master/influence/genericNeuralNet.py#L475
    """
    model.eval()
    inverse_hvp = None
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    for i in range(num_samples):
        cur_estimate = v.copy()

        for _ in tqdm(range(recursion_depth), desc='computing inverse hvp'):
            for x, y in loader:
                with torch.set_grad_enabled(True):
                    outputs = model.forward(inputs=[x], labels=[y])
                    losses, _ = model.compute_loss(inputs=[x], labels=[y], outputs=outputs)
                    total_loss = sum([losses[k] for k in losses.keys()])
                    hv = hessian_vector_product(loss=total_loss, params=tuple(model.parameters()), v=cur_estimate)

                cur_estimate = [a + (1 - damping) * b - c / scale for (a, b, c) in zip(v, cur_estimate, hv)]

                break

        if inverse_hvp is None:
            inverse_hvp = [b / scale for b in cur_estimate]
        else:
            inverse_hvp = [a + b / scale for (a, b) in zip(inverse_hvp, cur_estimate)]

    inverse_hvp = [a / num_samples for a in inverse_hvp]
    return inverse_hvp
