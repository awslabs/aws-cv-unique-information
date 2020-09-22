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
import copy

import torch
import torch.nn.functional as F
import torch.autograd

from nnlib.nnlib.utils import capture_arguments_of_init
from nnlib.nnlib import losses, utils
from sample_info.methods import BaseClassifier
from sample_info.modules import nn_utils
from sample_info.modules.ntk import JacobianEstimator
import nnlib.nnlib.method_utils


class StandardClassifier(BaseClassifier):
    @capture_arguments_of_init
    def __init__(self, input_shape, architecture_args, device='cuda', **kwargs):
        super(StandardClassifier, self).__init__(**kwargs)

        self.args = None  # this will be modified by the decorator
        self.input_shape = [None] + list(input_shape)
        self.architecture_args = architecture_args

        # create the network
        self.classifier, output_shape = nn_utils.parse_network_from_config(args=self.architecture_args['classifier'],
                                                                           input_shape=self.input_shape,
                                                                           detailed_output=True)
        self.num_classes = output_shape[-1]
        self.classifier = self.classifier.to(device)

    def forward(self, inputs, labels=None, detailed_output=False, **kwargs):
        x = inputs[0].to(self.device)

        details = self.classifier(x)
        pred = details.pop('pred')

        out = {
            'pred': pred
        }

        if detailed_output:
            for k, v in details.items():
                out[k] = v

        return out

    def compute_loss(self, inputs, labels, outputs, **kwargs):
        pred = outputs['pred']
        y = labels[0].to(self.device)

        # classification loss
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        classifier_loss = losses.get_classification_loss(target=y_one_hot, logits=pred,
                                                         loss_function='ce')

        batch_losses = {
            'classifier': classifier_loss,
        }

        return batch_losses, outputs


class ClassifierL2(StandardClassifier):
    @capture_arguments_of_init
    def __init__(self, input_shape, architecture_args, l2_reg_coef=0.0, device='cuda', seed=42, **kwargs):
        torch.manual_seed(seed)
        super(ClassifierL2, self).__init__(input_shape, architecture_args, device=device, **kwargs)
        self.l2_reg_coef = l2_reg_coef
        if l2_reg_coef > 0:
            self.init_params = copy.deepcopy(dict(self.classifier.named_parameters()))
            for k, v in self.init_params.items():
                v.detach_()
                v.requires_grad = False  # to stop training

    def compute_loss(self, inputs, labels, outputs, **kwargs):
        pred = outputs['pred']
        y = labels[0].to(self.device).float()

        # classification loss
        n_outputs = pred.shape[-1]
        y = y.reshape((-1, n_outputs))
        classifier_loss = 0.5 * torch.sum((y - pred) ** 2, dim=1).mean(dim=0)

        batch_losses = {
            'classifier': classifier_loss,
        }

        if self.l2_reg_coef > 0:
            # add ||w - w_0||^2 type regularization
            l2_penalty = 0.0
            for k, v in self.classifier.named_parameters():
                l2_penalty = l2_penalty + torch.sum((v - self.init_params[k]) ** 2)
            batch_losses['l2_penalty'] = 0.5 * self.l2_reg_coef * l2_penalty

        return batch_losses, outputs


class ClassifierL2WithGradCollector(ClassifierL2):
    @capture_arguments_of_init
    def __init__(self, input_shape, architecture_args, device='cuda', **kwargs):
        super(ClassifierL2WithGradCollector, self).__init__(input_shape=input_shape,
                                                            architecture_args=architecture_args,
                                                            device=device,
                                                            **kwargs)
        self._grad_updates = defaultdict(dict)

    def forward(self, inputs, labels=None, detailed_output=False, partition=None, **kwargs):
        out = super(ClassifierL2WithGradCollector, self).forward(inputs=inputs, labels=labels,
                                                                 detailed_output=detailed_output,
                                                                 partition=partition,
                                                                 **kwargs)
        pred = out['pred']
        y = labels[0].to(self.device).float()

        # classification loss
        n_outputs = pred.shape[-1]
        y = y.reshape((-1, n_outputs))
        classifier_loss = 0.5 * torch.sum((y - pred) ** 2, dim=1)  # compute per-sample MSE

        if partition == 'train':
            n = pred.shape[0]
            for sample_idx in range(n):
                real_sample_idx = inputs[1][sample_idx].item()
                storage = self._grad_updates[real_sample_idx]
                grad = torch.autograd.grad(1.0 / n * classifier_loss[sample_idx], tuple(self.parameters()),
                                           retain_graph=True)
                for (name, _), v in zip(self.named_parameters(), grad):
                    if name not in storage:
                        storage[name] = v.detach()
                    else:
                        storage[name] += v.detach()

        return out

    def visualize(self, *args, **kwargs):
        return {}


class LinearizedModel(nnlib.nnlib.method_utils.Method):
    """
    NOTE: this linearized model works for binary classification only
    TODO: add code for handing multiple outputs
    """
    @capture_arguments_of_init
    def __init__(self, model, **kwargs):
        super(LinearizedModel, self).__init__(**kwargs)
        self.model = model

        # clone the model at initialization and disable training
        model_at_init = copy.deepcopy(model)
        self.excluded_models = [model_at_init]

    def forward(self, inputs, labels=None, **kwargs):
        x = inputs[0].to(self.device)

        # compute predictions and Jacobians at initialization
        model_at_init = self.excluded_models[0]

        with torch.set_grad_enabled(True):
            init_preds = model_at_init(inputs=inputs, labels=labels,
                                       grad_enabled=True,
                                       detailed_output=False,
                                       **kwargs)['pred']

            jacobians = defaultdict(list)
            n = x.shape[0]
            for idx in range(n):
                # retain graph is needed as we need some parts of the forward computational graph
                # are shared between different examples. Don't retain the graph if it is the last example.
                cur_jacobians = torch.autograd.grad(init_preds[idx][0], model_at_init.parameters(),
                                                    retain_graph=(idx != n - 1), create_graph=False)

                for (k, _), v in zip(model_at_init.named_parameters(), cur_jacobians):
                    jacobians[k].append(v)

        for k, v in jacobians.items():
            jacobians[k] = torch.stack(v, dim=0).detach()

        # compute predictions of linearized network
        pred = init_preds.detach()

        for name, init_grad in jacobians.items():
            init_grad = init_grad.reshape((n, -1))
            init_value = dict(model_at_init.named_parameters())[name]
            init_value = init_value.reshape((1, -1))
            cur_value = dict(self.model.named_parameters())[name]
            cur_value = cur_value.reshape((1, -1))
            add = torch.mm(init_grad, (cur_value - init_value).T)
            pred = pred + add

        out = {
            'pred': pred
        }

        # detach parameters of model_at_init. This makes sure that even when backward() is not called,
        # which happens during the validation, the computation graph for computing Jacobians will be cleared.
        for param in model_at_init.parameters():
            param.detach_()
            param.requires_grad = True

        return out

    def compute_loss(self, inputs, labels, outputs, **kwargs):
        pred = outputs['pred']
        y = labels[0].to(self.device).float()

        # classification loss
        n_outputs = pred.shape[-1]
        y = y.reshape((-1, n_outputs))
        classifier_loss = 0.5 * torch.sum((y - pred) ** 2, dim=1).mean(dim=0)

        batch_losses = {
            'classifier': classifier_loss,
        }

        return batch_losses, outputs


class LinearizedModelV2(nnlib.nnlib.method_utils.Method):
    """
    NOTE: Supports only full batch training. The train and val loaders should not shuffle examples.
    """
    @capture_arguments_of_init
    def __init__(self, model, train_data, val_data=None, l2_reg_coef=0.0, **kwargs):
        super(LinearizedModelV2, self).__init__(**kwargs)
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.l2_reg_coef = l2_reg_coef

        # copy the parameters at initialization
        self.init_params = copy.deepcopy(dict(model.named_parameters()))
        for k, v in self.init_params.items():
            v.detach_()
            v.requires_grad = False  # to stop training

        # compute all gradients
        self.jacobians = dict()
        jacobian_estimator = JacobianEstimator(projection='none')
        self.jacobians['train'] = jacobian_estimator.compute_jacobian(model=model, dataset=train_data, cpu=False)
        if val_data is not None:
            self.jacobians['val'] = jacobian_estimator.compute_jacobian(model=model, dataset=val_data, cpu=False)
        for partition in self.jacobians.keys():
            for k, v in self.jacobians[partition].items():
                v.detach_()  # in case they some computation graph was built

        # compute predictions at initialization
        self.init_preds = dict()
        self.init_preds['train'] = utils.apply_on_dataset(model=model, dataset=train_data,
                                                          output_keys_regexp='pred', cpu=False)['pred']
        if val_data is not None:
            self.init_preds['val'] = utils.apply_on_dataset(model=model, dataset=val_data,
                                                            output_keys_regexp='pred', cpu=False)['pred']
        for partition in self.init_preds.keys():
            self.init_preds[partition].detach_()  # in case they some computation graph was built

    def forward(self, inputs, labels=None, partition=None, **kwargs):
        # we ignore inputs as the data is readily available
        pred = self.init_preds[partition].detach()
        if partition == 'train':
            n_samples = len(self.train_data)
        elif partition == 'val':
            n_samples = len(self.val_data)
        else:
            raise ValueError(f"Cannot recognize parition value: {partition}")

        n_outputs = pred.shape[-1]
        for name, init_grad in self.jacobians[partition].items():
            init_grad = init_grad.reshape((n_samples * n_outputs, -1))
            init_value = self.init_params[name]
            init_value = init_value.reshape((1, -1))
            cur_value = dict(self.model.named_parameters())[name]
            cur_value = cur_value.reshape((1, -1))
            add = torch.mm(init_grad, (cur_value - init_value).T)
            add = add.reshape((n_samples, n_outputs))
            pred = pred + add

        out = {
            'pred': pred
        }

        return out

    def compute_loss(self, inputs, labels, outputs, **kwargs):
        pred = outputs['pred']
        y = labels[0].to(self.device).float()

        # classification loss
        n_outputs = pred.shape[-1]
        y = y.reshape((-1, n_outputs))
        classifier_loss = 0.5 * torch.sum((y - pred) ** 2, dim=1).mean(dim=0)

        batch_losses = {
            'classifier': classifier_loss,
        }

        if self.l2_reg_coef > 0:
            # add ||w - w_0||^2 type regularization
            l2_penalty = 0.0
            for k, v in self.model.named_parameters():
                l2_penalty = l2_penalty + torch.sum((v - self.init_params[k]) ** 2)
            batch_losses['l2_penalty'] = 0.5 * self.l2_reg_coef * l2_penalty

        return batch_losses, outputs
