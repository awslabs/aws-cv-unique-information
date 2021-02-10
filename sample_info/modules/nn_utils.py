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

import torch

from nnlib.nnlib import nn_utils as nn_utils_base
from nnlib.nnlib import utils
from sample_info import methods


class SimpleDetailedOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SimpleDetailedOutputWrapper, self).__init__()
        self.sequential_model = model

    def forward(self, *args, **kwargs):
        pred = self.sequential_model.forward(*args, **kwargs)
        return {
            'pred': pred
        }


def parse_network_from_config(args, input_shape, detailed_output=False):
    """Parses a sequential feed-forward neural network from json config."""

    # parse project-specific networks
    if isinstance(args, dict) and args['net'] == 'emnist-letters-pretrained-cnn':
        checkpoint_path = 'sample_info/modules/resources/emnist_letters_cnn_pretrained.mdl'
        net = utils.load(path=checkpoint_path, methods=methods, device='cpu').classifier.sequential_model

        num_classes = args.get('num_classes', 26)
        if num_classes != 26:
            layers = list(net.children())
            replace = torch.nn.Linear(in_features=layers[-1].in_features, out_features=num_classes)
            layers[-1] = replace
            net = torch.nn.Sequential(*layers)

        output_shape = (None, num_classes)
        print("output.shape:", output_shape)

    elif isinstance(args, dict) and args['net'] in ['leaky-resnet18', 'leaky-resnet34', 'leaky-resnet50']:
        from sample_info.modules.resnet_leaky_relu import resnet18, resnet34, resnet50

        resnet_fn = None
        if args['net'] == 'leaky-resnet18':
            resnet_fn = resnet18
        if args['net'] == 'leaky-resnet34':
            resnet_fn = resnet34
        if args['net'] == 'leaky-resnet50':
            resnet_fn = resnet50

        norm_layer = torch.nn.BatchNorm2d
        if args.get('norm_layer', '') == 'GroupNorm':
            norm_layer = nn_utils_base.group_norm_partial_apply_fn(num_groups=32)
        if args.get('norm_layer', '') == 'none':
            norm_layer = (lambda num_channels: nn_utils_base.Identity())

        num_classes = args.get('num_classes', 1000)
        pretrained = args.get('pretrained', False)

        # if pretraining is enabled but number of classes is not 1000 replace the last layer
        if pretrained and num_classes != 1000:
            net = resnet_fn(norm_layer=norm_layer, num_classes=1000, pretrained=pretrained)
            net.fc = torch.nn.Linear(net.fc.in_features, num_classes)
        else:
            net = resnet_fn(norm_layer=norm_layer, num_classes=num_classes, pretrained=pretrained)
        output_shape = nn_utils_base.infer_shape([net], input_shape)
        print("output.shape:", output_shape)

    else:
        # parse general-case networks
        net, output_shape = nn_utils_base.parse_network_from_config(args, input_shape)

    if detailed_output:
        net = SimpleDetailedOutputWrapper(net)

    return net, output_shape
