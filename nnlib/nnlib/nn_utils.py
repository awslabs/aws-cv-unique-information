""" Some tools for building basic NN blocks """
from torch import nn
import torch
import numpy as np

from .networks.conditional_distributions import ConditionalGaussian, ConditionalUniform, \
    ConditionalDiracDelta


def infer_shape(layers, input_shape, key=None):
    """Given a list of layers representing a sequential model and its input_shape, infers the output shape."""
    input_shape = [x for x in input_shape]
    if input_shape[0] is None:
        input_shape[0] = 4  # should be more than 1, otherwise batch norm will not work
    x = torch.tensor(np.random.normal(size=input_shape), dtype=torch.float, device='cpu')
    for layer in layers:
        x = layer(x)
    if key is not None:
        x = x[key]
    output_shape = list(x.shape)
    output_shape[0] = None
    return output_shape


def add_activation(layers, activation):
    """Adds an activation function into a list of layers."""
    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'softplus':
        layers.append(nn.Softplus())
    elif activation == 'softmax':
        layers.append(nn.Softmax(dim=1))
    elif activation == 'leaky-relu0.1':
        layers.append(nn.LeakyReLU(negative_slope=0.1))
    elif activation == 'linear':
        pass
    else:
        raise ValueError(f"Activation function with name '{activation}' is not implemented.")
    return layers


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self._shape = tuple([-1, ] + list(shape))

    def forward(self, x):
        return x.view(self._shape)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def group_norm_partial_apply_fn(num_groups=32):
    def fn(num_channels):
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    return fn


def parse_feed_forward(args, input_shape):
    """Parses a sequential feed-forward neural network from json config."""
    net = []
    for cur_layer in args:
        layer_type = cur_layer['type']
        prev_shape = infer_shape(net, input_shape)
        print(prev_shape)

        if layer_type == 'fc':
            dim = cur_layer['dim']
            assert len(prev_shape) == 2
            net.append(nn.Linear(prev_shape[1], dim))
            if cur_layer.get('batch_norm', False):
                net.append(nn.BatchNorm1d(dim))
            add_activation(net, cur_layer.get('activation', 'linear'))
            if 'dropout' in cur_layer:
                net.append(nn.Dropout(cur_layer['dropout']))

        if layer_type == 'flatten':
            net.append(Flatten())

        if layer_type == 'reshape':
            net.append(Reshape(cur_layer['shape']))

        if layer_type == 'conv':
            assert len(prev_shape) == 4
            net.append(nn.Conv2d(
                in_channels=prev_shape[1],
                out_channels=cur_layer['filters'],
                kernel_size=cur_layer['kernel_size'],
                stride=cur_layer['stride'],
                padding=cur_layer.get('padding', 0)
            ))
            if cur_layer.get('batch_norm', False):
                net.append(torch.nn.BatchNorm2d(
                    num_features=cur_layer['filters']))
            add_activation(net, cur_layer.get('activation', 'linear'))

        if layer_type == 'deconv':
            assert len(prev_shape) == 4
            net.append(nn.ConvTranspose2d(
                in_channels=prev_shape[1],
                out_channels=cur_layer['filters'],
                kernel_size=cur_layer['kernel_size'],
                stride=cur_layer['stride'],
                padding=cur_layer.get('padding', 0),
                output_padding=cur_layer.get('output_padding', 0)
            ))
            if cur_layer.get('batch_norm', False):
                net.append(torch.nn.BatchNorm2d(
                    num_features=cur_layer['filters']))
            add_activation(net, cur_layer.get('activation', 'linear'))

        if layer_type == 'identity':
            net.append(Identity())

        if layer_type == 'upsampling':
            net.append(torch.nn.UpsamplingNearest2d(
                scale_factor=cur_layer['scale_factor']
            ))

        if layer_type == 'gaussian':
            # this has to be the last layer
            net = nn.Sequential(*net)
            output_shape = infer_shape(net, input_shape)
            mu = nn.Sequential(nn.Linear(output_shape[1], cur_layer['dim']))
            logvar = nn.Sequential(nn.Linear(output_shape[1], cur_layer['dim']))
            output_shape = [None, cur_layer['dim']]
            print("output.shape:", output_shape)
            return ConditionalGaussian(net, mu, logvar), output_shape

        if layer_type == 'uniform':
            # this has to be the last layer
            net = nn.Sequential(*net)
            output_shape = infer_shape(net, input_shape)
            center = nn.Sequential(nn.Linear(output_shape[1], cur_layer['dim']))
            radius = nn.Sequential(nn.Linear(output_shape[1], cur_layer['dim']))
            output_shape = [None, cur_layer['dim']]
            print("output.shape:", output_shape)
            return ConditionalUniform(net, center, radius), output_shape

        if layer_type == 'dirac':
            # this has to be the last layer
            net = nn.Sequential(*net)
            output_shape = infer_shape(net, input_shape)
            print("output.shape:", output_shape)
            return ConditionalDiracDelta(net), output_shape

    output_shape = infer_shape(net, input_shape)
    print("output.shape:", output_shape)
    return nn.Sequential(*net), output_shape


def parse_network_from_config(args, input_shape):
    """Parses neural network architectures from json config."""

    # parse standard cases
    if isinstance(args, dict):
        if args['net'] in ['resnet18', 'resnet34', 'resnet50']:
            from torchvision.models import resnet18, resnet34, resnet50

            resnet_fn = None
            if args['net'] == 'resnet18':
                resnet_fn = resnet18
            if args['net'] == 'resnet34':
                resnet_fn = resnet34
            if args['net'] == 'resnet50':
                resnet_fn = resnet50

            norm_layer = torch.nn.BatchNorm2d
            if args.get('norm_layer', '') == 'GroupNorm':
                norm_layer = group_norm_partial_apply_fn(num_groups=32)
            if args.get('norm_layer', '') == 'none':
                norm_layer = (lambda num_channels: Identity())

            num_classes = args.get('num_classes', 1000)
            pretrained = args.get('pretrained', False)

            # if pretraining is enabled but number of classes is not 1000 replace the last layer
            if pretrained and num_classes != 1000:
                net = resnet_fn(norm_layer=norm_layer, num_classes=1000, pretrained=pretrained)
                net.fc = nn.Linear(net.fc.in_features, num_classes)
            else:
                net = resnet_fn(norm_layer=norm_layer, num_classes=num_classes, pretrained=pretrained)
            output_shape = infer_shape([net], input_shape)
            print("output.shape:", output_shape)
            return net, output_shape

        if args['net'] in ['resnet18-cifar', 'resnet34-cifar']:
            from .networks.resnet_cifar import resnet18, resnet34

            resnet_fn = None
            if args['net'] == 'resnet18-cifar':
                resnet_fn = resnet18
            if args['net'] == 'resnet34-cifar':
                resnet_fn = resnet34

            norm_layer = torch.nn.BatchNorm2d
            if args.get('norm_layer', '') == 'GroupNorm':
                norm_layer = group_norm_partial_apply_fn(num_groups=32)
            if args.get('norm_layer', '') == 'none':
                norm_layer = (lambda num_channels: Identity())
            net = resnet_fn(num_classes=args['num_classes'], norm_layer=norm_layer)
            output_shape = infer_shape([net], input_shape)
            print("output.shape:", output_shape)
            return net, output_shape

        if args['net'] in ['densenet121']:
            from torchvision.models import densenet121

            num_classes = args.get('num_classes', 1000)
            pretrained = args.get('pretrained', False)

            # if pretraining is enabled but number of classes is not 1000 replace the last layer
            if pretrained and num_classes != 1000:
                net = densenet121(num_classes=1000, pretrained=pretrained)
                net.classifier = nn.Linear(net.classifier.in_features, num_classes)
            else:
                net = densenet121(num_classes=num_classes, pretrained=pretrained)
            output_shape = infer_shape([net], input_shape)
            print("output.shape:", output_shape)
            return net, output_shape

    # parse feed forward
    return parse_feed_forward(args, input_shape)
