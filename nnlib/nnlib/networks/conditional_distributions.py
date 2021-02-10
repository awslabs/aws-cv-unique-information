""" Conditional distributions should return list of parameters.
They should have these functions defined:
    - sample(params)
    - mean(params)
    - kl_divergence
"""
from torch import nn
import torch


class ConditionalGaussian(nn.Module):
    """Conditional Gaussian distribution, where the mean and variance are
    parametrized with neural networks."""
    def __init__(self, net, mu, logvar):
        super(ConditionalGaussian, self).__init__()
        self.net = net
        self.mu = mu
        self.logvar = logvar

    def forward(self, x):
        h = self.net(x)
        return {
            'param:mu': self.mu(h),
            'param:logvar': self.logvar(h)
        }

    @staticmethod
    def mean(params):
        return params['param:mu']

    @staticmethod
    def sample(params):
        mu = params['param:mu']
        logvar = params['param:logvar']
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def kl_divergence(params):
        """ Computes KL(q(z|x) || p(z)) assuming p(z) is N(0, I). """
        mu = params['param:mu']
        logvar = params['param:logvar']
        kl = -0.5 * torch.sum(1 + logvar - (mu ** 2) - torch.exp(logvar), dim=1)
        return torch.mean(kl, dim=0)


class ConditionalUniform(nn.Module):
    """Conditional uniform distribution parametrized with a NN. """
    def __init__(self, net, center, radius):
        super(ConditionalUniform, self).__init__()
        self.net = net
        self.center = center
        self.radius = radius

    def forward(self, x):
        h = self.net(x)
        c = torch.tanh(self.center(h))
        r = torch.sigmoid(self.radius(h))
        return {
            'param:left': torch.clamp(c - r, -1, +1),
            'param:right': torch.clamp(c + r, -1, +1)
        }

    @staticmethod
    def mean(params):
        return 0.5 * (params['param:left'] + params['param:right'])

    @staticmethod
    def sample(params):
        left = params['param:left']
        right = params['param:right']
        eps = torch.rand_like(left)
        return left + (right - left) * eps

    @staticmethod
    def kl_divergence(params):
        """ Computes KL(q(z|x) || p(z)) assuming p(z) is U(-1, +1). """
        left = params['param:left']
        right = params['param:right']
        kl = torch.sum(torch.log(2.0 / (right - left + 1e-6)), dim=1)
        return torch.mean(kl, dim=0)


class ConditionalDiracDelta(nn.Module):
    """Conditional Dirac delta distribution parametrized with a NN.
    This can be used to make the VAE class act like a regular AE. """
    def __init__(self, net):
        super(ConditionalDiracDelta, self).__init__()
        self.net = net

    def forward(self, x):
        return {
            'param:mu': self.net(x)
        }

    @staticmethod
    def mean(params):
        return params['param:mu']

    @staticmethod
    def sample(params):
        return params['param:mu']

    @staticmethod
    def kl_divergence(params):
        return 0
