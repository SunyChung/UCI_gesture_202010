# base code : https://github.com/younggyoseo/vae-cf-pytorch

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, q_dims, dropout):
        super().__init__()
        self.q_dims = q_dims
        q_dims_ = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.Sequential()
        for i, (p, inp, out) in enumerate(zip(dropout, q_dims_[:-1], q_dims_[1:])):
            self.q_layers.add_module('_'.join(['dropout', str(i)]), nn.Dropout(p))
            self.q_layers.add_module('_'.join(['linear', str(i)]), nn.Linear(inp, out))

    def forward(self, x):
        h = F.normalize(x, p=2, dim=1)
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                mu, logvar = torch.split(h, self.q_dims[-1], dim=1)
        return mu, logvar

    def init_weights(self):
        for layer in self.q_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)


class Decoder(nn.Module):
    def __init__(self, p_dims, dropout):
        super().__init__()

        self.p_layers = nn.Sequential()
        for i, (p, inp, out) in enumerate(zip(dropout, p_dims[:-1], p_dims[1:])):
            self.p_layers.add_module('_'.join(['dropout', str(i)]), nn.Dropout(p))
            self.p_layers.add_module('_'.join(['linear', str(i)]), nn.Linear(inp, out))

    def forward(self, x):
        h = x
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def init_weights(self):
        for layer in self.p_layers:
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)


class MultiVAE(nn.Module):
    def __init__(self, p_dims, q_dims,
                 dropout_enc, dropout_dec):
        super(MultiVAE, self).__init__()

        self.encoder = Encoder(q_dims, dropout_enc)
        self.decoder = Decoder(p_dims, dropout_dec)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        return self.decoder(z), mu, logvar

    def reparametrize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.rand_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu