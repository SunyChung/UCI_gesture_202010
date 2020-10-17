# base code : https://github.com/mhw32/multimodal-vae-public
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

use_cuda = False


class MVAE(nn.Module):
    def __init__(self, n_latents):
        super(MVAE, self).__init__()
        self.feature_encoder = FeatureEncoder(n_latents)
        self.feature_decoder = FeatureDecoder(n_latents)
        self.label_encoder = LabelEncoder(n_latents)
        self.label_decoder = LabelDecoder(n_latents)
        self.experts = ProductOfExperts()
        self.n_latents = n_latents

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, feature=None, label=None):
        mu, logvar = self.infer(feature, label)
        z = self.reparametrize(mu, logvar)
        feature_recon = self.feature_decoder(z)
        label_recon = self.label_decoder(z)
        return feature_recon, label_recon, mu, logvar

    def infer(self, features=None, labels=None):
        # AttributeError: 'NoneType' object has no attribute 'size'
        if features is not None:
            batch_size = features.size(0)
        else:
            batch_size = labels.size(0)

        # def prior_expert(size, use_cuda)
        # default mu, logvar size : mu = Variable(torch.zeros(size))
        mu, logvar = prior_expert((1, batch_size, self.n_latents), use_cuda=use_cuda)

        if features is not None:
            feture_mu, feature_logvar = self.feature_encoder(features)
            # RuntimeError: All input tensors must be on the same device. Received cuda:0 and cpu
            # -> confusing to set it up properly, thus, turned off cuda execution for now;

            # RuntimeError: Tensors must have same number of dimensions: got 3 and 4
            # print(mu.shape)  # torch.Size([1, 8, 64])
            # print(feture_mu.shape)  # torch.Size([8, 64])
            # print(feture_mu.unsqueeze(0).shape)  # torch.Size([1, 8, 64])
            mu = torch.cat((mu, feture_mu.unsqueeze(0)), dim=0)

            # print(logvar.shape)  # torch.Size([1, 8, 64])
            # print(feature_logvar.shape)  # torch.Size([8, 64])
            logvar = torch.cat((logvar, feature_logvar.unsqueeze(0)), dim=0)

        if labels is not None:
            label_mu, label_logvar = self.label_encoder(labels)
            # print(mu.shape)  # torch.Size([2, 8, 64])
            # print(label_mu.shape)  # torch.Size([8, 64])
            mu = torch.cat((mu, label_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, label_logvar.unsqueeze(0)), dim=0)
            # print(mu.shape)  # torch.Size([3, 8, 64]) : keeps building up the batch size,
            # only to stop due to other error!
            # print(logvar.shape) # torch.Size([3, 8, 64])
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar


class FeatureEncoder(nn.Module):
    def __init__(self, n_latents):
        super(FeatureEncoder, self).__init__()
        self.fc1 = nn.Linear(19, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, n_latents)
        self.fc32 = nn.Linear(512, n_latents)
        self.swish = Swish()

    def forward(self, x):
        # print('feature encoder')
        # print('feature encoder input shape : ', x.shape)  # torch.Size([8, 19])
        h = self.swish(self.fc1(x))
        # print(h.shape)  # torch.Size([8, 512])
        h = self.swish(self.fc2(h))
        # print(h.shape)  # torch.Size([8, 512])
        return self.fc31(h), self.fc32(h)


class FeatureDecoder(nn.Module):
    def __init__(self, n_latents):
        super(FeatureDecoder, self).__init__()
        self.fc1 = nn.Linear(n_latents, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 19)
        self.swish = Swish()

    def forward(self, z):
        # rint('feature decoder')
        h = self.swish(self.fc1(z))
        # print(h.shape)  # torch.Size([8, 512])
        h = self.swish(self.fc2(h))
        # print(h.shape)  # torch.Size([8, 512])
        h = self.swish(self.fc3(h))
        # print(h.shape)  # torch.Size([8, 512])
        return self.fc4(h)


class LabelEncoder(nn.Module):
    def __init__(self, n_latents):
        super(LabelEncoder, self).__init__()
        self.fc1 = nn.Embedding(5, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, n_latents)
        self.fc32 = nn.Linear(512, n_latents)
        self.swish = Swish()

    def forward(self, x):
        # print('label encoder input shape : ', x.shape)  # torch.Size([8, 1])
        # print('label encoder')
        h = self.swish(self.fc1(x))
        # print(h.shape)  # torch.Size([8, 1, 512])
        h = self.swish(self.fc2(h))
        # print(h.shape)  # torch.Size([8, 1, 512])
        h = h.squeeze(1)
        # print(h.shape)  # torch.Size([8, 512])
        return self.fc31(h), self.fc32(h)


class LabelDecoder(nn.Module):
    def __init__(self, n_latents):
        super(LabelDecoder, self).__init__()
        self.fc1 = nn.Linear(n_latents, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 5)
        self.swish = Swish()

    def forward(self, z):
        # print('label decoder')
        h = self.swish(self.fc1(z))
        # print(h.shape)  # torch.Size([8, 512])
        h = self.swish(self.fc2(h))
        # print(h.shape)  # torch.Size([8, 512])
        h = self.swish(self.fc3(h))
        # print(h.shape)  # torch.Size([8, 512])
        return self.fc4(h)


class ProductOfExperts(nn.Module):
    def forward(self, mu, logvar, eps=1e-10):
        var = torch.exp(logvar) + eps
        T = 1. / (var + eps)
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)


def prior_expert(size, use_cuda):
    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar
