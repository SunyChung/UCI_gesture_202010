# base code : https://github.com/mhw32/multimodal-vae-public
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from .data_loader import load_1d_data

raw_train_x, raw_train_y = load_1d_data('raw', 'train')
raw_test_x, raw_test_y = load_1d_data('raw', 'test')


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

    def forward(self, feature, label):
        mu, logvar = self.infer(feature, label)
        z = self.reparametrize(mu, logvar)
        feature_recon = self.feature_decoder(z)
        label_recon = self.label_decoder(z)
        return feature_recon, label_recon, mu, logvar

    def infer(self, features, labels):
        batch_size = features.size(0)
        use_cuda = next(self.parameters().is_cuda)

        mu, logvar = prior_expert((1, batch_size, self.n_latents), use_cuda=use_cuda)

        if features is not None:
            feture_mu, feature_logvar = self.feature_encoder(features)
            mu = torch.cat((mu, feture_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, feature_logvar.unsqueeze(0)), dim=0)

        if labels is not None:
            label_mu, label_logvar = self.label_encoder(labels)
            mu = torch.cat((mu, label_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, label_logvar.unsqueeze(0)), dim=0)

        mu, logvar = self.experts(mu, logvar)
        return mu, logvar


class FeatureEncoder(nn.Module):
    def __init__(self, n_latents):
        super(FeatureEncoder, self).__init__()
        self.fc1 = nn.Embedding(18, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, n_latents)
        self.fc32 = nn.Linear(512, n_latents)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x))
        h = self.swish(self.fc2(h))
        return self.fc31(h), self.fc32(h)


class FeatureDecoder(nn.Module):
    def __init__(self, n_latents):
        super(FeatureDecoder, self).__init__()
        self.fc1 = nn.Linear(n_latents, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 18)
        self.swish = Swish()

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
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
        h = self.swish(self.fc1(x))
        h = self.swish(self.fc2(h))
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
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
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


def prior_expert(size, use_cuda=False):
    mu = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar
