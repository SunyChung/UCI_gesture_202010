# base code : https://github.com/mhw32/multimodal-vae-public
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

PREFIX_LIST = ['a1', 'a2', 'a3', 'b1', 'b3', 'c1', 'c3']
WINDOW_SIZE = 8


def load_1d_data(data_name, data_type):
    train_test_path = 'dataset/train_test/'

    if data_name == 'raw':
        data_dict = {'Hold': 0, 'Rest': 1, 'Preparation': 2, 'Retraction': 3, 'Stroke': 4}
    else:
        data_dict = {'H': 0, 'D': 1, 'P': 2, 'R': 3, 'S': 4}

    x = []
    y = []
    for prefix in PREFIX_LIST:
        data = open(train_test_path + str(prefix) + '_' + str(data_name) + '_' + str(data_type) + '.csv', 'r')
        label = open(train_test_path + str(prefix) + '_' + str(data_name) + '_' + str(data_type) + '_label.txt', 'r')

        for line in data:
            temp = []
            for value in line.rstrip().split(','):
                temp = temp + [float(value)]
            # print(np.shape(temp))
            x.append(temp)

        for l in label:
            y.append(data_dict[l.rstrip().replace(',', '')])

    if data_name == 'raw':
        x = np.array(x).reshape((-1, WINDOW_SIZE, 18))
    else:
        x = np.array(x).reshape((-1, WINDOW_SIZE, 32))
    y = np.array(y).reshape((-1, 1))
    print(np.shape(x))
    print(np.shape(y))
    return x, y


raw_train_x, raw_train_y = load_1d_data('raw', 'train')
raw_test_x, raw_test_y = load_1d_data('raw', 'test')

va3_train_x, va3_train_y = load_1d_data('va3', 'train')
va3_test_x, va3_test_y = load_1d_data('va3', 'test')


class MVAE(nn.Module):
    def __init__(self, n_latents):
        super(MVAE, self).__init__()
        self.raw_encoder = RawEncoder(n_latents, n_coords=18)
        self.raw_decoder = RawDecoder(n_latents, n_coords=18)
        self.va3_encoder = Va3Encoder(n_latents, n_coords=32)
        self.va3_decoder = Va3Decoder(n_latents, n_coords=32)
        self.experts = ProductOfExperts()
        self.n_latents = n_latents

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, raw, va3):
        mu, logvar = self.infer(raw, va3)
        z = self.reparametrize(mu, logvar)
        raw_recon = self.raw_decoder(z)
        va3_recon = self.va3_decoder(z)
        return raw_recon, va3_recon, mu, logvar

    def infer(self, raw, va3):
        batch_size = raw.size(0)
        use_cuda = next(self.parameters().is_cuda)

        mu, logvar = prior_expert((1, batch_size, self.n_latents), use_cuda=use_cuda)

        if raw is not None:
            raw_mu, raw_logvar = self.raw_encoder(raw)
            mu = torch.cat((mu, raw_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, raw_logvar.unsqueeze(0)), dim=0)

        if va3 is not None:
            va3_mu, va3_logvar = self.va3_vel_encoder(va3)
            mu = torch.cat((mu, va3_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, va3_logvar.unsqueeze(0)), dim=0)

        mu, logvar = self.experts(mu, logvar)
        return mu, logvar


class RawEncoder(nn.Module):
    def __init__(self, n_latents, n_coords):
        super(RawEncoder, self).__init__()
        self.fc1 = nn.Embedding(n_coords, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, n_latents)
        self.fc32 = nn.Linear(512, n_latents)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x))
        h = self.swish(self.fc2(h))
        return self.fc31(h), self.fc32(h)


class RawDecoder(nn.Module):
    def __init__(self, n_latents, n_coords):
        super(RawDecoder, self).__init__()
        self.fc1 = nn.Linear(n_latents, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, n_coords)
        self.swish = Swish()

    def forward(self, z):
        h = self.swish(self.fc1(z))
        h = self.swish(self.fc2(h))
        h = self.swish(self.fc3(h))
        return self.fc4(h)


class Va3Encoder(nn.Module):
    def __init__(self, n_latents, n_coords):
        super(Va3Encoder, self).__init__()
        self.fc1 = nn.Embedding(n_coords, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, n_latents)
        self.fc32 = nn.Linear(512, n_latents)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x))
        h = self.swish(self.fc2(h))
        return self.fc31(h), self.fc32(h)


class Va3Decoder(nn.Module):
    def __init__(self, n_latents, n_coords):
        super(Va3Decoder, self).__init__()
        self.fc1 = nn.Linear(n_latents, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, n_coords)
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
