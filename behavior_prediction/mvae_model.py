# base code : https://github.com/mhw32/multimodal-vae-public

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.parameter import Parameter


DATA_NAME_LIST = ['a1', 'a2', 'a3', 'b1', 'b3', 'c1', 'c3']
WINDOW_SIZE = 8


def load_data(data_name, data_type, coord_len, sub_type=None):
	train_test_path = 'dataset/train_test/'

	if data_name == 'raw':
		data_dict = {'Hold' : 0, 'Rest' : 1, 'Preparation' : 2, 'Retraction' : 3, 'Stroke' : 4}
	else:
		data_dict = {'H' : 0, 'D' : 1, 'P' : 2, 'R' : 3, 'S' : 4}
	x = []
	y = []
	for name in DATA_NAME_LIST:
		if sub_type is None:
			data = open(train_test_path + str(name) + '_' + str(data_name) + '_' + str(data_type) + '.csv', 'r')
			label = open(train_test_path + str(name) + '_' + str(data_name) + '_' + str(data_type) + '_label.txt', 'r')
		else:
			data = open(train_test_path + str(name) + '_' + str(data_name) + '_' + str(sub_type) + '_' + str(data_type) + '.csv', 'r')
			label = open(train_test_path + str(name) + '_' + str(data_name) + '_' + str(sub_type) + '_' + str(data_type) + '_label.txt', 'r')			
		for line in data:
			temp = []
			for value in line.rstrip().split(','):
				temp = temp + [float(value)]
			x.append(temp)
		for l in label:
			y.append(data_dict[l.rstrip()])
	print(np.shape(x))
	x = np.array(x).reshape(len(y), WINDOW_SIZE, coord_len)
	y = np.array(y)
	print(np.shape(x)) 
	print(np.shape(y))
	return x, y

raw_train_x, raw_train_y = load_data('raw', 'train', 18)
# x : (55104, 18) -> (6888, 8, 18)
# y : (6888,)
raw_test_x, raw_test_y = load_data('raw', 'test', 18)

va3_vel_train_x, va3_vel_train_y = load_data('va3', 'train', 12, 'vel')
va3_acc_train_x, va3_acc_train_y = load_data('va3', 'train', 12, 'acc')
va3_sca_train_x, va3_sca_train_y = load_data('va3', 'train', 8, 'sca')

va3_vel_test_x, va3_vel_test_y = load_data('va3', 'test', 12, 'vel')
va3_acc_test_x, va3_acc_test_y = load_data('va3', 'test', 12, 'acc')
va3_sca_test_x, va3_sca_test_y = load_data('va3', 'test', 8, 'sca')



class MVAE(nn.Module):
	def __init__(self, n_latents):
		super(MVAE, self).__init__()
		self.raw_encoder = CoordEncoder(n_latents, n_coords=18)
		self.raw_decoder = CoordDecoder(n_latents, n_coords=18)
		self.va3_vel_encoder = CoordEncoder(n_latents, n_coords=12)
		self.va3_vel_decoder = CoordDecoder(n_latents, n_coords=12)
		self.va3_acc_encoder = CoordEncoder(n_latents, n_coords=12)
		self.va3_acc_decoder = CoordDecoder(n_latents, n_coords=12)
		self.va3_sca_encoder = CoordEncoder(n_latents, n_coords=8)
		self.va3_sca_decoder = CoordDecoder(n_latents, n_coords=8)
		self.experts = ProductOfExperts()
		self.n_latents = n_latents

	def reparametrize(self, mu, logvar):
		if self.training:
			std = logvar.mul(0.5).exp_()
			eps = Variable(std.data.new(std.size()).normal_())
			return eps.mul(std).add_(mu)
		else:
			return mu

	def forward(self, raw, va3_vel, va3_acc, va3_sca):
		mu, logvar = self.infer(raw, va3_vel, va3_acc, va3_sca)
		z = self.reparametrize(mu, logvar)
		raw_recon = self.raw_decoder(z)
		va3_vel_recon = self.va3_vel_decoder(z)
		va3_acc_recon = self.va3_acc_decoder(z)
		va3_sca_recon = self.va3_sca_decoder(z)
		return raw_recon, va3_vel_recon, va3_acc_recon, va3_sca_recon, mu, logvar

	# how many experts can MVAE handle?
	# the paper just used 2 experts!
	def infer(self, raw, va3_vel, va3_acc, va3_sca):
		# the thing is that 'raw' & 'va3' data length doesn't have same length!
		# it differs by 4!
		# what should I adjust?!?!
		# batch_size =
		mu, logvar = prior_expert((), use_cuda=False)

		if raw is not None:
			raw_mu, raw_logvar = self.raw_encoder(raw)
			mu = torch.cat((mu, raw_mu.unsqueeze(0)), dim=0)
			logvar = torch.cat((logvar, raw_logvar.unsqueeze(0)), dim=0)

		if va3_vel is not None:
			va3_vel_mu, va3_vel_logvar = self.va3_vel_encoder(va3_vel)
			mu = torch.cat((mu, va3_vel_mu.unsqueeze(0)), dim=0)
			logvar = torch.cat((logvar, va3_vel_logvar.unsqueeze(0)), dim=0)

		if va3_acc is not None:
			va3_acc_mu, va3_acc_logvar = self.va3_acc_encoder(va3_acc)
			mu = torch.cat((mu, va3_acc_mu.unsqueeze(0)), dim=0)
			logvar = torch.cat((logvar, va3_acc_logvar.unsqueeze(0)), dim=1)

		if va3_sca is not None:
			va3_sca_mu, va3_sca_logvar = self.va3_sca_encoder(va3_sca)
			mu = torch.cat((mu, va3_sca_mu.unsqueeze(0)), dim=0)
			logvar = torch.cat((logvar, va3_sca_logvar.unsqueeze(0)), dim=0)

		mu, logvar = self.experts(mu, logvar)
		return mu, logvar


class CoordEncoder(nn.Module):
	def __init__(self, n_latents,  n_coords):
		super(CoordEncoder, self).__init__()
		# what dimension should I use?!
		self.fc1 = nn.Embedding(n_coords, 512)
		self.fc2 = nn.Linear(512, 512)
		self.fc31 = nn.Linear(512, n_latents)
		self.fc32 = nn.Linear(512, n_latents)
		self.swish = Swish()

	def forward(self, x):
		h = self.swish(self.fc1(x))
		h = self.swish(self.fc2(h))
		return self.fc31(h), self.fc32(h)


class CoordDecoder(nn.Module):
	def __init__(self, n_latents,  n_coords):
		super(CoordDecoder, self).__init__()
		# what dimension should I use?!
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
