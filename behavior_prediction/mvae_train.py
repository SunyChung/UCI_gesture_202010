# base code : https://github.com/mhw32/multimodal-vae-public
import os
import sys
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from .mvae_model import MVAE
from .data_loader import load_1d_data


def binary_cross_entropy_with_logit(input, target):
    if not (target.size()) == input.size():
        raise ValueError('target size ({}) must be the same as the input size ({})'
                         .format(target.size(), input.size()))
    return (torch.clamp(input, 0) - input * target
            + torch.log(1 + torch.exp(-torch.abs(input))))


def cross_entropy(input, target, eps=1e-6):
    if not (target.size(0) == input.size(0)):
        raise ValueError('target size ({}) must be the same as the input size ({})'
                         .format(target.size(0), input.size(0)))
    log_input = F.log_softmax(input + eps, dim=1)
    y_onehot = Variable(log_input.data.new(log_input.size()).zero_())
    y_onehot = y_onehot.scatter(1, target.unsqueeze(1), 1)
    loss = y_onehot * log_input
    return -loss


def elbo_loss(recon_features, features, recon_label, label, mu, logvar,
              lambda_feature=1.0, lambda_label=1.0, annealing_factor=1):
    feature_bce, label_bce = 0, 0
    if recon_features is not None and features is not None:
        feature_bce = torch.sum(binary_cross_entropy_with_logit(
            # check the feature dimensions!
            recon_features.view(-1, 8, 18),
            features.view(-1, 8, 18)), dim=1
        )
    if recon_label is not None and label is not None:
        label_bce = torch.sum(cross_entropy(
            # check the feature dimensions!
            recon_label.view(-1, 8, 1),
            label.view(-1, 8, 1)), dim=1
        )
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    ELBO = torch.mean(lambda_feature * feature_bce + lambda_label * label_bce
                      + annealing_factor * KLD)
    return ELBO


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_latents', type=int, default=64,
                        help='size of the latent embedding [default : 64]')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training [default : 16')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train [default : 200]')
    parser.add_argument('--annealing-epochs', type=int, default=100, metavar='N',
                        help='number of epochs to anneal KL for [default : 100]')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate [default : 1e-3]')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status [default : 10]')
    parser.add_argument('--lambda-feature', type=float, default=1.,
                        help='multiplier for feature reconstruction [default : 1]')
    parser.add_argument('--lambda-label', type=float, default=1.,
                        help='multiplier for label reconstruction [default : 1]')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()

    train_features, train_labels = load_1d_data('raw', 'train')
    N_mini_batches = len(train_features)
    test_features, test_labels = load_1d_data('raw', 'test')

    model = MVAE(args.n_latents)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def train(epoch):
        model.train()
        train_loss_meter = AverageMeter()

        for batch_idx, (feature, label) in enumerate(zip(train_features, train_labels)):
            if epoch < args.annealing_epochs:
                annealing_factor = (float(batch_idx + (epoch - 1) * N_mini_batches + 1) /
                                    float(args.annealing_epochs * N_mini_batches))
            else:
                annealing_factor = 1.0

            if args.cuda:
                feature = feature.cuda()
                label = label.cuda()

            feature = Variable(feature)
            label = Variable(label)
            batch_size = len(feature)

            optimizer.zero_grad()

            recon_feature_1, recon_label_1, mu_1, logvar_1 = model(feature, label)
            recon_feature_2, recon_label_2, mu_2, logvar_2 = model(feature)
            recon_feature_3, recon_label_3, mu_3, logvar_3 = model(label=label)

            joint_loss = elbo_loss(recon_feature_1, feature, recon_label_1, label, mu_1, logvar_1,
                                   lambda_feature=args.lamda_feature, lambda_label=args.lamda_label,
                                   annealing_factor=annealing_factor)
            feature_loss = elbo_loss(recon_feature_2, feature, None, None, mu_2, logvar_2,
                                     lambda_feature=args.lamda_feature, lambda_label=args.lamda_label,
                                     annealing_factor=annealing_factor)
            label_loss = elbo_loss(None, None, recon_label_3, label, mu_3, logvar_3,
                                   lambda_feature=args.lamda_feature, lambda_label=args.lamda_label,
                                   annealing_factor=annealing_factor)
            train_loss = joint_loss + feature_loss + label_loss
            train_loss_meter.update(train_loss.data[0], batch_size)
            train_loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('train epoch : {} [{}/{} ({:.0f}%)]\tLoss : {:.6f}\tAnnealing-Factor : {:.3f}'
                      .format(epoch, batch_idx * len(feature), len(train_features),
                              100. * batch_idx / len(train_features), train_loss_meter.avg, annealing_factor))
        print('----- epoch : {}\tloss : {:.4f} -----'.format(epoch, train_loss_meter.avg))

    def test(epoch):
        model.eval()
        test_loss_meter = AverageMeter()

        for batch_idx, (feature, label) in enumerate(zip(test_features, test_labels)):
            if args.cuda:
                feature = feature.cuda()
                label = label.cuda()

            feature = Variable(feature, volatile=True)
            label = Variable(label, volatile=True)
            batch_size = len(feature)

            recon_feature_1, recon_label_1, mu_1, logvar_1 = model(feature, label)
            recon_feature_2, recon_label_2, mu_2, logvar_2 = model(feature)
            recon_feature_3, recon_label_3, mu_3, logvar_3 = model(label=label)

            joint_loss = elbo_loss(recon_feature_1, feature, recon_label_1, label, mu_1, logvar_1)
            feature_loss = elbo_loss(recon_feature_2, feature, None, None, mu_2, logvar_2)
            label_loss = elbo_loss(None, None, recon_label_3, label, mu_3, logvar_3)
            test_loss = joint_loss + feature_loss + label_loss
            test_loss_meter.update(test_loss.data[0], batch_size)

        print('----- test loss : {:.4f}'.format(test_loss_meter.avg))
        return test_loss_meter.avg