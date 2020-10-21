import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class FeatureEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(FeatureEncoder, self).__init__()
        self.fc1 = nn.Linear(18, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, latent_dim)
        self.swish = Swish()

    def forward(self, x):
        h = self.swish(self.fc1(x))
        # or h = F.relu(self.fc1(x))
        h = self.swish(self.fc2(h))
        return self.fc3(h)


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)