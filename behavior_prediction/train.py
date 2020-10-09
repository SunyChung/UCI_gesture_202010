# base code : https://github.com/mhw32/multimodal-vae-public

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from mvae_model import MVAE


# def elbo_loss(recon_):