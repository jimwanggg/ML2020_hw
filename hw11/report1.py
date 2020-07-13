import torch
from torch import optim
from torch.autograd import Variable
import torchvision
import os
import matplotlib.pyplot as plt
from model import Generator, Discriminator
from utils import same_seeds
from data import get_dataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import sys

if __name__ == "__main__":
    # hyperparameters 
    batch_size = 64
    z_dim = 100
    lr = 1e-4
    n_epoch = 30
    model_name = sys.argv[1]
    file_name = sys.argv[2]
    same_seeds(0)

    # model
    G = Generator(in_dim=z_dim).cuda()
    G.load_state_dict(torch.load(model_name))
    G.eval()
    z_sample = Variable(torch.randn(100, z_dim)).cuda()
    f_imgs_sample = (G(z_sample).data + 1) / 2.0
    torchvision.utils.save_image(f_imgs_sample, file_name, nrow=10)
