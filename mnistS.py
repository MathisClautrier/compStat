from models import SVAE

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from collections import defaultdict

from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='training')
parser.add_argument('--z-dim', type=int, default=10)
parser.add_argument('--GPU',  action='store_true')



args = parser.parse_args()

train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True,
    transform=transforms.ToTensor()), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, download=True,
    transform=transforms.ToTensor()), batch_size=64)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'            
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

model = SVAE(28*28,128, args.z_dim)
optimizer = optim.Adam(modelN.parameters(), lr=1e-3)
ES = EarlyStopping()

for i in tqdm(range(1000)):
    model.train()
    for X,y in train_loader:
        optimizer.zero_grad()
        X= (X> torch.distributions.Uniform(0, 1).sample(X.shape)).float()
        X = X.reshape(-1,784)
        (z_mean, z_var), (q_z, p_z), z, x_ = model(X)
        loss = nn.BCEWithLogitsLoss(reduction='none')(x_, X).sum(-1).mean()
        loss+=torch.distributions.kl.kl_divergence(q_z, p_z).mean()

        loss.backward(retain_graph=True)
        optimizer.step()
    model.eval()
    val_loss=0
    for X, y_mb in test_loader:
        X= (X> torch.distributions.Uniform(0, 1).sample(X.shape)).float()
        X = X.reshape(-1,784)
        (z_mean, z_var), (q_z, p_z), z, x_ = model(X)
        val_loss += nn.BCEWithLogitsLoss(reduction='none')(x_, X).sum(-1).mean().item()
        val_loss+=torch.distributions.kl.kl_divergence(q_z, p_z).mean().item()
    ES(val_loss)
    if ES.early_stop:
        print("Early stopping")
        break

torch.save(model.state_dict(),"SVAE"+str(args.z_dim))