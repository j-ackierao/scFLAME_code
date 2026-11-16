"""
vade.py

VaDE model, proposed in:
"Variational Deep Embedding: An Unsupervised and Generative Approach to Clustering"
- Jiang et al., 2017
Code adapted from the PyTorch implementation (https://github.com/GuHY777/VaDE-pytorch)
Contains the model architecture (VAE with GMM prior) to be imported.

Changes from the repository code: the decoder's final Sigmoid is removed (inputs are
log-normalised counts, not [0, 1] pixels), input_dim/hid_dim are taken from args,
GMM initialisation works on CPU as well as GPU, and the pretrained weights path is
passed in explicitly.
"""

import os
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    acc = sum(w[i, j] for i, j in zip(row_ind, col_ind)) * 1.0 / Y_pred.size
    return acc, w

def block(in_c, out_c):
    layers = [nn.Linear(in_c, out_c), nn.ReLU(True)]
    return layers

# ============================================================
# Encoder / Decoder
# ============================================================

class Encoder(nn.Module):
    def __init__(self, input_dim=784, inter_dims=[500, 500, 2000], hid_dim=10):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            *block(input_dim, inter_dims[0]),
            *block(inter_dims[0], inter_dims[1]),
            *block(inter_dims[1], inter_dims[2]),
        )
        self.mu_l = nn.Linear(inter_dims[-1], hid_dim)
        self.log_sigma2_l = nn.Linear(inter_dims[-1], hid_dim)

    def forward(self, x):
        e = self.encoder(x)
        mu = self.mu_l(e)
        log_sigma2 = self.log_sigma2_l(e)
        return mu, log_sigma2

class Decoder(nn.Module):
    def __init__(self, input_dim=784, inter_dims=[500, 500, 2000], hid_dim=10):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            *block(hid_dim, inter_dims[-1]),
            *block(inter_dims[-1], inter_dims[-2]),
            *block(inter_dims[-2], inter_dims[-3]),
            nn.Linear(inter_dims[-3], input_dim)
            # Final Sigmoid from the original repository removed
        )

    def forward(self, z):
        return self.decoder(z)

# ============================================================
# VaDE
# ============================================================

class VaDE(nn.Module):
    def __init__(self, args):
        super(VaDE, self).__init__()
        self.encoder = Encoder(input_dim=args.input_dim, inter_dims=[500, 500, 2000], hid_dim=args.hid_dim)
        self.decoder = Decoder(input_dim=args.input_dim, inter_dims=[500, 500, 2000], hid_dim=args.hid_dim)

        self.pi_ = nn.Parameter(torch.FloatTensor(args.nClusters,).fill_(1) / args.nClusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.FloatTensor(args.nClusters, args.hid_dim).fill_(0), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.FloatTensor(args.nClusters, args.hid_dim).fill_(0), requires_grad=True)

        self.args = args

    def pre_train(self, dataloader, pretrain_path, pre_epoch=30):
        if not os.path.exists(pretrain_path):
            Loss = nn.MSELoss()
            opti = Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()))

            print('Pretraining......')
            epoch_bar = tqdm(range(pre_epoch))
            for _ in epoch_bar:
                L = 0
                for x, y in dataloader:
                    if self.args.cuda:
                        x = x.cuda()
                    z, _ = self.encoder(x)
                    x_ = self.decoder(z)
                    loss = Loss(x, x_)
                    L += loss.detach().cpu().numpy()
                    opti.zero_grad()
                    loss.backward()
                    opti.step()
                epoch_bar.write('L2={:.4f}'.format(L / len(dataloader)))

            self.encoder.log_sigma2_l.load_state_dict(self.encoder.mu_l.state_dict())

            Z, Y = [], []
            with torch.no_grad():
                for x, y in dataloader:
                    if self.args.cuda:
                        x = x.cuda()
                    z1, z2 = self.encoder(x)
                    assert F.mse_loss(z1, z2) == 0
                    Z.append(z1)
                    Y.append(y)

            Z = torch.cat(Z, 0).detach().cpu().numpy()
            Y = torch.cat(Y, 0).detach().numpy()

            gmm = GaussianMixture(n_components=self.args.nClusters, covariance_type='diag')
            pre = gmm.fit_predict(Z)
            print('GMM Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

            self.pi_.data = torch.from_numpy(gmm.weights_).cuda().float() if self.args.cuda else torch.from_numpy(gmm.weights_).float()
            self.mu_c.data = torch.from_numpy(gmm.means_).cuda().float() if self.args.cuda else torch.from_numpy(gmm.means_).float()
            self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float() if self.args.cuda else torch.from_numpy(gmm.covariances_).float())

            torch.save(self.state_dict(), pretrain_path)
        else:
            self.load_state_dict(torch.load(pretrain_path, map_location="cuda" if self.args.cuda else "cpu"))

    def predict(self, x):
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c))
        yita = yita_c.detach().cpu().numpy()
        return np.argmax(yita, axis=1)

    def ELBO_Loss(self, x, L=1):
        det = 1e-10
        L_rec = 0
        z_mu, z_sigma2_log = self.encoder(x)

        for l in range(L):
            z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
            x_pro = self.decoder(z)
            L_rec += F.mse_loss(x_pro, x)
        L_rec /= L

        Loss = L_rec * x.size(1)

        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c

        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        yita_c = torch.exp(torch.log(pi.unsqueeze(0)) + self.gaussian_pdfs_log(z, mu_c, log_sigma2_c)) + det
        yita_c = yita_c / (yita_c.sum(1).view(-1, 1))

        Loss += 0.5 * torch.mean(torch.sum(yita_c * torch.sum(
            log_sigma2_c.unsqueeze(0) +
            torch.exp(z_sigma2_log.unsqueeze(1) - log_sigma2_c.unsqueeze(0)) +
            (z_mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2) / torch.exp(log_sigma2_c.unsqueeze(0)),
            2), 1))

        Loss -= torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / yita_c), 1)) + \
                0.5 * torch.mean(torch.sum(1 + z_sigma2_log, 1))

        return Loss

    def gaussian_pdfs_log(self, x, mus, log_sigma2s):
        G = []
        for c in range(self.args.nClusters):
            G.append(self.gaussian_pdf_log(x, mus[c:c+1, :], log_sigma2s[c:c+1, :]).view(-1, 1))
        return torch.cat(G, 1)

    @staticmethod
    def gaussian_pdf_log(x, mu, log_sigma2):
        return -0.5 * (torch.sum(np.log(np.pi * 2) + log_sigma2 + (x - mu).pow(2) / torch.exp(log_sigma2), 1))

    def get_latent(self, x):
        """Return the posterior mean z_mu as the latent representation."""
        z_mu, z_sigma2_log = self.encoder(x)
        return z_mu.detach().cpu().numpy()