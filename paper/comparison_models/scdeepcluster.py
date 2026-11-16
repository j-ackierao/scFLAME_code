"""
scdeepcluster.py

scDeepCluster model, proposed in:
"Clustering single-cell RNA-seq data with a model-based deep learning approach"
- Tian et al., 2019
Code adapted from the original GitHub repository (https://github.com/ttgump/scDeepCluster_pytorch)

Optional arguments (defaults reproduce the real-data benchmark):
  - scDeepCluster(dtype=torch.float32, mean_max=1e4): the simulation benchmark
    uses dtype=torch.float64 and mean_max=1e6.
  - fit(save_ckpt=False): checkpoint files are not written unless True.

Note: pretrain_autoencoder takes an optional `clip_grad` argument. When None (default)
it behaves exactly as the original. For the Segerstolpe dataset the benchmark uses
clip_grad=5.0 (gradient-norm clipping during pretraining) due to the low sparsity in SmartSeq2.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math, os
import pandas as pd

from sklearn.cluster import KMeans
from sklearn import metrics

import pylab as plt
import seaborn as sns
from scipy import sparse

import scanpy as sc

import scipy as sp
from sklearn.model_selection import train_test_split


def read_dataset(adata, transpose=False, test_split=False, copy=False):

    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError

    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error

    if adata.X.size < 50e6: # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if transpose: adata = adata.transpose()

    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['DCA_split'] = spl.values
    else:
        adata.obs['DCA_split'] = 'train'

    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')
    print('### Autoencoder: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata


def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size

def geneSelection(data, threshold=0, atleast=10, 
                  yoffset=.02, xoffset=5, decay=1.5, n=None, 
                  plot=True, markers=None, genes=None, figsize=(6,3.5),
                  markeroffsets=None, labelsize=10, alpha=1, verbose=1):
    
    if sparse.issparse(data):
        zeroRate = 1 - np.squeeze(np.array((data>threshold).mean(axis=0)))
        A = data.multiply(data>threshold)
        A.data = np.log2(A.data)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        meanExpr[detected] = np.squeeze(np.array(A[:,detected].mean(axis=0))) / (1-zeroRate[detected])
    else:
        zeroRate = 1 - np.mean(data>threshold, axis=0)
        meanExpr = np.zeros_like(zeroRate) * np.nan
        detected = zeroRate < 1
        mask = data[:,detected]>threshold
        logs = np.zeros_like(data[:,detected]) * np.nan
        logs[mask] = np.log2(data[:,detected][mask])
        meanExpr[detected] = np.nanmean(logs, axis=0)

    lowDetection = np.array(np.sum(data>threshold, axis=0)).squeeze() < atleast
    zeroRate[lowDetection] = np.nan
    meanExpr[lowDetection] = np.nan
            
    if n is not None:
        up = 10
        low = 0
        for t in range(100):
            nonan = ~np.isnan(zeroRate)
            selected = np.zeros_like(zeroRate).astype(bool)
            selected[nonan] = zeroRate[nonan] > np.exp(-decay*(meanExpr[nonan] - xoffset)) + yoffset
            if np.sum(selected) == n:
                break
            elif np.sum(selected) < n:
                up = xoffset
                xoffset = (xoffset + low)/2
            else:
                low = xoffset
                xoffset = (xoffset + up)/2
        if verbose>0:
            print('Chosen offset: {:.2f}'.format(xoffset))
    else:
        nonan = ~np.isnan(zeroRate)
        selected = np.zeros_like(zeroRate).astype(bool)
        selected[nonan] = zeroRate[nonan] > np.exp(-decay*(meanExpr[nonan] - xoffset)) + yoffset
                
    if plot:
        if figsize is not None:
            plt.figure(figsize=figsize)
        plt.ylim([0, 1])
        if threshold>0:
            plt.xlim([np.log2(threshold), np.ceil(np.nanmax(meanExpr))])
        else:
            plt.xlim([0, np.ceil(np.nanmax(meanExpr))])
        x = np.arange(plt.xlim()[0], plt.xlim()[1]+.1,.1)
        y = np.exp(-decay*(x - xoffset)) + yoffset
        if decay==1:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-x+{:.2f})+{:.2f}'.format(np.sum(selected),xoffset, yoffset), 
                     color='k', fontsize=labelsize, transform=plt.gca().transAxes)
        else:
            plt.text(.4, 0.2, '{} genes selected\ny = exp(-{:.1f}*(x-{:.2f}))+{:.2f}'.format(np.sum(selected),decay,xoffset, yoffset), 
                     color='k', fontsize=labelsize, transform=plt.gca().transAxes)

        plt.plot(x, y, color=sns.color_palette()[1], linewidth=2)
        xy = np.concatenate((np.concatenate((x[:,None],y[:,None]),axis=1), np.array([[plt.xlim()[1], 1]])))
        t = plt.matplotlib.patches.Polygon(xy, color=sns.color_palette()[1], alpha=.4)
        plt.gca().add_patch(t)
        
        plt.scatter(meanExpr, zeroRate, s=1, alpha=alpha, rasterized=True)
        if threshold==0:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of zero expression')
        else:
            plt.xlabel('Mean log2 nonzero expression')
            plt.ylabel('Frequency of near-zero expression')
        plt.tight_layout()
        
        if markers is not None and genes is not None:
            if markeroffsets is None:
                markeroffsets = [(0, 0) for g in markers]
            for num,g in enumerate(markers):
                i = np.where(genes==g)[0]
                plt.scatter(meanExpr[i], zeroRate[i], s=10, color='k')
                dx, dy = markeroffsets[num]
                plt.text(meanExpr[i]+dx+.1, zeroRate[i]+dy, g, color='k', fontsize=labelsize)
    
    return selected

# ============================================================
# Network building blocks
# ============================================================

def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)

def euclidean_dist(x, y):
    return torch.sum(torch.square(x - y), dim=1)

# ============================================================
# ZINB LOSS
# ============================================================

class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        
        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge
        
        result = torch.mean(result)
        return result


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
    
    def forward(self, x):
        if self.training:
            x = x + self.sigma * torch.randn_like(x)
        return x


class MeanAct(nn.Module):
    def __init__(self, max_value=1e4):
        super(MeanAct, self).__init__()
        self.max_value = max_value

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=self.max_value)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)

# ============================================================
# scDeepCluster
# ============================================================

class scDeepCluster(nn.Module):
    def __init__(self, input_dim, z_dim, encodeLayer=[], decodeLayer=[], 
            activation="relu", sigma=1., alpha=1., gamma=1., device="cuda",
            dtype=torch.float32, mean_max=1e4):
        super(scDeepCluster, self).__init__()
        self.dtype = dtype
        torch.set_default_dtype(dtype)
        self.z_dim = z_dim
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.encoder = buildNetwork([input_dim]+encodeLayer, type="encode", activation=activation)
        self.decoder = buildNetwork([z_dim]+decodeLayer, type="decode", activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct(max_value=mean_max))
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())

        self.zinb_loss = ZINBLoss().to(self.device)
        self.to(device)
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q
    
    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def forwardAE(self, x):
        h = self.encoder(x+torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        return z0, _mean, _disp, _pi

    def forward(self, x):
        h = self.encoder(x+torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        q = self.soft_assign(z0)
        return z0, q, _mean, _disp, _pi

    def encodeBatch(self, X, batch_size=256):
        self.eval()
        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs = Variable(xbatch).to(self.device)
            z, _, _, _= self.forwardAE(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded.to(self.device)

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
        kldloss = kld(p, q)
        return self.gamma*kldloss

    def pretrain_autoencoder(self, X, X_raw, size_factor, batch_size=256, lr=0.001, epochs=400,
                             ae_save=True, ae_weights='AE_weights.pth.tar', clip_grad=None):
        """
        clip_grad: if not None, gradient-norm clipping (max_norm=clip_grad) is applied
                   after each backward pass. Used (5.0) for the Segerstolpe dataset only;
                   None reproduces the original behaviour.
        """
        self.train()
        dataset = TensorDataset(X.detach().cpu(), X_raw.detach().cpu(), size_factor.detach().cpu())
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        for epoch in range(epochs):
            loss_val = 0
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch).to(self.device)
                x_raw_tensor = Variable(x_raw_batch).to(self.device)
                sf_tensor = Variable(sf_batch).to(self.device)
                _, mean_tensor, disp_tensor, pi_tensor = self.forwardAE(x_tensor)
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=sf_tensor)
                optimizer.zero_grad()
                loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=clip_grad)
                optimizer.step()
                loss_val += loss.item() * len(x_batch)
            print('Pretrain epoch %3d, ZINB loss: %.8f' % (epoch+1, loss_val/X.shape[0]))

        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit(self, X, X_raw, size_factor, n_clusters, init_centroid=None, y=None, y_pred_init=None, lr=1., batch_size=256, 
            num_epochs=100, update_interval=1, tol=1e-3, save_dir="", save_ckpt=False):
        '''X: tensor data
        save_ckpt: if True, writes FTcheckpoint_*.pth.tar files to save_dir (original
                   behaviour). Disabled by default for the benchmark.'''
        self.train()
        print("Clustering stage")
        X = torch.tensor(X, dtype=self.dtype)
        X_raw = torch.tensor(X_raw, dtype=self.dtype)
        size_factor = torch.tensor(size_factor, dtype=self.dtype)
        self.mu = Parameter(torch.Tensor(n_clusters, self.z_dim).to(self.device))
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)

        print("Initializing cluster centers with kmeans.")
        if init_centroid is None:
            kmeans = KMeans(n_clusters, n_init=20)
            data = self.encodeBatch(X)
            self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
            self.y_pred_last = self.y_pred
            self.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=self.dtype))
        else:
            self.mu.data.copy_(torch.tensor(init_centroid, dtype=self.dtype))
            self.y_pred = y_pred_init
            self.y_pred_last = self.y_pred
        if y is not None:
            acc = np.round(cluster_acc(y, self.y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print('Initializing k-means: NMI= %.4f, ARI= %.4f' % (nmi, ari))

        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))

        final_acc, final_nmi, final_ari, final_epoch = 0, 0, 0, 0


        for epoch in range(num_epochs):
            if epoch%update_interval == 0:
                # update the targe distribution p
                latent = self.encodeBatch(X.to(self.device))
                q = self.soft_assign(latent)
                p = self.target_distribution(q).data

                # evalute the clustering performance
                self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()

                if y is not None:
                    final_acc = acc = np.round(cluster_acc(y, self.y_pred), 5)
                    final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    final_ari = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    print('Clustering   %d: NMI= %.4f, ARI= %.4f' % (epoch+1, nmi, ari))

                # save current model
                if save_ckpt and ((epoch>0 and delta_label < tol) or epoch%10 == 0):
                    self.save_checkpoint({'epoch': epoch+1,
                            'state_dict': self.state_dict(),
                            'mu': self.mu,
                            'y_pred': self.y_pred,
                            'y_pred_last': self.y_pred_last,
                            'y': y
                            }, epoch+1, filename=save_dir)

                # check stop criterion
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred
                if epoch>0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break


            # train 1 epoch for clustering loss
            train_loss = 0.0
            recon_loss_val = 0.0
            cluster_loss_val = 0.0
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                xrawbatch = X_raw[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                sfbatch = size_factor[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                optimizer.zero_grad()
                inputs = Variable(xbatch).to(self.device)
                rawinputs = Variable(xrawbatch).to(self.device)
                sfinputs = Variable(sfbatch).to(self.device)
                target = Variable(pbatch).to(self.device)

                zbatch, qbatch, meanbatch, dispbatch, pibatch = self.forward(inputs)

                cluster_loss = self.cluster_loss(target, qbatch)
                recon_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs)

                loss = cluster_loss*self.gamma + recon_loss
                loss.backward()
                optimizer.step()
                cluster_loss_val += cluster_loss.item() * len(inputs)
                recon_loss_val += recon_loss.item() * len(inputs)
                train_loss += loss.item() * len(inputs)

            print("Epoch %3d: Total: %.8f Clustering Loss: %.8f ZINB Loss: %.8f" % (
                epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num))

        return self.y_pred, final_acc, final_nmi, final_ari, final_epoch