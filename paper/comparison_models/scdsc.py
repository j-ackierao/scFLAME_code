"""
scdsc.py

scDSC model, proposed in:
"Deep structural clustering for single-cell RNA-seq data 
jointly through autoencoder and graph neural network" - Gan et al., 2022
Code adapted from the original GitHub repository (https://github.com/DHUDBlab/scDSC)
Contains the model architecture (AE + GCN + ZINB) to be imported.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pairwise_distances
from sklearn.preprocessing import normalize
from torch.optim import Adam

def eva(y_true, y_pred, epoch=0):
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    print(f"[Epoch {epoch}] ARI={ari:.4f} NMI={nmi:.4f}")
    return ari, nmi

# ============================================================
# Graph construction (EXACT scDSC logic)
# ============================================================

import scipy.sparse as sp

def construct_and_load_graph(features, topk=10, method="ncos"):
    features = features.copy()
    
    if method == 'heat':
        dist = -0.5 * pairwise_distances(features, metric='manhattan') ** 2
        dist = np.exp(dist)
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)
    elif method == 'p':
        y = features.T - np.mean(features.T)
        features = features - np.mean(features)
        dist = np.dot(features, features.T) / (
            np.linalg.norm(features) * np.linalg.norm(y)
        )

    n = dist.shape[0]
    inds = []
    for i in range(n):
        ind = np.argpartition(dist[i], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    # Build edge list
    rows, cols = [], []
    for i, v in enumerate(inds):
        for vv in v:
            if vv != i:
                rows.append(i)
                cols.append(vv)

    # Build sparse adjacency
    adj = sp.coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=(n, n), dtype=np.float32
    )

    # Symmetrise
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # Self-loops
    adj = adj + sp.eye(n)

    # Row-normalise
    adj = normalize(adj)

    # Convert to torch sparse tensor
    adj = adj.tocoo()
    indices = torch.from_numpy(np.vstack((adj.row, adj.col)).astype(np.int64))
    values = torch.from_numpy(adj.data)
    shape = torch.Size(adj.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

# ============================================================
# AE
# ============================================================

class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3,
                 n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):

        super(AE, self).__init__()

        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.BN1 = nn.BatchNorm1d(n_enc_1)

        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.BN2 = nn.BatchNorm1d(n_enc_2)

        self.enc_3 = nn.Linear(n_enc_2, n_enc_3)
        self.BN3 = nn.BatchNorm1d(n_enc_3)

        self.z_layer = nn.Linear(n_enc_3, n_z)

        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.BN4 = nn.BatchNorm1d(n_dec_1)

        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.BN5 = nn.BatchNorm1d(n_dec_2)

        self.dec_3 = nn.Linear(n_dec_2, n_dec_3)
        self.BN6 = nn.BatchNorm1d(n_dec_3)

        self.x_bar_layer = nn.Linear(n_dec_3, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.BN1(self.enc_1(x)))
        enc_h2 = F.relu(self.BN2(self.enc_2(enc_h1)))
        enc_h3 = F.relu(self.BN3(self.enc_3(enc_h2)))

        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.BN4(self.dec_1(z)))
        dec_h2 = F.relu(self.BN5(self.dec_2(dec_h1)))
        dec_h3 = F.relu(self.BN6(self.dec_3(dec_h2)))

        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z, dec_h3

# ============================================================
# GNN Layer
# ============================================================

class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)
        # When passing through the network layer, the input and output variances are the same
        # including forward propagation and backward propagation

    def forward(self, features, adj, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adj, support)
        if active:
            output = F.relu(output)
        return output

# ============================================================
# ZINB LOSS
# ============================================================

class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0,    ridge_lambda=0.0):
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
    def __init__(self):
        super(MeanAct, self).__init__()
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()
    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)

# ============================================================
# SDCN
# ============================================================

class SDCN(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3,
                 n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1):

        super(SDCN, self).__init__()

        self.ae = AE(n_enc_1, n_enc_2, n_enc_3,
                     n_dec_1, n_dec_2, n_dec_3,
                     n_input, n_z)

        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2, n_enc_3)
        self.gnn_4 = GNNLayer(n_enc_3, n_z)
        self.gnn_5 = GNNLayer(n_z, n_clusters)

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        self._dec_mean = nn.Sequential(nn.Linear(n_dec_3, n_input), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(n_dec_3, n_input), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(n_dec_3, n_input), nn.Sigmoid())

        self.v = v
        self.zinb_loss = ZINBLoss()

    def forward(self, x, adj):

        x_bar, tra1, tra2, tra3, z, dec_h3 = self.ae(x)

        sigma = 0.5

        h = self.gnn_1(x, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * tra2, adj)
        h = self.gnn_4((1 - sigma) * h + sigma * tra3, adj)
        h = self.gnn_5((1 - sigma) * h + sigma * z, adj, active=False)

        predict = F.softmax(h, dim=1)

        _mean = self._dec_mean(dec_h3)
        _disp = self._dec_disp(dec_h3)
        _pi = self._dec_pi(dec_h3)

        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, _mean, _disp, _pi, self.zinb_loss

# ============================================================
# Target distribution
# ============================================================

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

#
# Pretrain
#

def pretrain_ae(ae, X, epochs=200, batch_size=256, lr=1e-3):
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    ae.train()
    for epoch in range(epochs):
        for (batch,) in loader:
            x_bar, _, _, _, _, _ = ae(batch)
            loss = F.mse_loss(x_bar, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            with torch.no_grad():
                x_bar, _, _, _, _, _ = ae(X)
                loss = F.mse_loss(x_bar, X)
                print(f"[Pretrain epoch {epoch}] loss: {loss.item():.4f}")
    ae.eval()

# ============================================================
# TRAIN
# ============================================================

def train(X, y, adj, args, device="cuda"):

    model = SDCN(*args["arch"], args["n_input"], args["n_z"], args["n_clusters"]).to(device)
    opt = Adam(model.parameters(), lr=args["lr"])

    X = torch.tensor(X, dtype=torch.float32).to(device)
    adj = torch.tensor(adj, dtype=torch.float32).to(device)

    # init kmeans
    with torch.no_grad():
        _, _, _, z, _, _, _, _ = model(X, adj)

    km = KMeans(n_clusters=args["n_clusters"], n_init=20)
    y_pred = km.fit_predict(z.cpu().numpy())
    model.cluster_layer.data = torch.tensor(km.cluster_centers_).to(device)

    p = None

    for epoch in range(args["epochs"]):

        x_bar, q, pred, z, mean, disp, pi, zinb_loss = model(X, adj)

        if epoch % 1 == 0:
            tmp_q = q.detach()
            p = target_distribution(tmp_q)

            eva(y, tmp_q.argmax(1).cpu().numpy(), epoch)

        pred = torch.clamp(pred, 1e-8, 1.0)
        loss_kl = F.kl_div(
            torch.log(pred),
            p,
            reduction="batchmean"
        )
        loss_recon = F.mse_loss(x_bar, X)

        loss = loss_kl + loss_recon

        opt.zero_grad()
        loss.backward()
        opt.step()

    return q.argmax(1).cpu().numpy(), z.cpu().detach().numpy()

def target_distribution(q):
    # Pij
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()
