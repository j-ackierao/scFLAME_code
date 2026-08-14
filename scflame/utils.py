"""
scflame.utils
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------
# NB / Gaussian math
# ------------------------------------------------------------------

def log_nb_pmf(x: torch.Tensor, mu: torch.Tensor, r: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Log-pmf of the negative binomial distribution (mean-dispersion parameterisation)."""
    mu = mu.clamp(min=eps)
    r = r.clamp(min=eps)
    return (
        torch.lgamma(x + r) - torch.lgamma(r) - torch.lgamma(x + 1.0)
        + r * (torch.log(r) - torch.log(r + mu))
        + x * (torch.log(mu) - torch.log(r + mu))
    )


def sample_diag_gaussian(m: torch.Tensor, log_s: torch.Tensor, mc: int = 1) -> torch.Tensor:
    """Reparameterised samples from N(m, diag(exp(log_s))^2), shape (mc, *m.shape)."""
    eps = torch.randn(mc, *m.shape, device=m.device)
    return m.unsqueeze(0) + eps * torch.exp(log_s).unsqueeze(0)


def kl_diag_gaussian_to_std_normal(m: torch.Tensor, log_s: torch.Tensor) -> torch.Tensor:
    """KL(N(m, diag(exp(log_s))^2) || N(0, I)), summed over the latent dimension."""
    var = torch.exp(2.0 * log_s)
    return 0.5 * torch.sum(var + m * m - 1.0 - 2.0 * log_s, dim=-1)


# ------------------------------------------------------------------
# EII Gaussian mixture (equal, isotropic, identical covariance) 
# ------------------------------------------------------------------

class EII_GMM:
    """Gaussian mixture model with a single shared isotropic covariance (EII in mclust notation)."""

    def __init__(self, n_components: int = 3, n_init: int = 10, max_iter: int = 100,
                 tol: float = 1e-4, random_state: Optional[int] = None):
        self.n_components = n_components
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.means_ = self.sigma2_ = self.weights_ = self.lower_bound_ = None

    def _initialize_parameters(self, X: np.ndarray):
        N, D = X.shape
        K = self.n_components
        km = KMeans(n_clusters=K, n_init=10, random_state=self.random_state)
        labels = km.fit_predict(X)
        means = km.cluster_centers_
        weights = np.bincount(labels, minlength=K) / N
        sigma2 = sum(
            np.sum((X[labels == k] - means[k]) ** 2)
            for k in range(K) if (labels == k).any()
        )
        return means, max(sigma2 / (N * D), 1e-6), weights

    def _e_step(self, X: np.ndarray, means: np.ndarray, sigma2: float, weights: np.ndarray):
        N, D = X.shape
        K = self.n_components
        log_prob = np.zeros((N, K))
        for k in range(K):
            log_prob[:, k] = (
                -0.5 * D * np.log(2 * np.pi * sigma2)
                - np.sum((X - means[k]) ** 2, axis=1) / (2 * sigma2)
            )
        log_prob += np.log(weights + 1e-10)
        log_norm = logsumexp(log_prob, axis=1, keepdims=True)
        return np.exp(log_prob - log_norm), np.sum(log_norm)

    def _m_step(self, X: np.ndarray, resp: np.ndarray):
        N, D = X.shape
        K = self.n_components
        N_k = resp.sum(axis=0)
        weights = N_k / N
        means = np.array([(resp[:, k:k + 1] * X).sum(0) / N_k[k] for k in range(K)])
        sigma2 = sum(
            np.sum(resp[:, k] * np.sum((X - means[k]) ** 2, axis=1))
            for k in range(K)
        )
        return means, max(sigma2 / (N * D), 1e-6), weights

    def fit(self, X) -> "EII_GMM":
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        best_ll, best_params = -np.inf, None
        for i in range(self.n_init):
            seed = (self.random_state + i) if self.random_state is not None else None
            np.random.seed(seed)
            means, sigma2, weights = self._initialize_parameters(X)
            ll = -np.inf
            for _ in range(self.max_iter):
                resp, new_ll = self._e_step(X, means, sigma2, weights)
                if abs(new_ll - ll) < self.tol:
                    break
                ll = new_ll
                means, sigma2, weights = self._m_step(X, resp)
            if ll > best_ll:
                best_ll, best_params = ll, (means, sigma2, weights)
        self.means_, self.sigma2_, self.weights_ = best_params
        self.lower_bound_ = best_ll
        self.covariances_ = np.full(self.n_components, self.sigma2_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        resp, _ = self._e_step(X, self.means_, self.sigma2_, self.weights_)
        return resp

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ------------------------------------------------------------------
# Clustering evaluation
# ------------------------------------------------------------------

def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate clustering accuracy via Hungarian matching."""
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximise
    return cm[row_ind, col_ind].sum() / cm.sum()


# ------------------------------------------------------------------
# Marker-gene scoring
# ------------------------------------------------------------------

def gene_scores(L: np.ndarray, nu_k: np.ndarray, pi_k: np.ndarray, cluster_k: int) -> np.ndarray:
    """Per-gene scaled log-fold-change score of cluster `cluster_k` vs the population average."""
    nu_bar = (pi_k[:, None] * nu_k).sum(0)
    raw_scores = L @ (nu_k[cluster_k] - nu_bar)
    # scale by total loading magnitude to avoid over-weighting globally high-abundance genes
    scale_factor = np.abs(L).sum(axis=1) + 1e-5
    return raw_scores / scale_factor


def pairwise_gene_scores(L: np.ndarray, nu_k: np.ndarray, k: int, j: int, scale: bool = True) -> np.ndarray:
    """Pairwise log-fold-change score between clusters k and j: s_kj = L(nu_k - nu_j)."""
    raw_scores = L @ (nu_k[k] - nu_k[j])
    if scale:
        scale_factor = np.abs(L).sum(axis=1) + 1e-5
        return raw_scores / scale_factor
    return raw_scores


def get_top_genes_per_cluster(L, nu_k, pi_k, gene_names, top_n=5) -> dict:
    """Top `top_n` marker genes (by |score|) for each cluster, as (gene, score) pairs."""
    L_np = L.cpu().numpy() if torch.is_tensor(L) else L
    nu_np = nu_k.cpu().numpy() if torch.is_tensor(nu_k) else nu_k
    pi_np = pi_k.cpu().numpy() if torch.is_tensor(pi_k) else pi_k

    K = nu_np.shape[0]
    top_genes = {}
    for k in range(K):
        scores = gene_scores(L_np, nu_np, pi_np, k)
        abs_order = np.argsort(-np.abs(scores))[:top_n]
        top_genes[k] = [(gene_names[i], scores[i]) for i in abs_order]

    return top_genes

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_dataset(data_dir: str, n_hvgs: int, sanitize_gene_names: bool = True):
    """Load counts + labels, drop rare cell types, subset to top HVGs, and
    load matching dispersions / library sizes."""
    df = pd.read_csv(os.path.join(data_dir, "counts.csv"), index_col=0)

    log_df = np.log1p(df.values)
    top_idx = np.argsort(log_df.var(axis=0))[-n_hvgs:]
    df_hv = df.iloc[:, top_idx]
    gene_names = df_hv.columns.tolist()

    X = torch.tensor(df_hv.values.astype(np.float32), dtype=torch.float32).to(DEVICE)

    clusters_df = pd.read_csv(os.path.join(data_dir, "clusters.csv")).set_index("cell_id")
    clusters_df = clusters_df.loc[df_hv.index]
    c_true = clusters_df["celltype"].astype(int).to_numpy()

    dispersions_csv = pd.read_csv(os.path.join(data_dir, "dispersion.csv"))
    if sanitize_gene_names:
        dispersions_csv["gene"] = dispersions_csv["gene"].str.replace("-", ".", regex=False)
        dispersions_csv["gene"] = dispersions_csv["gene"].str.replace("+", ".", regex=False)
    dispersions = dispersions_csv.set_index("gene").loc[gene_names]["dispersion"].values

    lib_sizes_csv = pd.read_csv(os.path.join(data_dir, "library_sizes.csv")).set_index("cell_id")
    C = lib_sizes_csv.loc[df_hv.index]["tmm_lib_size"].values

    return X, gene_names, c_true, dispersions, C

# ============================================================
# Synthetic data generator (useful for unit tests)
# ============================================================

def scflame_synthetic_separated(N=800, D=100, K=4, q=3, r_true=5.0,
                                 cluster_separation=2.0, seed=42):
    """Simulate count data from the scFLAME generative model with well-separated clusters."""
    torch.manual_seed(seed)

    pi_true = torch.softmax(torch.randn(K), dim=0)
    L_true = 0.8 * torch.randn(D, q)
    beta_true = torch.randn(D) * 0.3
    cluster_means = torch.randn(K, q) * cluster_separation

    c_true = torch.multinomial(pi_true, N, replacement=True)
    zs_true = torch.zeros(N, q)
    for i in range(N):
        k = c_true[i].item()
        zs_true[i] = cluster_means[k] + torch.randn(q) * 0.8

    X = torch.zeros(N, D)
    for i in range(N):
        z = zs_true[i]
        logmu = beta_true + L_true @ z
        mu = torch.exp(logmu).clamp(min=1e-8)
        gamma_shape = r_true
        gamma_rate = r_true / mu
        lam = torch.distributions.Gamma(gamma_shape, gamma_rate).sample()
        X[i] = torch.poisson(lam)

    return X, c_true, zs_true, {
        "pi": pi_true, "L": L_true, "beta": beta_true,
        "cluster_means": cluster_means, "r": r_true,
    }
