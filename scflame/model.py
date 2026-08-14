"""
scflame.model
=============
Functions to fit the scFLAME model.

train_nb_fa - Warm-up stage, NBFA model.

train_scflame - Training function for scFLAME, the joint NBFA + mixture model.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .utils import DEVICE, log_nb_pmf, sample_diag_gaussian, kl_diag_gaussian_to_std_normal, EII_GMM


# ============================================================
# NBFA warm-up stage
# ============================================================

def train_nb_fa(X, C=None, q=10, mc_samples=5, epochs=300, lr=1e-2, disp=None, verbose=True):
    """Fit a plain NB factor model using VI.

    Args:
        X: (N, D) count matrix.
        C: optional (N,) library-size factors.
        q: latent dimensionality.
        disp: optional fixed per-gene dispersion (array or scalar). If None,
            dispersion is learned jointly with the other global parameters.

    Returns a dict of fitted parameters used to initialise `train_scflame`.
    """
    N, D = X.shape
    if C is None:
        C = torch.ones(N, device=DEVICE)
    else:
        if isinstance(C, np.ndarray):
            C = torch.from_numpy(C)
        C = C.to(DEVICE).float()

    L = nn.Parameter(torch.randn(D, q, device=DEVICE) * 0.1)
    alpha = nn.Parameter(torch.zeros(D, device=DEVICE))
    if disp is None:
        log_phi = nn.Parameter(torch.log(torch.ones(D, device=DEVICE) * 5.0))
    else:
        if isinstance(disp, np.ndarray):
            disp = torch.from_numpy(disp)
        elif isinstance(disp, (float, int)):
            disp = torch.full((D,), float(disp))
        log_phi = torch.log(disp.float()).to(DEVICE)

    m = nn.Parameter(torch.randn(N, q, device=DEVICE) * 0.1)
    log_s = nn.Parameter(torch.full((N, q), -1.0, device=DEVICE))

    global_params = [L, alpha, log_phi] if disp is None else [L, alpha]
    opt_global = optim.Adam(global_params, lr=lr)
    opt_local = optim.Adam([m, log_s], lr=lr * 2)  # faster learning rate for per-cell params

    trace = []
    t0 = time.time()

    for ep in range(epochs):
        opt_global.zero_grad()
        opt_local.zero_grad()
        mc_z = sample_diag_gaussian(m, log_s, mc=mc_samples)
        total_ll = torch.zeros(mc_samples, N, device=DEVICE)
        for s in range(mc_samples):
            logmu = alpha.unsqueeze(0) + mc_z[s] @ L.t() + torch.log(C).unsqueeze(1)
            mu = torch.exp(logmu).clamp(min=1e-8)
            phi = torch.exp(log_phi).unsqueeze(0).clamp(min=0.1, max=100)
            total_ll[s] = log_nb_pmf(X.float(), mu, phi).sum(dim=1)
        elbo = torch.sum(total_ll.mean(0) - kl_diag_gaussian_to_std_normal(m, log_s))
        (-elbo).backward()
        # Gradient clipping to avoid exploding gradients
        clip_params = [L, alpha, log_phi, m, log_s] if disp is None else [L, alpha, m, log_s]
        nn.utils.clip_grad_norm_(clip_params, max_norm=5.0)
        opt_global.step()
        opt_local.step()
        trace.append(elbo.item())
        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            print(f"Epoch {ep:4d} | ELBO {elbo.item():.1f} | ELBO/N {elbo.item() / N:.4f}")

    if verbose:
        print(f"NBFA training time: {time.time() - t0:.2f}s")

    return dict(L=L, alpha=alpha, log_phi=log_phi, m=m, log_s=log_s, trace=trace, C=C)


# ============================================================
# scFLAME -- joint factor analysis + Gaussian-mixture clustering
# ============================================================

def m_step_alpha(r: torch.Tensor, alpha_0: torch.Tensor) -> torch.Tensor:
    """Dirichlet posterior update for the mixture weights."""
    return alpha_0 + r.sum(dim=0)


def compute_elbo_scflame(X, L, gamma, log_phi, A, m, log_s, r, nu_k, omega2_k,
                          alpha, alpha_0, mc_samples=5) -> torch.Tensor:
    """ELBO for scFLAME at the current variational parameters."""
    N, D = X.shape
    K = len(alpha)
    s = torch.exp(log_s)
    E_log_pi = torch.digamma(alpha) - torch.digamma(alpha.sum())
    mc_z = sample_diag_gaussian(m, log_s, mc=mc_samples)
    Lz = torch.einsum("snq,dq->snd", mc_z, L)
    log_lik = torch.zeros(mc_samples, N, device=X.device)

    for s_idx in range(mc_samples):
        logmu = gamma.unsqueeze(0) + Lz[s_idx] + torch.log(A).unsqueeze(1)
        mu = torch.exp(logmu).clamp(min=1e-8)
        phi = torch.exp(log_phi).unsqueeze(0).clamp(min=0.1, max=100)
        log_lik[s_idx] = log_nb_pmf(X, mu, phi).sum(dim=1)

    E_log_p_y = log_lik.mean(0).sum()
    E_log_p_z_given_c = torch.tensor(0.0, device=X.device)

    for k in range(K):
        diff_sq = (m - nu_k[k].unsqueeze(0)) ** 2
        log_p_z_k = -0.5 * torch.sum(
            torch.log(2 * torch.pi * omega2_k[k]) + (diff_sq + s ** 2) / omega2_k[k], dim=1
        )
        E_log_p_z_given_c += torch.sum(r[:, k] * log_p_z_k)

    E_log_p_c = torch.sum(r * E_log_pi.unsqueeze(0))
    log_B_alpha_0 = torch.lgamma(alpha_0).sum() - torch.lgamma(alpha_0.sum())
    E_log_p_pi = -log_B_alpha_0 + torch.sum((alpha_0 - 1.0) * E_log_pi)
    H_q_z = 0.5 * torch.sum(1 + np.log(2 * np.pi) + 2 * log_s)
    H_q_c = -torch.sum(r * torch.log(r + 1e-10))
    log_B_alpha = torch.lgamma(alpha).sum() - torch.lgamma(alpha.sum())
    neg_H_q_pi = -log_B_alpha + torch.sum((alpha - 1.0) * E_log_pi)

    return E_log_p_y + E_log_p_z_given_c + E_log_p_c + E_log_p_pi + H_q_z + H_q_c - neg_H_q_pi


def e_step_responsibilities_scflame(m, s, nu_k, omega2_k, alpha, temperature=1.0) -> torch.Tensor:
    """E-step: Closed-form update of cluster responsibilities q(c), with optional annealing temperature."""
    N, q = m.shape
    K = len(alpha)
    E_log_pi = torch.digamma(alpha) - torch.digamma(alpha.sum())
    log_r = torch.zeros(N, K, device=m.device)

    for k in range(K):
        diff_sq = (m - nu_k[k].unsqueeze(0)) ** 2
        log_p_z_k = -0.5 * torch.sum(
            torch.log(2 * torch.pi * omega2_k[k]) + (diff_sq + s ** 2) / omega2_k[k], dim=1
        )
        log_r[:, k] = (E_log_pi[k] + log_p_z_k) / temperature

    return torch.exp(log_r - torch.logsumexp(log_r, dim=1, keepdim=True))


def e_step_latent_scflame(X, L, gamma, log_phi, A, m, log_s, r, nu_k, omega2_k,
                           mc_samples=5, lr=1e-2, steps=10):
    """E-step:Gradient-based update of latent variational parameters q(z)."""
    N, D = X.shape
    K = r.shape[1]
    m_p = m.detach().clone().requires_grad_(True)
    ls_p = log_s.detach().clone().requires_grad_(True)
    opt = optim.Adam([m_p, ls_p], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        s_p = torch.exp(ls_p)
        mc_z = sample_diag_gaussian(m_p, ls_p, mc=mc_samples)
        Lz = torch.einsum("snq,dq->snd", mc_z, L)
        log_lik = torch.zeros(mc_samples, N, device=X.device)

        for s_idx in range(mc_samples):
            logmu = gamma.unsqueeze(0) + Lz[s_idx] + torch.log(A).unsqueeze(1)
            mu = torch.exp(logmu).clamp(min=1e-8)
            phi = torch.exp(log_phi).unsqueeze(0).clamp(min=0.1, max=100)
            log_lik[s_idx] = log_nb_pmf(X, mu, phi).sum(dim=1)

        E_log_p_y = log_lik.mean(0)
        E_log_p_z = torch.zeros(N, device=X.device)

        for k in range(K):
            diff_sq = (m_p - nu_k[k].unsqueeze(0)) ** 2
            log_p_z_k = -0.5 * torch.sum(
                torch.log(2 * torch.pi * omega2_k[k]) + (diff_sq + s_p ** 2) / omega2_k[k], dim=1
            )
            E_log_p_z += r[:, k] * log_p_z_k

        H_q_z = 0.5 * torch.sum(1 + np.log(2 * np.pi) + 2 * ls_p, dim=1)
        (-torch.sum(E_log_p_y + E_log_p_z + H_q_z)).backward()
        nn.utils.clip_grad_norm_([m_p, ls_p], max_norm=5.0)
        opt.step()

    return m_p.detach(), ls_p.detach()


def m_step_cluster_params(m: torch.Tensor, s: torch.Tensor, r: torch.Tensor):
    """M-step: closed-form update of the cluster means and shared (per-dimension) variances."""
    N, q = m.shape
    K = r.shape[1]
    N_k = r.sum(dim=0)
    pi_k = N_k / N
    nu_k = torch.zeros(K, q, device=m.device)

    for k in range(K):
        nu_k[k] = (r[:, k:k + 1] * m).sum(0) / N_k[k]

    omega2_shared = torch.zeros(q, device=m.device)
    for k in range(K):
        diff_sq = (m - nu_k[k].unsqueeze(0)) ** 2
        omega2_shared += (r[:, k:k + 1] * (diff_sq + s ** 2)).sum(0)

    return nu_k, (omega2_shared / N).clamp(min=1e-6).unsqueeze(0).repeat(K, 1), pi_k


def train_scflame(X, nbfa_result, gmm_result, K, alpha_prior=0.1,
                   epochs=200, mc_samples=5,
                   lr_latent=1e-3, latent_steps=3,
                   lr_L=1e-3, L_steps=1, L_reg=0.0,
                   temp_annealing=True, temp_warmup=None,
                   max_temp=2.0, verbose=False):
    """Fit scFLAME with VI, initialised with NBFA.

    Args:
        X: (N, D) count matrix.
        nbfa_result: output of `train_nb_fa` on the same data.
        gmm_result: either a fitted `EII_GMM`, or a dict with keys
            'nu_k', 'omega2_k', 'pi_k', 'r' (see `make_*_init` below).
        K: number of mixture components.
        alpha_prior: Dirichlet prior on the mixture weights (scalar or array of length K).
        epochs: number of training epochs.
        mc_samples: Monte Carlo samples per ELBO evaluation.
        lr_latent: learning rate for the latent variables.
        latent_steps: number of gradient steps to take for the latent variables per epoch.
        lr_L: learning rate for L and phi.
        L_steps: number of gradient steps to take for L and phi per epoch.
        L_reg: optional ridge penalty on L, keeping it near initialisation.
        temp_annealing: if True, anneal the responsibility "temperature"
            from `max_temp` down to 1 over the first third of training,
            which softens early cluster assignments.
        temp_warmup: number of annealing epochs (default: 1/3 of `epochs`).
        max_temp: starting temperature for annealing (default: 2.0).
        verbose: if True, print per-epoch training progress.

    Returns a dict with the fitted variational parameters and training trace.
    """
    N, D = X.shape
    device = X.device

    L = nn.Parameter(nbfa_result["L"].detach().clone())
    L_init = nbfa_result["L"].detach().clone()
    gamma = nbfa_result["alpha"].detach().clone()
    log_phi = nn.Parameter(nbfa_result["log_phi"].detach().clone())
    q = L.shape[1]
    A = nbfa_result["C"].detach().clone() if "C" in nbfa_result else torch.ones(N, device=device)
    m = nbfa_result["m"].detach().clone()
    log_s = nbfa_result["log_s"].detach().clone()

    if isinstance(alpha_prior, (int, float)):
        alpha_0 = torch.full((K,), float(alpha_prior), device=device)
    else:
        alpha_0 = torch.as_tensor(alpha_prior, device=device, dtype=torch.float32)
    alpha = alpha_0.clone()

    if temp_annealing:
        if temp_warmup is None:
            temp_warmup = epochs // 3
        temp_schedule = np.concatenate([
            np.linspace(max_temp, 1.0, temp_warmup),
            np.ones(epochs - temp_warmup),
        ])
    else:
        temp_schedule = np.ones(epochs)

    if hasattr(gmm_result, "means_"):
        nu_k = torch.tensor(gmm_result.means_, dtype=torch.float32, device=device)
        omega2_k = torch.full((K, q), gmm_result.sigma2_, dtype=torch.float32, device=device)
        pi_k = torch.tensor(gmm_result.weights_, dtype=torch.float32, device=device)
        r = torch.tensor(gmm_result.predict_proba(m.cpu().numpy()), dtype=torch.float32, device=device)
    else:
        nu_k = torch.tensor(gmm_result["nu_k"], dtype=torch.float32, device=device)
        omega2_k = torch.tensor(gmm_result["omega2_k"], dtype=torch.float32, device=device)
        pi_k = torch.tensor(gmm_result["pi_k"], dtype=torch.float32, device=device)
        r = torch.tensor(gmm_result["r"], dtype=torch.float32, device=device)

    opt_main = optim.Adam([L, log_phi], lr=lr_L)

    trace = {"elbo": [], "phi_mean": []}
    t0 = time.time()

    for ep in range(epochs):
        temp = temp_schedule[ep]

        # E-step: latents, then responsibilities
        m, log_s = e_step_latent_scflame(
            X, L, gamma, log_phi, A, m, log_s, r, nu_k, omega2_k,
            mc_samples=mc_samples, lr=lr_latent, steps=latent_steps,
        )
        s = torch.exp(log_s)
        r = e_step_responsibilities_scflame(m, s, nu_k, omega2_k, alpha, temperature=temp)

        # M-step: cluster parameters and mixture-weight pseudo-counts
        nu_k, omega2_k, pi_k = m_step_cluster_params(m, s, r)
        alpha = m_step_alpha(r, alpha_0)

        # M-step: factor loadings and dispersion
        opt_main.zero_grad()
        mc_z = sample_diag_gaussian(m.detach(), log_s.detach(), mc=mc_samples)
        Lz = torch.einsum("snq,dq->snd", mc_z, L)

        total_recon = 0.0
        for si in range(mc_samples):
            mu = torch.exp(gamma.unsqueeze(0) + Lz[si] + torch.log(A).unsqueeze(1)).clamp(min=1e-8)
            phi = torch.exp(log_phi).unsqueeze(0).clamp(min=0.05, max=100.0)
            total_recon += log_nb_pmf(X, mu, phi).sum()

        # small ridge penalty keeping L near its NBFA-initialised value
        loss = -(total_recon / mc_samples) + L_reg * torch.sum((L - L_init) ** 2)
        loss.backward()
        opt_main.step()

        with torch.no_grad():
            cur_elbo = compute_elbo_scflame(X, L, gamma, log_phi, A, m, log_s, r, nu_k, omega2_k, alpha, alpha_0)
            trace["elbo"].append(cur_elbo.item())
            trace["phi_mean"].append(torch.exp(log_phi).mean().item())

        if verbose and ep % 20 == 0:
            print(f"  Ep {ep:3d} | ELBO: {trace['elbo'][-1]:.2f} | Avg Phi: {trace['phi_mean'][-1]:.3f}")

    if verbose:
        print(f"scFLAME training time: {time.time() - t0:.2f}s")

    return {
        "r": r, "m": m, "log_s": log_s, "nu_k": nu_k, "L": L.detach(),
        "log_phi": log_phi.detach(), "trace": trace, "gamma": gamma, "A": A,
        "alpha": alpha, "pi_k": pi_k,
    }


# ============================================================
# Clustering initialisations
# ============================================================

def make_eii_gmm_init(nbfa_result, K, device, seed) -> EII_GMM:
    """EII GMM initialisation of scFLAME."""
    np.random.seed(seed)
    m_np = nbfa_result["m"].detach().cpu().numpy()
    gmm = EII_GMM(n_components=K, n_init=10, random_state=seed)
    gmm.fit(m_np)
    return gmm


def make_random_init(nbfa_result, K, device, seed) -> dict:
    """Random initialisation of clusters for scFLAME."""
    np.random.seed(seed)
    m_np = nbfa_result["m"].detach().cpu().numpy()
    N, q = m_np.shape
    idx = np.random.choice(N, K, replace=False)
    nu_k = m_np[idx].copy()
    omega2_k = np.ones((K, q)) * np.var(m_np)
    pi_k = np.ones(K) / K
    m_t = nbfa_result["m"].detach()
    s_t = torch.exp(nbfa_result["log_s"].detach())
    r = e_step_responsibilities_scflame(
        m_t, s_t,
        torch.tensor(nu_k, dtype=torch.float32, device=device),
        torch.tensor(omega2_k, dtype=torch.float32, device=device),
        torch.tensor(pi_k, dtype=torch.float32, device=device),
    )
    return {"nu_k": nu_k, "omega2_k": omega2_k, "pi_k": pi_k, "r": r.cpu().numpy()}


def make_random_far_init(nbfa_result, K, device, seed) -> dict:
    """Random far initialisation of clusters for scFLAME."""
    np.random.seed(seed)
    q = nbfa_result["m"].shape[1]
    nu_k = np.random.randn(K, q) * 3
    omega2_k = np.ones((K, q))
    pi_k = np.ones(K) / K
    m_t = nbfa_result["m"].detach()
    s_t = torch.exp(nbfa_result["log_s"].detach())
    r = e_step_responsibilities_scflame(
        m_t, s_t,
        torch.tensor(nu_k, dtype=torch.float32, device=device),
        torch.tensor(omega2_k, dtype=torch.float32, device=device),
        torch.tensor(pi_k, dtype=torch.float32, device=device),
    )
    return {"nu_k": nu_k, "omega2_k": omega2_k, "pi_k": pi_k, "r": r.cpu().numpy()}
