"""
scflame.merging
================
Functions for post-hoc greedy cluster merging for scFLAME.

Typical usage (see scripts/run_merge.py):

    initial_result = train_scflame(X, nbfa_result, gmm_init, K_init, ...)
    nu_k, Sigma_k, pi_k = m_step_cluster_params_full(
        initial_result["m"], torch.exp(initial_result["log_s"]), initial_result["r"])
    merge_input = {**initial_result, "nu_k": nu_k, "Sigma_k": Sigma_k, "pi_k": pi_k}
    final_result, K_reached, history, allocations = greedy_merge_full(
        X, merge_input, K_final=K_target, criterion="icl")
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .utils import EII_GMM, clustering_accuracy

VALID_CRITERIA = ("bic", "icl", 'dbi', 'bhattacharyya') 


# ============================================================
# Pre-screening: Mahalanobis distance between cluster means
# (cheap O(K^2) screen before the more expensive scoring below)
# ============================================================

def mahalanobis_between_clusters(nu_j, nu_k, omega2_j, omega2_k, pi_j, pi_k):
    """
    Mahalanobis distance between cluster means nu_j and nu_k, using the
    responsibility-weighted average (diagonal) covariance as the metric --
    consistent with Hennig (2010).
    """
    pi_m = pi_j + pi_k
    omega2_avg = (pi_j * omega2_j + pi_k * omega2_k) / pi_m
    diff = nu_j - nu_k
    return (diff ** 2 / omega2_avg.clamp(min=1e-8)).sum().item()


def get_merge_candidates(nu_k, omega2_k, pi_k, top_m=None):
    """
    Compute Mahalanobis distances for all O(K^2) cluster pairs, return
    pairs sorted by ascending distance (closest first).

    top_m=None  -> evaluate all pairs (safe for small K)
    top_m=K     -> evaluate only the K cheapest pairs (recommended for large K)
    """
    K = nu_k.shape[0]
    distances = []
    for j in range(K):
        for kk in range(j + 1, K):
            d = mahalanobis_between_clusters(
                nu_k[j], nu_k[kk], omega2_k[j], omega2_k[kk], pi_k[j], pi_k[kk],
            )
            distances.append((d, j, kk))

    distances.sort(key=lambda x: x[0])  # ascending: closest pairs first

    if top_m is not None:
        distances = distances[:top_m]

    return distances  # list of (distance, j, kk)


def bhattacharyya_distance(nu_i, Sigma_i, nu_j, Sigma_j):
    """Bhattacharyya distance between two full-covariance Gaussians."""
    Sigma = 0.5 * (Sigma_i + Sigma_j)

    L = torch.linalg.cholesky(Sigma)  # for numerical stability
    diff = nu_i - nu_j

    sol = torch.cholesky_solve(diff.unsqueeze(1), L)
    term1 = 0.125 * (diff.unsqueeze(0) @ sol).squeeze()

    logdet_Sigma = 2 * torch.log(torch.diagonal(L)).sum()
    logdet_i = torch.logdet(Sigma_i)
    logdet_j = torch.logdet(Sigma_j)
    term2 = 0.5 * (logdet_Sigma - 0.5 * (logdet_i + logdet_j))

    return (term1 + term2).item()


def compute_bhattacharyya_matrix(nu_k, Sigma_k):
    """Full K x K matrix of pairwise Bhattacharyya distances."""
    K = nu_k.shape[0]
    D = torch.zeros(K, K, device=nu_k.device)
    for i in range(K):
        for j in range(i + 1, K):
            d = bhattacharyya_distance(nu_k[i], Sigma_k[i], nu_k[j], Sigma_k[j])
            D[i, j] = D[j, i] = d
    return D


# ============================================================
# Merge proposal, and full-covariance E/M steps
# ============================================================

def propose_merge_full(r, m, s, nu_k, pi_k, Sigma_k, j, kk):
    """Merge clusters j and kk by summing their responsibilities, then re-fit
    full-covariance cluster parameters on the reduced responsibility matrix."""
    K = pi_k.shape[0]
    r_new = r.clone()
    r_new[:, j] = r[:, j] + r[:, kk]
    keep = [i for i in range(K) if i != kk]
    r_new = r_new[:, keep]

    nu_new, Sigma_new, pi_new = m_step_cluster_params_full(m, s, r_new)
    return r_new, nu_new, pi_new, Sigma_new


def mvn_log_prob(m, s, nu_k, Sigma_k):
    """
    E_q(z)[log p(z | c=k)] for a full-covariance component, accounting for
    posterior variance `s**2` via the trace term.

    m:       (N, q) posterior means
    s:       (N, q) posterior std
    nu_k:    (q,)   component mean
    Sigma_k: (q, q) component covariance
    """
    q = m.shape[1]
    L_chol = torch.linalg.cholesky(Sigma_k)
    diff = m - nu_k.unsqueeze(0)

    v = torch.linalg.solve_triangular(L_chol, diff.T, upper=False).T
    mahal = (v ** 2).sum(dim=1)

    Sigma_inv = torch.cholesky_inverse(L_chol)
    trace_term = (Sigma_inv.diag() * s ** 2).sum(dim=1)

    log_det = 2.0 * L_chol.diagonal().log().sum()
    log_p = -0.5 * (q * np.log(2 * np.pi) + log_det + mahal + trace_term)
    return log_p  # (N,)


def compute_elbo_scflame_full(X, m, log_s, r, nu_k, Sigma_k, pi_k) -> torch.Tensor:
    """ELBO of the full-covariance cluster model (factor-analysis likelihood
    terms are held fixed during merging, so only the clustering terms enter)."""
    N, D = X.shape
    K = len(pi_k)
    s = torch.exp(log_s)

    E_log_p_z = torch.tensor(0.0, device=X.device)
    for k in range(K):
        E_log_p_z += (r[:, k] * mvn_log_prob(m, s, nu_k[k], Sigma_k[k])).sum()

    E_log_p_c = torch.sum(r * torch.log(pi_k + 1e-10).unsqueeze(0))
    H_q_c = -torch.sum(r * torch.log(r + 1e-10))

    return E_log_p_z + E_log_p_c + H_q_c


def e_step_responsibilities_full(m, s, nu_k, Sigma_k, pi_k, r, temperature=1.0, alpha_0=0.1):
    """Responsibility update for the full-covariance model with a Dirichlet
    posterior on the mixture weights."""
    N, K = m.shape[0], len(pi_k)
    log_r = torch.zeros(N, K, device=m.device)

    N_k = r.sum(dim=0)
    alpha_k = alpha_0 + N_k
    alpha_sum = alpha_k.sum()
    E_log_pi = torch.digamma(alpha_k) - torch.digamma(alpha_sum)

    for k in range(K):
        log_r[:, k] = (E_log_pi[k] + mvn_log_prob(m, s, nu_k[k], Sigma_k[k])) / temperature

    return torch.exp(log_r - torch.logsumexp(log_r, dim=1, keepdim=True))


def m_step_cluster_params_full(m, s, r, min_n_k=1.0):
    """M-step for full-covariance cluster means/covariances/weights."""
    N, q = m.shape
    K = r.shape[1]
    N_k = r.sum(dim=0).clamp(min=min_n_k)  # prevent div by zero
    pi_k = N_k / N_k.sum()  # renormalise after clamp

    nu_k = torch.zeros(K, q, device=m.device)
    Sigma_k = torch.zeros(K, q, q, device=m.device)

    for k in range(K):
        nu_k[k] = (r[:, k:k + 1] * m).sum(0) / N_k[k]

    for k in range(K):
        diff = m - nu_k[k].unsqueeze(0)
        outer = torch.einsum("n,ni,nj->ij", r[:, k], diff, diff)
        s_sq = (r[:, k:k + 1] * s ** 2).sum(0)
        Sigma_k[k] = (outer + torch.diag(s_sq)) / N_k[k]
        Sigma_k[k] += 1e-4 * torch.eye(q, device=m.device)

    return nu_k, Sigma_k, pi_k


# ============================================================
# BIC / ICL / DBI 
# ============================================================

def count_parameters_scflame(K: int, q: int) -> int:
    """
    Free parameters in the GMM over latent space:
      - K cluster means (nu_k): K * q
      - K full covariance matrices (Sigma_k): K * q*(q+1)/2
      - mixture weights (pi_k): K - 1
    NBFA parameters (L, alpha, phi) are deliberately excluded (fixed during merging)
    """
    means_params = K * q
    cov_params = K * q * (q + 1) // 2
    weight_params = K - 1
    return means_params + cov_params + weight_params


def expected_complete_log_likelihood(m, log_s, r, nu_k, Sigma_k, pi_k, alpha_0=0.1) -> float:
    s = torch.exp(log_s)
    K = len(pi_k)

    E_log_p_z = torch.tensor(0.0, device=m.device)
    for k in range(K):
        E_log_p_z += (r[:, k] * mvn_log_prob(m, s, nu_k[k], Sigma_k[k])).sum()

    E_log_p_c = torch.sum(r * torch.log(pi_k.clamp(min=1e-10)).unsqueeze(0))

    # Dirichlet prior: penalises small pi_k when alpha_0 < 1
    log_prior = (alpha_0 - 1.0) * torch.log(pi_k.clamp(min=1e-10)).sum()

    return (E_log_p_z + E_log_p_c + log_prior).item()


def assignment_entropy(r: torch.Tensor) -> float:
    """H(q(c)) = -sum_i sum_k r_ik log r_ik (always >= 0)."""
    return -torch.sum(r * torch.log(r.clamp(min=1e-10))).item()


def compute_bic_icl(m, log_s, r, nu_k, Sigma_k, pi_k, N: int):
    """
    Returns (BIC, ICL) - higher is better (we MAXIMISE these).
    """
    K = len(pi_k)
    q = m.shape[1]
    p = count_parameters_scflame(K, q)

    Q = expected_complete_log_likelihood(m, log_s, r, nu_k, Sigma_k, pi_k)
    H = assignment_entropy(r)
    bic = 2.0 * Q - p * np.log(N)
    icl = bic - 2.0 * H  # ICL <= BIC; entropy term is non-negative

    return bic, icl


def compute_dbi(m, s, r, nu_k) -> float:
    """Davies-Bouldin index for soft clustering with uncertainty. Lower is better."""
    N, q = m.shape
    K = nu_k.shape[0]
    s_sq = torch.exp(2 * s) if s.shape == m.shape else s ** 2

    N_k = r.sum(dim=0) + 1e-8

    S = torch.zeros(K, device=m.device)
    for k in range(K):
        diff_sq = (m - nu_k[k].unsqueeze(0)) ** 2
        dist_sq = diff_sq + s_sq
        S[k] = torch.sqrt((r[:, k:k + 1] * dist_sq).sum() / N_k[k])

    M = torch.cdist(nu_k, nu_k, p=2) + 1e-8

    dbi = 0.0
    for i in range(K):
        ratios = [(S[i] + S[j]) / M[i, j] for j in range(K) if j != i]
        dbi += torch.max(torch.stack(ratios))

    return (dbi / K).item()


# ============================================================
# Greedy merge loop
# ============================================================

def greedy_merge_full(
    X,
    initial_result: dict,
    top_m: Optional[int] = None,
    verbose: bool = True,
    c_true: Optional[np.ndarray] = None,
    K_true: Optional[int] = None,
    criterion: str = "icl",
    K_final: int = 2,
    checkpoint_dir: Optional[str] = None,
    checkpoint_tag: str = "run",
):
    """
    Greedily merge the pair of clusters that most improves `criterion`,
    stopping once K reaches `K_final`.

    Args:
        X: (N, D) data matrix
        initial_result: dict with keys 'r', 'm', 'log_s', 'nu_k', 'Sigma_k', 'pi_k'
            (e.g. from `m_step_cluster_params_full` applied to a `train_scflame` fit).
        top_m: if given, only evaluate the top_m closest cluster pairs (by Mahalanobis distance) at each merge step; if None, evaluate all O(K^2) pairs.
        verbose: if True, print progress at each merge step
        c_true: if given, compute ARI/NMI/accuracy against these true labels at each step
        K_true: if given, print the true number of clusters at each step
        criterion: one of {VALID_CRITERIA}. 'bic'/'icl' are maximised;
            'dbi'/'bhattacharyya' are minimised. 
        K_final: stop merging once K reaches this value
        checkpoint_dir: if given, save the merged result to this directory after
            every merge step (as f"{checkpoint_tag}_K{K}.pt"); the directory is
            created if it doesn't exist. If None, no checkpoints are written.
        checkpoint_tag: string tag to include in checkpoint filenames (default "run")

    Returns (final_result, K_reached, history, allocations), where `history` is a
    list of per-step dicts and `allocations` maps K -> predicted labels at that K.
    """
    if criterion not in VALID_CRITERIA:
        raise ValueError(f"criterion must be one of {VALID_CRITERIA}, got {criterion!r}")

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    current = dict(initial_result)
    K = current["pi_k"].shape[0]

    def compute_scores(result):
        r, m, log_s = result["r"], result["m"], result["log_s"]
        nu_k, Sigma_k, pi_k = result["nu_k"], result["Sigma_k"], result["pi_k"]
        s = torch.exp(log_s)

        bic, icl = compute_bic_icl(m, log_s, r, nu_k, Sigma_k, pi_k, X.shape[0])
        elbo = compute_elbo_scflame_full(X, m, log_s, r, nu_k, Sigma_k, pi_k).item()
        dbi = compute_dbi(m, s, r, nu_k)

        return {"bic": bic, "icl": icl, "elbo": elbo, "dbi": dbi}

    def is_distance_based():
        return criterion == "bhattacharyya"

    def is_maximised():
        # bic/icl: higher is better. dbi: lower is better.
        return criterion in ("bic", "icl")

    def labels_of(result):
        return result["r"].argmax(dim=1).cpu().numpy()

    def eval_metrics(result):
        preds = labels_of(result)
        if c_true is None:
            return None, None, None
        return (
            adjusted_rand_score(c_true, preds),
            normalized_mutual_info_score(c_true, preds),
            clustering_accuracy(c_true, preds),
        )

    # --- initial scoring ---
    current_scores = compute_scores(current)
    current["final_elbo"] = current_scores["elbo"]
    init_ari, init_nmi, init_acc = eval_metrics(current)
    init_dbi = current_scores["dbi"] if "dbi" in current_scores else None

    history = [{
        "K": K, "merged_pair": None, "delta_criterion": 0.0,
        "ari": init_ari, "nmi": init_nmi, "acc": init_acc, "dbi": init_dbi,
        **{f"{s}_score": current_scores[s] for s in current_scores},
    }]
    allocations = {K: labels_of(current).copy()}

    if verbose:
        if is_distance_based():
            print(f"Start | K={K} | {criterion.upper()} (distance-based)")
        else:
            print(f"Start | K={K} | {criterion.upper()}={current_scores[criterion]:.4f}")

    # ============================================================
    # Main loop
    # ============================================================
    while K > K_final:
        omega2_diag = torch.stack([current["Sigma_k"][k].diagonal() for k in range(K)])

        candidates = get_merge_candidates(
            current["nu_k"], omega2_diag, current["pi_k"],
            top_m=top_m if top_m is None else min(top_m, K * (K - 1) // 2),
        )

        if verbose:
            print(f"Evaluating {len(candidates)} candidate merges")

        best_value = np.inf if (is_distance_based() or not is_maximised()) else -np.inf
        best_pair = best_result = best_scores = best_dist = None

        for maha_dist, j, k in candidates:
            r_m, nu_m, pi_m, Sig_m = propose_merge_full(
                current["r"], current["m"], torch.exp(current["log_s"]),
                current["nu_k"], current["pi_k"], current["Sigma_k"], j, k,
            )
            candidate_result = {
                "r": r_m, "m": current["m"], "log_s": current["log_s"],
                "nu_k": nu_m, "Sigma_k": Sig_m, "pi_k": pi_m,
            }
            cand_scores = compute_scores(candidate_result)

            if is_distance_based():
                # Use the true Bhattacharyya distance for the merge decision;
                # `maha_dist` was only used to cheaply pre-screen candidates.
                dist_val = bhattacharyya_distance(
                    current["nu_k"][j], current["Sigma_k"][j],
                    current["nu_k"][k], current["Sigma_k"][k],
                )
                crit_val = dist_val
                better = crit_val < best_value
            else:
                dist_val = maha_dist
                crit_val = cand_scores[criterion]
                better = (crit_val > best_value) if is_maximised() else (crit_val < best_value)

            if better:
                best_value, best_pair = crit_val, (j, k)
                best_result, best_scores, best_dist = candidate_result, cand_scores, dist_val

        # ========================================================
        # Apply the best merge
        # ========================================================
        delta_criterion = best_value if is_distance_based() else (best_value - current_scores[criterion])

        current = best_result
        current_scores = best_scores
        current["final_elbo"] = current_scores["elbo"]
        K -= 1

        if checkpoint_dir is not None:
            torch.save(current, os.path.join(checkpoint_dir, f"{checkpoint_tag}_K{K}.pt"))

        allocations[K] = labels_of(current).copy()
        step_ari, step_nmi, step_acc = eval_metrics(current)
        step_dbi = current_scores["dbi"] if "dbi" in current_scores else None

        if verbose:
            ari_str = f"{step_ari:.4f}" if step_ari is not None else "n/a"
            nmi_str = f"{step_nmi:.4f}" if step_nmi is not None else "n/a"
            acc_str = f"{step_acc:.4f}" if step_acc is not None else "n/a"
            dbi_str = f"{step_dbi:.4f}" if step_dbi is not None else "n/a"
            print(
                f"Merge {best_pair} | K: {K + 1}->{K} | {criterion} Delta={delta_criterion:+.4f} | "
                f"ELBO={current_scores['elbo']:.2f} | BIC={current_scores['bic']:.2f} | "
                f"ICL={current_scores['icl']:.2f} | ARI={ari_str} | NMI={nmi_str} | ACC={acc_str} | DBI={dbi_str}"
            )

        history.append({
            "K": K, "merged_pair": best_pair, "delta_criterion": delta_criterion,
            "mahal_or_bhat_dist": best_dist,
            "ari": step_ari, "nmi": step_nmi, "acc": step_acc, "dbi": step_dbi,
            **{f"{s}_score": current_scores[s] for s in current_scores},
        })

        if K <= K_final and verbose:
            print(f"Reached K={K} (target K_final={K_final}), stopping.")

    return current, K, history, allocations
