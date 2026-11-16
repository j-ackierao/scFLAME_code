import torch
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kl_diag_gaussian_to_std_normal(m, log_s):
    var = torch.exp(2.0 * log_s)
    return 0.5 * torch.sum(var + m * m - 1.0 - 2.0 * log_s, dim=-1)

def compute_expected_log_likelihood(X, mu, L, log_psi, m, S):
    """
    Compute E_q[log p(x|z,c)] for all samples and clusters.
    This is computed ONCE after M-step and cached for next E-step.
    
    Returns:
        log_lik: N x K tensor of expected log likelihoods
    """
    N, D = X.shape
    K = len(L)
    device = X.device
    
    log_lik = torch.zeros(N, K, device=device)
    
    for k in range(K):
        x_mean = mu[k].unsqueeze(0) + m[:, k, :] @ L[k].t()  # N x D
        residual = X - x_mean  # N x D
        log_psi_k = log_psi[k].clamp(min=-10, max=5)
        psi_k = torch.exp(log_psi_k)  # D
        
        # Mahalanobis distance for mean part
        mahal = torch.sum(residual**2 / psi_k.unsqueeze(0), dim=1)  # N
        
        # Trace term: tr(Ψ^{-1} L S L^T) for each sample
        LS = L[k] * S[:, k, :].unsqueeze(1)  # N x D x q (broadcasting)
        trace_term = torch.sum(LS * L[k].unsqueeze(0), dim=2) / psi_k.unsqueeze(0)  # N x D
        trace_term = torch.sum(trace_term, dim=1)  # N
        
        # Log likelihood
        log_det_psi = torch.sum(log_psi_k)
        log_lik[:, k] = -0.5 * (D * np.log(2*np.pi) + log_det_psi + mahal + trace_term)
    
    return log_lik

# ---------- E-step Updates ----------

def e_step_responsibilities(X, L, alpha, m, S, cached_log_lik):
    """
    Closed-form update for responsibilities using cached log likelihood.
    
    Args:
        cached_log_lik: N x K tensor of E_q[log p(x|z,c)] from previous M-step
    
    Returns:
        r: N x K normalized responsibilities
    """
    N = X.shape[0]
    K = len(L)
    
    # Prior over z: N(0, I)
    log_p_z = -0.5 * (m**2 + S).sum(dim=2)  # N x K
    
    # E[log π]
    E_log_pi = torch.digamma(alpha) - torch.digamma(alpha.sum())
    
    # Unnormalized log responsibilities using cached log_lik
    log_s = 0.5 * torch.log(S)
    kl_z = kl_diag_gaussian_to_std_normal(m, log_s)  # N x K
    log_r_unnorm = E_log_pi.unsqueeze(0) + cached_log_lik - kl_z
    
    # Normalize
    log_r = F.log_softmax(log_r_unnorm, dim=1)
    r = torch.exp(log_r)
    
    return r

def e_step_latent_posterior(X, mu, L, log_psi, r):
    """
    Closed-form update for latent posterior q(z_ik) = N(m_ik, S_ik).
    
    Returns:
        m: N x K x q posterior means
        S: N x K x q posterior variances (diagonal covariance)
    """
    N, D = X.shape
    K = len(L)
    q = L[0].shape[1]
    device = X.device
    
    m = torch.zeros(N, K, q, device=device)
    S = torch.zeros(N, K, q, device=device)
    
    for k in range(K):
        log_psi_k = log_psi[k].clamp(min=-10, max=5)
        psi_k = torch.exp(log_psi_k)  # D
        
        # Posterior covariance: S = (I + L^T Ψ^{-1} L)^{-1}
        # For diagonal Ψ: L^T Ψ^{-1} L is q x q
        LT_psi_inv_L = (L[k].t() / psi_k.unsqueeze(0)) @ L[k]  # q x q
        S_inv = torch.eye(q, device=device) + LT_psi_inv_L
        S_k = torch.inverse(S_inv)  # q x q
        
        # Posterior mean: m = S L^T Ψ^{-1} (x - μ)
        residual = X - mu[k].unsqueeze(0)  # N x D
        LT_psi_inv_residual = (L[k].t() / psi_k.unsqueeze(0)) @ residual.t()  # q x N
        m[:, k, :] = (S_k @ LT_psi_inv_residual).t()  # N x q
        
        # Store diagonal of S for each sample (same for all samples in a cluster)
        S[:, k, :] = torch.diag(S_k).unsqueeze(0).expand(N, -1)
    
    return m, S

# ---------- M-step Updates ----------

def m_step_alpha(r, alpha_0):
    """Closed-form update for alpha."""
    return alpha_0 + r.sum(dim=0)

def m_step_mu(X, r, m, L):
    """Closed-form update for cluster means."""
    K = len(L)
    D = X.shape[1]
    device = X.device
    mu = torch.zeros(K, D, device=device)
    
    for k in range(K):
        r_k = r[:, k].unsqueeze(1)  # N x 1
        r_k_sum = r_k.sum()
        
        # μ_k = (Σ r_ik x_i - L_k Σ r_ik m_ik) / Σ r_ik
        weighted_x = (r_k * X).sum(dim=0)  # D
        weighted_Lm = (r_k * (m[:, k, :] @ L[k].t())).sum(dim=0)  # D
        mu[k] = (weighted_x - weighted_Lm) / (r_k_sum + 1e-8)
    
    return mu

def m_step_L_psi_gradient(X, mu, L, log_psi, r, m, S, lr=1e-2, steps=5):
    """
    Gradient-based update for L and psi. Can adjust learning rate and steps.
    """
    K = len(L)
    
    # Make L and log_psi require grad
    L_params = [l.detach().requires_grad_(True) for l in L]
    psi_params = [lp.detach().requires_grad_(True) for lp in log_psi]
    
    opt = torch.optim.Adam(L_params + psi_params, lr=lr)
    
    for _ in range(steps):
        opt.zero_grad()
        
        # Compute expected log likelihood (part of ELBO)
        ell = 0.0
        for k in range(K):
            r_k = r[:, k]  # N
            
            # E[log p(x | z, k)]
            x_mean = mu[k].unsqueeze(0) + m[:, k, :] @ L_params[k].t()  # N x D
            residual = X - x_mean
            
            log_psi_k = psi_params[k].clamp(min=-10, max=5)
            psi_k = torch.exp(log_psi_k)
            
            # Mean part
            mahal = torch.sum(residual**2 / psi_k.unsqueeze(0), dim=1)  # N
            
            # Variance part: tr(Ψ^{-1} L S L^T)
            LS = L_params[k] * S[:, k, :].unsqueeze(1)
            trace_term = torch.sum(LS * L_params[k].unsqueeze(0), dim=2) / psi_k.unsqueeze(0)
            trace_term = torch.sum(trace_term, dim=1)
            
            log_det = torch.sum(log_psi_k)
            D = X.shape[1]
            log_lik = -0.5 * (D * np.log(2*np.pi) + log_det + mahal + trace_term)
            
            ell += torch.sum(r_k * log_lik)
        
        loss = -ell
        loss.backward()
        opt.step()
    
    # Return updated parameters (detached)
    return [l.detach() for l in L_params], [lp.detach() for lp in psi_params]

def compute_elbo(X, alpha, alpha_0, L, r, m, S, cached_log_lik):
    """
    Compute ELBO using cached log likelihood from M-step.
    
    Args:
        cached_log_lik: N x K tensor of E_q[log p(x|z,c)] (pre-computed after M-step)
    """
    N, D = X.shape
    K, q = len(L), L[0].shape[1]
    
    E_log_pi = torch.digamma(alpha) - torch.digamma(alpha.sum())
    log_r = torch.log(r + 1e-10)
    
    # E[log p(pi)] - Dirichlet prior
    E_log_p_pi = torch.sum((alpha_0 - 1.0) * E_log_pi)
    
    # E[log p(c | pi)]
    E_log_p_c = torch.sum(r * E_log_pi.unsqueeze(0))
    
    # KL[q(z|c) || p(z|c)] - sum_k E_q[log p(z|k)] + H[q(z|c)]
    log_s = 0.5 * torch.log(S)
    kl_z = kl_diag_gaussian_to_std_normal(m, log_s)  # N x K
    E_log_p_z_minus_H = -torch.sum(r * kl_z)
    
    # E[log p(x|z,c)] - use cached values
    E_log_p_x = torch.sum(r * cached_log_lik)
    
    # -H[q(pi)] - Negative entropy of Dirichlet
    log_B_alpha = torch.sum(torch.lgamma(alpha)) - torch.lgamma(alpha.sum())
    neg_H_q_pi = -log_B_alpha + torch.sum((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(alpha.sum())))
    
    # -H[q(c)] - Negative entropy of categorical
    neg_H_q_c = torch.sum(r * log_r)
    
    elbo = E_log_p_pi + E_log_p_c + E_log_p_z_minus_H + E_log_p_x - neg_H_q_pi - neg_H_q_c
    
    return elbo

def init_r_even(N, K, device):
    # Even split
    labels = torch.arange(N, device=device) % K
    labels = labels[torch.randperm(N, device=device)]

    r = torch.zeros(N, K, device=device)
    r[torch.arange(N), labels] = 1.0
    return r

# ---------- Training with Closed-Form Updates ----------

def train_mixture_fa_closed_form(X, K=3, q=2, epochs=100, lr_m_step=1e-2, m_step_iters=5, alpha_prior=1.0, verbose=True):
    """
    Train Mixture of Factor Analyzers using closed-form E-step and gradient M-step.
    Initialised with evenly sized clusters unless init="kmeans".
    """
    N, D = X.shape
    device = X.device

    if isinstance(alpha_prior, (int, float)):
        alpha_0 = torch.full((K,), float(alpha_prior), device=device)  # Symmetric prior
    else:
        alpha_0 = torch.as_tensor(alpha_prior, device=device, dtype=torch.float32)

    assert alpha_0.shape == (K,), f"alpha_prior must have shape ({K},), got {alpha_0.shape}"
    assert (alpha_0 > 0).all(), "All Dirichlet concentrations must be positive"
    
    # Initialize parameters
    alpha = alpha_0.clone() + torch.randn(K, device=device) * 0.1
    mu = torch.randn(K, D, device=device) * 0.1
    L = [torch.randn(D, q, device=device) * 0.1 for _ in range(K)]
    log_psi = [torch.zeros(D, device=device) for _ in range(K)]
    
    # Initialize variational parameters (will be updated in closed form)
    m = torch.randn(N, K, q, device=device) * 0.1
    S = torch.ones(N, K, q, device=device) * 0.5
    
    cached_log_lik = None
    trace = []
    start = time.time()
    
    for ep in range(epochs):
        # ========== E-step ==========
        if ep == 0:
            r = init_r_even(N, K, device)
            for k in range(K):
                mu[k] = X[r[:, k] == 1].mean(dim=0)
        else:
            # Use cached log-likelihood from previous M-step
            r = e_step_responsibilities(X, L, alpha, m, S, cached_log_lik)
        m, S = e_step_latent_posterior(X, mu, L, log_psi, r)
        
        # ========== M-step ==========
        
        # Closed-form
        alpha = m_step_alpha(r, alpha_0)
        mu = m_step_mu(X, r, m, L)
        
        # Update L and psi (gradient-based)
        L, log_psi = m_step_L_psi_gradient(X, mu, L, log_psi, r, m, S, 
                                           lr=lr_m_step, steps=m_step_iters)
        
        with torch.no_grad():
            cached_log_lik = compute_expected_log_likelihood(X, mu, L, log_psi, m, S)
        
        # ========== Compute ELBO for monitoring ==========
        if ep % max(1, epochs//10) == 0 or ep == epochs-1:
            with torch.no_grad():
                elbo = compute_elbo(X, alpha, alpha_0, L, r, m, S, cached_log_lik)
                trace.append(elbo.item())
                
            if verbose and (ep % max(1, epochs//10) == 0 or ep == epochs-1):
                    cluster_sizes = r.sum(dim=0)
                    print(f"Epoch {ep:4d} | ELBO {elbo.item():8.1f} | ELBO/N {elbo.item()/N:7.3f} | "
                        f"Sizes {cluster_sizes.cpu().numpy().astype(int)}")
    
    print(f"Training time: {time.time()-start:.2f}s")
    
    return dict(alpha=alpha, mu=mu, L=L, log_psi=log_psi,
                r=r, m=m, S=S, trace=trace)