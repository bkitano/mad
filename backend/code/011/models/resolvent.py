"""
Resolvent computation methods for DPLR SSMs.

Implements:
1. Exact Woodbury resolvent: (zI - A)^{-1} via Woodbury identity
2. Neumann series resolvent: truncated Neumann approximation
3. Efficient kernel computation (avoids forming N x N resolvent)

Mathematical background (from Proposal 011):

DPLR matrix: A = Lambda + P @ Q^*
  - Lambda: diagonal matrix of eigenvalues (complex)
  - P, Q: low-rank factors (N x r)

Resolvent: R(z) = (zI - A)^{-1} = (M - PQ^*)^{-1} where M = zI - Lambda

Exact Woodbury (corrected signs):
  (M - PQ^*)^{-1} = D_z + D_z P (I - Q^* D_z P)^{-1} Q^* D_z
  where D_z = M^{-1} = diag(1/(z - lambda_i))

Neumann factorization:
  M - PQ^* = M(I - M^{-1}PQ^*) = M(I - E_z)  where E_z := D_z P Q^*
  => R(z) = (I - E_z)^{-1} D_z

Neumann approximation:
  (I - E_z)^{-1} ≈ S_k(E_z) = I + E + E^2 + ... + E^{k-1}
  => R_k(z) ≈ S_k(E_z) D_z

For rank-r E_z, the powers factor as:
  E_z^j = D_z P (Q^* D_z P)^{j-1} Q^*  for j >= 1

So the efficient kernel computation is:
  K_hat_k(z) = C R_k(z) B = C D_z B + sum_{m=1}^{k-1} (C D_z P)(Q^* D_z P)^{m-1}(Q^* D_z B)
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


def hippo_legs_init(N: int, r: int = 1, dtype=torch.float64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    HiPPO-LegS initialization for DPLR SSM.

    Eigenvalues: lambda_n = -1/2 + i * pi * n (stable with oscillatory imaginary part)
    Low-rank factors: P, Q ~ N(0, 1/sqrt(N))

    Returns:
        Lambda: (N,) complex diagonal eigenvalues
        P: (N, r) complex low-rank factor
        Q: (N, r) complex low-rank factor
    """
    real_parts = -0.5 * torch.ones(N, dtype=dtype)
    imag_parts = math.pi * torch.arange(N, dtype=dtype)
    Lambda = torch.complex(real_parts, imag_parts)  # (N,)

    P = torch.complex(
        torch.randn(N, r, dtype=dtype) / math.sqrt(N),
        torch.randn(N, r, dtype=dtype) / math.sqrt(N),
    )
    Q = torch.complex(
        torch.randn(N, r, dtype=dtype) / math.sqrt(N),
        torch.randn(N, r, dtype=dtype) / math.sqrt(N),
    )
    return Lambda, P, Q


def random_B_C(N: int, d: int, dtype=torch.float64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random input/output projections for SSM."""
    B = torch.complex(
        torch.randn(N, d, dtype=dtype) / math.sqrt(N),
        torch.randn(N, d, dtype=dtype) / math.sqrt(N),
    )
    C = torch.complex(
        torch.randn(d, N, dtype=dtype) / math.sqrt(N),
        torch.randn(d, N, dtype=dtype) / math.sqrt(N),
    )
    return B, C


def woodbury_resolvent(
    z: torch.Tensor,          # (L,) complex frequencies
    Lambda: torch.Tensor,     # (N,) complex eigenvalues
    P: torch.Tensor,          # (N, r) complex
    Q: torch.Tensor,          # (N, r) complex
) -> torch.Tensor:
    """
    Exact Woodbury resolvent: (zI - A)^{-1} for A = Lambda + P Q^*

    Using (M - PQ^*)^{-1} = D_z + D_z P (I - Q^* D_z P)^{-1} Q^* D_z
    where M = zI - Lambda, D_z = M^{-1}

    Args:
        z: (L,) complex frequency points
        Lambda: (N,) complex diagonal eigenvalues
        P: (N, r) complex low-rank factor
        Q: (N, r) complex low-rank factor

    Returns:
        R: (L, N, N) complex resolvent matrices
    """
    L = z.shape[0]
    N = Lambda.shape[0]
    r = P.shape[1]

    # D_z = diag(1/(z_j - lambda_i)) for each z_j: (L, N)
    D_z = 1.0 / (z.unsqueeze(1) - Lambda.unsqueeze(0))

    # D_z P: (L, N, r)
    D_z_P = D_z.unsqueeze(2) * P.unsqueeze(0)

    # Q^* D_z P = F: (L, r, r)
    Q_star = Q.conj().T  # (r, N)
    F = torch.einsum('rn,lnp->lrp', Q_star, D_z_P)

    # (I - F)^{-1}: (L, r, r)
    I_r = torch.eye(r, dtype=z.dtype, device=z.device).unsqueeze(0).expand(L, -1, -1)
    inv_term = torch.linalg.solve(I_r - F, I_r)

    # Q^* D_z: (L, r, N)
    Q_D_z = Q_star.unsqueeze(0) * D_z.unsqueeze(1)

    # Correction: D_z_P @ inv_term @ Q_D_z: (L, N, N)
    correction = torch.bmm(torch.bmm(D_z_P, inv_term), Q_D_z)

    # R(z) = D_z + correction (note: PLUS sign for Woodbury of (M - PQ^*))
    D_z_full = torch.zeros(L, N, N, dtype=z.dtype, device=z.device)
    idx = torch.arange(N, device=z.device)
    D_z_full[:, idx, idx] = D_z

    R = D_z_full + correction
    return R


def neumann_resolvent(
    z: torch.Tensor,          # (L,) complex frequencies
    Lambda: torch.Tensor,     # (N,) complex eigenvalues
    P: torch.Tensor,          # (N, r) complex
    Q: torch.Tensor,          # (N, r) complex
    k: int = 4,               # truncation order
) -> torch.Tensor:
    """
    Neumann series resolvent: R_k(z) = S_k(E_z) D_z

    where E_z = D_z P Q^* and S_k(E) = I + E + E^2 + ... + E^{k-1}

    Note: R(z) = (I - E_z)^{-1} D_z (resolvent is S * D_z, NOT D_z * S)

    Args:
        z: (L,) complex frequency points
        Lambda: (N,) complex diagonal eigenvalues
        P: (N, r) complex low-rank factor
        Q: (N, r) complex low-rank factor
        k: truncation order (number of Neumann terms)

    Returns:
        R: (L, N, N) complex resolvent matrices
    """
    L = z.shape[0]
    N = Lambda.shape[0]
    r = P.shape[1]

    # D_z: (L, N)
    D_z = 1.0 / (z.unsqueeze(1) - Lambda.unsqueeze(0))

    # D_z as full diagonal: (L, N, N)
    D_z_full = torch.zeros(L, N, N, dtype=z.dtype, device=z.device)
    idx = torch.arange(N, device=z.device)
    D_z_full[:, idx, idx] = D_z

    # Build S_k(E_z) using factored form: E_z^j = D_z P (Q^* D_z P)^{j-1} Q^*
    # S_k(E_z) = I + sum_{j=1}^{k-1} D_z P (Q^* D_z P)^{j-1} Q^*

    # Precompute pieces
    D_z_P = D_z.unsqueeze(2) * P.unsqueeze(0)  # (L, N, r)
    Q_star = Q.conj().T  # (r, N)
    F = torch.einsum('rn,lnp->lrp', Q_star, D_z_P)  # (L, r, r) - core matrix

    Q_star_exp = Q_star.unsqueeze(0).expand(L, -1, -1)  # (L, r, N)

    # S_k = I + sum of rank-r terms
    S = torch.zeros(L, N, N, dtype=z.dtype, device=z.device)
    S[:, idx, idx] = 1.0  # I

    F_power = torch.eye(r, dtype=z.dtype, device=z.device).unsqueeze(0).expand(L, -1, -1).clone()

    for j in range(1, k):
        # E_z^j = D_z_P @ F^{j-1} @ Q^*
        term = torch.bmm(torch.bmm(D_z_P, F_power), Q_star_exp)  # (L, N, N)
        S = S + term
        F_power = torch.bmm(F_power, F)

    # R_k(z) = S_k(E_z) @ D_z
    R = torch.bmm(S, D_z_full)
    return R


def compute_ssm_kernel_exact(
    z: torch.Tensor,          # (L,) complex frequencies
    Lambda: torch.Tensor,     # (N,) complex eigenvalues
    P: torch.Tensor,          # (N, r) complex
    Q: torch.Tensor,          # (N, r) complex
    B: torch.Tensor,          # (N, d) complex input projection
    C: torch.Tensor,          # (d, N) complex output projection
) -> torch.Tensor:
    """
    Compute SSM kernel K_hat(z) = C R(z) B using exact Woodbury.

    Returns:
        K: (L, d, d) complex kernel values
    """
    R = woodbury_resolvent(z, Lambda, P, Q)  # (L, N, N)
    C_exp = C.unsqueeze(0).expand(R.shape[0], -1, -1)
    B_exp = B.unsqueeze(0).expand(R.shape[0], -1, -1)
    K = torch.bmm(torch.bmm(C_exp, R), B_exp)
    return K


def compute_ssm_kernel_neumann(
    z: torch.Tensor,          # (L,) complex frequencies
    Lambda: torch.Tensor,     # (N,) complex eigenvalues
    P: torch.Tensor,          # (N, r) complex
    Q: torch.Tensor,          # (N, r) complex
    B: torch.Tensor,          # (N, d) complex input projection
    C: torch.Tensor,          # (d, N) complex output projection
    k: int = 4,               # truncation order
) -> torch.Tensor:
    """
    Compute SSM kernel using Neumann resolvent (efficient form).

    K_hat_k(z) = C S_k(E_z) D_z B
               = C D_z B + sum_{m=1}^{k-1} (C D_z P) (Q^* D_z P)^{m-1} (Q^* D_z B)

    Note: The base term is C @ (D_z B), not (C D_z) @ (D_z B) which would be C D_z^2 B.
    The correction terms use (C D_z P) which correctly has one D_z factor.

    Returns:
        K: (L, d, d) complex kernel values
    """
    L = z.shape[0]
    N = Lambda.shape[0]
    r = P.shape[1]
    d = B.shape[1]

    # D_z: (L, N)
    D_z = 1.0 / (z.unsqueeze(1) - Lambda.unsqueeze(0))

    # D_z B: (L, N, d) - scale rows of B by D_z
    D_z_B = D_z.unsqueeze(2) * B.unsqueeze(0)

    # Base term: C @ (D_z B) = C D_z B: (L, d, d)
    C_exp = C.unsqueeze(0).expand(L, -1, -1)  # (L, d, N)
    K = torch.bmm(C_exp, D_z_B)  # (L, d, d)

    # Precompute factors:
    # D_z P: (L, N, r)
    D_z_P = D_z.unsqueeze(2) * P.unsqueeze(0)

    # C D_z P: (L, d, r) - C @ diag(D_z) @ P
    C_D_z_P = torch.bmm(C_exp, D_z_P)

    # F = Q^* D_z P: (L, r, r) - the core r x r matrix
    Q_star = Q.conj().T  # (r, N)
    F = torch.einsum('rn,lnp->lrp', Q_star, D_z_P)

    # Q^* D_z B: (L, r, d)
    Q_D_z_B = torch.einsum('rn,lnd->lrd', Q_star, D_z_B)

    # Accumulate Neumann terms:
    # term_m = (C D_z P) @ F^{m-1} @ (Q^* D_z B) for m = 1, ..., k-1
    F_power = torch.eye(r, dtype=z.dtype, device=z.device).unsqueeze(0).expand(L, -1, -1).clone()

    for m in range(1, k):
        term = torch.bmm(torch.bmm(C_D_z_P, F_power), Q_D_z_B)
        K = K + term
        F_power = torch.bmm(F_power, F)

    return K


def compute_ssm_kernel_neumann_full(
    z: torch.Tensor,          # (L,) complex frequencies
    Lambda: torch.Tensor,     # (N,) complex eigenvalues
    P: torch.Tensor,          # (N, r) complex
    Q: torch.Tensor,          # (N, r) complex
    B: torch.Tensor,          # (N, d) complex input projection
    C: torch.Tensor,          # (d, N) complex output projection
    k: int = 4,               # truncation order
) -> torch.Tensor:
    """
    Compute SSM kernel via full resolvent matrix (for verification).

    Returns:
        K: (L, d, d) complex kernel values
    """
    R = neumann_resolvent(z, Lambda, P, Q, k=k)
    C_exp = C.unsqueeze(0).expand(R.shape[0], -1, -1)
    B_exp = B.unsqueeze(0).expand(R.shape[0], -1, -1)
    K = torch.bmm(torch.bmm(C_exp, R), B_exp)
    return K


def compute_spectral_radius(
    z: torch.Tensor,          # (L,) complex frequencies
    Lambda: torch.Tensor,     # (N,) complex eigenvalues
    P: torch.Tensor,          # (N, r) complex
    Q: torch.Tensor,          # (N, r) complex
) -> torch.Tensor:
    """
    Compute spectral radius of E_z = D_z P Q^* for each frequency.

    Since E_z has rank r, its nonzero eigenvalues are those of F = Q^* D_z P (r x r).

    Returns:
        rho: (L,) spectral radius for each frequency
    """
    D_z = 1.0 / (z.unsqueeze(1) - Lambda.unsqueeze(0))  # (L, N)
    D_z_P = D_z.unsqueeze(2) * P.unsqueeze(0)  # (L, N, r)
    Q_star = Q.conj().T
    F = torch.einsum('rn,lnp->lrp', Q_star, D_z_P)  # (L, r, r)

    # Spectral radius = max |eigenvalue| of F
    eigenvalues = torch.linalg.eigvals(F)  # (L, r)
    rho = eigenvalues.abs().max(dim=1).values  # (L,)

    return rho
