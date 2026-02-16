"""
Neumann-Cayley Orthogonal SSM (NC-SSM)

From proposal 028-neumann-cayley-input-dependent-orthogonal-ssm.

Key idea: Replace the exact O(n^3) matrix inversion in the Cayley transform with
a k-term Neumann series approximation, enabling input-dependent near-orthogonal
state transitions at O(kn^2) per token.

The Cayley transform maps skew-symmetric A to orthogonal W:
  W = (I + A)^{-1} (I - A)     [exact, O(n^3)]

Neumann approximation:
  (I + A)^{-1} ~ S_k(-A) = sum_{j=0}^{k-1} (-A)^j

For k=4, using radix-2 binary splitting (proposal Step 2):
  S_4(-A) = (I - A)(I + A^2)   [2 GEMMs]
  W_tilde = S_4(-A) * (I - A)  [1 GEMM]
  Total: 3 GEMMs for a near-orthogonal transition.

Orthogonality guarantee (proposal Lemma):
  ||W_tilde^T W_tilde - I||_2 <= 2*rho^k/(1-rho) + rho^{2k}/(1-rho)^2
  For rho_max=0.3, k=4: deviation < 0.017 (< 2%)

State update (proposal Step 3):
  h_t = W_tilde(x_t) * h_{t-1} + B * x_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NeumannCayleySSM(nn.Module):
    """
    Single-layer Neumann-Cayley SSM.

    Given input x_t in R^d:
    1. Project to skew-symmetric parameters: v_t = W_proj * x_t in R^{n(n-1)/2}
    2. Construct skew-symmetric A(x_t) in R^{n x n}
    3. Scale: A_scaled = rho_max * A / max(||A||_2, rho_max)
    4. Neumann-Cayley: W_tilde = S_k(-A) * (I - A)
    5. State update: h_t = W_tilde * h_{t-1} + B * x_t

    Args:
        d_model: Input/output dimension
        n: State dimension (small, e.g., 8)
        k: Neumann truncation order (default 4)
        rho_max: Spectral radius bound (default 0.3)
        n_power_iter: Number of power iterations for spectral norm (default 2)
    """

    def __init__(
        self,
        d_model: int,
        n: int,
        k: int = 4,
        rho_max: float = 0.3,
        n_power_iter: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.k = k
        self.rho_max = rho_max
        self.n_power_iter = n_power_iter

        # Number of upper-triangular entries in n x n skew-symmetric matrix
        self.n_skew_params = n * (n - 1) // 2

        # --- Step 1 (proposal): Input -> skew-symmetric parameters ---
        # v_t = W_proj * x_t in R^{n(n-1)/2}
        self.W_proj = nn.Linear(d_model, self.n_skew_params)

        # --- Input projection B: R^d -> R^n ---
        self.B = nn.Linear(d_model, n, bias=False)

        # --- Output projection C: R^n -> R^d ---
        self.C = nn.Linear(n, d_model, bias=False)

        # --- Direct feedthrough D ---
        self.D = nn.Parameter(torch.zeros(d_model))

        # Pre-compute upper-triangular indices for constructing skew-symmetric matrix
        self.register_buffer(
            'triu_row', torch.triu_indices(n, n, offset=1)[0]
        )
        self.register_buffer(
            'triu_col', torch.triu_indices(n, n, offset=1)[1]
        )

        self._init_params()

    def _init_params(self):
        """Initialize for stable training."""
        # Small init for W_proj so initial A has small spectral norm
        nn.init.xavier_uniform_(self.W_proj.weight, gain=0.1)
        nn.init.zeros_(self.W_proj.bias)

        # Standard init for B, C
        nn.init.xavier_uniform_(self.B.weight, gain=0.5)
        nn.init.xavier_uniform_(self.C.weight, gain=0.5)

    def _build_skew_symmetric(self, v: torch.Tensor) -> torch.Tensor:
        """
        Build skew-symmetric matrix from upper-triangular entries.

        Proposal Step 1:
          A(x_t)_{ij} = v^{idx(i,j)}  if i < j
          A(x_t)_{ij} = -v^{idx(j,i)} if i > j
          A(x_t)_{ij} = 0              if i = j

        Args:
            v: (batch, n_skew_params) upper-triangular entries

        Returns:
            A: (batch, n, n) skew-symmetric matrix
        """
        batch_size = v.shape[0]
        A = torch.zeros(batch_size, self.n, self.n, device=v.device, dtype=v.dtype)

        # Fill upper triangle
        A[:, self.triu_row, self.triu_col] = v
        # Fill lower triangle (negative of upper)
        A[:, self.triu_col, self.triu_row] = -v

        return A

    def _spectral_norm_scale(self, A: torch.Tensor) -> torch.Tensor:
        """
        Scale A so that ||A||_2 <= rho_max.

        Proposal Step 1:
          A_scaled = rho_max * A / max(||A||_2, rho_max)

        Uses power iteration to estimate ||A||_2.

        Args:
            A: (batch, n, n) skew-symmetric matrix

        Returns:
            A_scaled: (batch, n, n) with ||A_scaled||_2 <= rho_max
        """
        batch_size = A.shape[0]

        # Power iteration for spectral norm estimation
        # Initialize random vector
        u = torch.randn(batch_size, self.n, 1, device=A.device, dtype=A.dtype)
        u = F.normalize(u, dim=1)

        with torch.no_grad():
            for _ in range(self.n_power_iter):
                # v = A^T u / ||A^T u||
                v = torch.bmm(A.transpose(1, 2), u)
                v = F.normalize(v, dim=1)
                # u = A v / ||A v||
                u = torch.bmm(A, v)
                u = F.normalize(u, dim=1)

        # sigma = u^T A v
        sigma = torch.bmm(u.transpose(1, 2), torch.bmm(A, v)).squeeze(-1).squeeze(-1)
        sigma = sigma.abs()  # (batch,)

        # Scale: A_scaled = rho_max * A / max(sigma, rho_max)
        scale = self.rho_max / torch.clamp(sigma, min=self.rho_max)  # (batch,)
        A_scaled = A * scale.view(batch_size, 1, 1)

        return A_scaled

    def _neumann_cayley(self, A: torch.Tensor) -> torch.Tensor:
        """
        Compute approximate Cayley transform via Neumann series.

        Proposal Step 2:
          W_tilde_k = S_k(-A) * (I - A)

        For k=4, using radix-2 binary splitting:
          A2 = A @ A                    [1 GEMM]
          S4 = (I - A) @ (I + A2)      [1 GEMM]
          W  = S4 @ (I - A)            [1 GEMM]

        General case (k arbitrary):
          S_k = sum_{j=0}^{k-1} (-A)^j
          W   = S_k @ (I - A)

        Args:
            A: (batch, n, n) scaled skew-symmetric matrix with ||A||_2 <= rho_max

        Returns:
            W: (batch, n, n) near-orthogonal transition matrix
        """
        batch_size = A.shape[0]
        I = torch.eye(self.n, device=A.device, dtype=A.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        I_minus_A = I - A

        if self.k == 4:
            # Efficient radix-2 binary splitting (proposal):
            # S4 = (I - A)(I + A^2)
            A2 = torch.bmm(A, A)                          # 1 GEMM
            S4 = torch.bmm(I_minus_A, I + A2)             # 1 GEMM
            W = torch.bmm(S4, I_minus_A)                  # 1 GEMM
        elif self.k == 2:
            # S2 = I - A
            # W = (I - A)(I - A) = (I - A)^2
            W = torch.bmm(I_minus_A, I_minus_A)           # 1 GEMM
        else:
            # General case: accumulate S_k = sum_{j=0}^{k-1} (-A)^j
            S = I.clone()
            neg_A_power = I.clone()
            for j in range(1, self.k):
                neg_A_power = torch.bmm(neg_A_power, -A)  # (-A)^j
                S = S + neg_A_power
            W = torch.bmm(S, I_minus_A)

        return W

    def _compute_orthogonality_deviation(self, W: torch.Tensor) -> torch.Tensor:
        """
        Compute ||W^T W - I||_F for monitoring.

        Args:
            W: (batch, n, n) transition matrix

        Returns:
            deviation: scalar, mean Frobenius norm of W^T W - I
        """
        I = torch.eye(self.n, device=W.device, dtype=W.dtype)
        WtW = torch.bmm(W.transpose(1, 2), W)
        deviation = (WtW - I).norm(dim=(1, 2)).mean()
        return deviation

    def forward(self, u: torch.Tensor) -> tuple:
        """
        Forward pass with sequential recurrence.

        Proposal Step 3:
          h_t = W_tilde(x_t) * h_{t-1} + B * x_t

        Args:
            u: (batch, seq_len, d_model) input

        Returns:
            y: (batch, seq_len, d_model) output
            ortho_dev: scalar, mean orthogonality deviation throughout sequence
        """
        batch_size, seq_len, _ = u.shape
        device = u.device
        dtype = u.dtype

        # Initialize hidden state
        h = torch.zeros(batch_size, self.n, device=device, dtype=dtype)

        outputs = []
        ortho_devs = []

        for t in range(seq_len):
            u_t = u[:, t, :]  # (batch, d_model)

            # Step 1: Input -> skew-symmetric parameters
            v_t = self.W_proj(u_t)  # (batch, n_skew_params)

            # Step 2: Build skew-symmetric matrix A(x_t)
            A_t = self._build_skew_symmetric(v_t)  # (batch, n, n)

            # Step 3: Scale to ensure ||A||_2 <= rho_max
            A_t = self._spectral_norm_scale(A_t)

            # Step 4: Neumann-Cayley approximate orthogonal W
            W_t = self._neumann_cayley(A_t)  # (batch, n, n)

            # Monitor orthogonality (for success criteria)
            with torch.no_grad():
                ortho_devs.append(self._compute_orthogonality_deviation(W_t))

            # Step 5: State update h_t = W_t @ h_{t-1} + B @ u_t
            h = torch.bmm(W_t, h.unsqueeze(-1)).squeeze(-1) + self.B(u_t)

            # Output: y_t = C @ h_t + D * u_t
            y_t = self.C(h) + u_t * self.D

            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        mean_ortho_dev = torch.stack(ortho_devs).mean()

        return y, mean_ortho_dev


class NCSSMClassifier(nn.Module):
    """
    NC-SSM wrapped for classification (S5 permutation composition task).

    Architecture:
      Embedding -> [NC-SSM -> LayerNorm + Residual] x num_layers -> MLP -> logits
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n: int,
        num_classes: int,
        num_layers: int = 2,
        k: int = 4,
        rho_max: float = 0.3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            NeumannCayleySSM(d_model, n, k=k, rho_max=rho_max)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        # MLP head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, tokens: torch.Tensor) -> tuple:
        """
        Args:
            tokens: (batch, seq_len) integer token indices

        Returns:
            logits: (batch, seq_len, num_classes)
            mean_ortho_dev: scalar, mean orthogonality deviation across layers
        """
        x = self.embedding(tokens)
        x = self.dropout(x)

        all_ortho_devs = []
        for ssm, norm in zip(self.layers, self.norms):
            residual = x
            ssm_out, ortho_dev = ssm(norm(x))
            x = residual + ssm_out
            all_ortho_devs.append(ortho_dev)

        logits = self.head(x)
        mean_ortho_dev = torch.stack(all_ortho_devs).mean()

        return logits, mean_ortho_dev
