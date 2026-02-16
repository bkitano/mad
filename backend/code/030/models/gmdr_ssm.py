"""
Group-Matrix Displacement Rank SSM (GM-DR-SSM)

From proposal 030: Parameterizes SSM state transitions as low displacement rank
group matrices for the hyperoctahedral group B_n = Z_2^n x S_n.

Key equations (from proposal):
    h_t = A(x_t) h_{t-1} + B x_t

    A(x_t) = sum_{g in N} alpha_g(x_t) B_g   +  sum_{i=1}^{r} diag(a_i(x_t)) B_{g_i}
             |------- group conv kernel -------|  |------- displacement perturbation ------|

    alpha_g(x_t) = softmax(x_t @ W_alpha)      (kernel weights, |N| scalars)
    a_i(x_t) = tanh(x_t @ W_a_i)               (perturbation vectors, r vectors of dim n)

Variables:
    n = state dimension (p = block size, single block for MVE)
    r = displacement rank (controls deviation from exact equivariance)
    |N| = kernel neighborhood size (identity + generators)
    B_g = signed permutation matrix (group diagonal for g in B_n)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import permutations
from typing import Optional


def make_B4_group_diagonals() -> torch.Tensor:
    """Precompute all 384 group diagonal matrices for B_4.

    B_4 = Z_2^4 x S_4 has |B_4| = 2^4 * 4! = 384 elements.
    Each element is a 4x4 signed permutation matrix.

    Returns: (384, 4, 4) tensor of signed permutation matrices.
    """
    diags = []
    for perm in permutations(range(4)):
        for signs in range(2**4):
            mat = torch.zeros(4, 4)
            for i, j in enumerate(perm):
                sign = 1.0 if (signs >> i) & 1 == 0 else -1.0
                mat[i, j] = sign
            diags.append(mat)
    return torch.stack(diags)  # (384, 4, 4)


def select_kernel_neighborhood(group_diags: torch.Tensor, kernel_size: int = 3):
    """Select kernel neighborhood N subset of B_4.

    We pick:
        idx 0 = identity matrix (I_4)
        idx 1 = transposition of first two elements (swap 0<->1)
        idx 2 = sign flip of first element

    These generate a large subgroup and capture both permutation and sign structure.

    For kernel_size > 3, we add more generators.
    """
    n = group_diags.shape[1]

    # Find identity
    identity_idx = None
    for i in range(group_diags.shape[0]):
        if torch.allclose(group_diags[i], torch.eye(n)):
            identity_idx = i
            break

    # Find swap(0,1): permutation (1,0,2,3) with all positive signs
    swap01_idx = None
    target_swap = torch.zeros(n, n)
    target_swap[0, 1] = 1.0
    target_swap[1, 0] = 1.0
    for i in range(2, n):
        target_swap[i, i] = 1.0
    for i in range(group_diags.shape[0]):
        if torch.allclose(group_diags[i], target_swap):
            swap01_idx = i
            break

    # Find sign flip of coord 0: identity with mat[0,0] = -1
    signflip0_idx = None
    target_flip = torch.eye(n)
    target_flip[0, 0] = -1.0
    for i in range(group_diags.shape[0]):
        if torch.allclose(group_diags[i], target_flip):
            signflip0_idx = i
            break

    indices = [identity_idx, swap01_idx, signflip0_idx]

    # Add more generators if kernel_size > 3
    if kernel_size > 3:
        # swap(1,2): permutation (0,2,1,3)
        target_swap12 = torch.zeros(n, n)
        target_swap12[0, 0] = 1.0
        target_swap12[1, 2] = 1.0
        target_swap12[2, 1] = 1.0
        target_swap12[3, 3] = 1.0
        for i in range(group_diags.shape[0]):
            if torch.allclose(group_diags[i], target_swap12):
                indices.append(i)
                break
    if kernel_size > 4:
        # sign flip of coord 1
        target_flip1 = torch.eye(n)
        target_flip1[1, 1] = -1.0
        for i in range(group_diags.shape[0]):
            if torch.allclose(group_diags[i], target_flip1):
                indices.append(i)
                break

    return indices[:kernel_size]


def select_anchor_elements(
    group_diags: torch.Tensor, r: int, kernel_indices: list
) -> list:
    """Select r anchor group elements for displacement perturbation.

    We pick elements NOT in the kernel neighborhood, with diverse structure.
    Use cyclic permutation (0->1->2->3->0) and various sign patterns.
    """
    n = group_diags.shape[1]
    anchors = []

    # Candidate: 3-cycle (0->1->2, 3 fixed), all positive
    target_3cycle = torch.zeros(n, n)
    target_3cycle[0, 1] = 1.0
    target_3cycle[1, 2] = 1.0
    target_3cycle[2, 0] = 1.0
    if n > 3:
        target_3cycle[3, 3] = 1.0
    for i in range(group_diags.shape[0]):
        if i not in kernel_indices and torch.allclose(group_diags[i], target_3cycle):
            anchors.append(i)
            break

    # Candidate: 4-cycle (0->1->2->3->0), all positive
    target_4cycle = torch.zeros(n, n)
    target_4cycle[0, 1] = 1.0
    target_4cycle[1, 2] = 1.0
    target_4cycle[2, 3] = 1.0
    target_4cycle[3, 0] = 1.0
    for i in range(group_diags.shape[0]):
        if (
            i not in kernel_indices
            and i not in anchors
            and torch.allclose(group_diags[i], target_4cycle)
        ):
            anchors.append(i)
            break

    # Candidate: swap(0,1) with sign flip on coord 2
    target_swap_flip = torch.zeros(n, n)
    target_swap_flip[0, 1] = 1.0
    target_swap_flip[1, 0] = 1.0
    target_swap_flip[2, 2] = -1.0
    if n > 3:
        target_swap_flip[3, 3] = 1.0
    for i in range(group_diags.shape[0]):
        if (
            i not in kernel_indices
            and i not in anchors
            and torch.allclose(group_diags[i], target_swap_flip)
        ):
            anchors.append(i)
            break

    # Candidate: full reversal (0<->3, 1<->2), all positive
    target_rev = torch.zeros(n, n)
    target_rev[0, 3] = 1.0
    target_rev[1, 2] = 1.0
    target_rev[2, 1] = 1.0
    target_rev[3, 0] = 1.0
    for i in range(group_diags.shape[0]):
        if (
            i not in kernel_indices
            and i not in anchors
            and torch.allclose(group_diags[i], target_rev)
        ):
            anchors.append(i)
            break

    # Fill remaining with random group elements not yet used
    used = set(kernel_indices + anchors)
    for i in range(group_diags.shape[0]):
        if len(anchors) >= r:
            break
        if i not in used:
            anchors.append(i)

    return anchors[:r]


class GMDRSSMStep(nn.Module):
    """Single-step GM-DR-SSM state update.

    Computes h_{t+1} = A(x_t) h_t + B x_t

    where A(x_t) is a low displacement rank group matrix for B_4.

    Args:
        n: State dimension (must be 4 for B_4)
        kernel_size: Number of group elements in kernel neighborhood |N|
        disp_rank: Displacement rank r (0 = exact group conv, >0 = perturbation)
        d_in: Input feature dimension
        perturbation_scale: Scale factor for tanh perturbation (epsilon in proposal)
    """

    def __init__(
        self,
        n: int = 4,
        kernel_size: int = 3,
        disp_rank: int = 2,
        d_in: int = 32,
        perturbation_scale: float = 0.1,
    ):
        super().__init__()
        self.n = n
        self.r = disp_rank
        self.kernel_size = kernel_size
        self.perturbation_scale = perturbation_scale

        # Precompute all B_4 group diagonals (384, 4, 4)
        group_diags = make_B4_group_diagonals()

        # Select kernel neighborhood
        kernel_idx = select_kernel_neighborhood(group_diags, kernel_size)
        self.kernel_idx = kernel_idx

        # Register kernel matrices as buffer (not trained)
        # kernel_mats: (kernel_size, n, n)
        kernel_mats = group_diags[kernel_idx]
        self.register_buffer("kernel_mats", kernel_mats)

        # Input-dependent kernel weights: x -> softmax(W_alpha @ x)
        # Proposal eq: alpha_g(x_t) = sigma(x_t @ w_g)
        self.W_alpha = nn.Linear(d_in, kernel_size)

        # Displacement perturbation
        if disp_rank > 0:
            # Select anchor group elements for perturbation
            anchor_idx = select_anchor_elements(group_diags, disp_rank, kernel_idx)
            self.anchor_idx = anchor_idx

            # Register anchor matrices as buffer
            anchor_mats = group_diags[anchor_idx]
            self.register_buffer("anchor_mats", anchor_mats)

            # Input-dependent perturbation vectors: x -> tanh(W_a @ x) * epsilon
            # Proposal eq: a_i(x_t) = tanh(x_t @ W_{a_i})
            self.W_a = nn.Linear(d_in, disp_rank * n)

        # Input projection B: x_t -> B @ x_t (added to state)
        self.B_proj = nn.Linear(d_in, n)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (batch, n) current hidden state
            x: (batch, d_in) current input features

        Returns:
            h_new: (batch, n) updated hidden state
        """
        batch = h.shape[0]

        # --- Group convolution kernel ---
        # alpha: (batch, kernel_size), softmax ensures convex combination
        alpha = F.softmax(self.W_alpha(x), dim=-1)

        # A_group = sum_g alpha_g * B_g
        # kernel_mats: (kernel_size, n, n)
        # alpha: (batch, kernel_size) -> (batch, kernel_size, 1, 1)
        A = torch.einsum("bk,kij->bij", alpha, self.kernel_mats)  # (batch, n, n)

        # --- Displacement perturbation ---
        if self.r > 0:
            # a: (batch, r, n) perturbation vectors, scaled by epsilon
            a = (
                torch.tanh(self.W_a(x)).view(batch, self.r, self.n)
                * self.perturbation_scale
            )

            # Add diag(a_i) @ B_{g_i} for each perturbation direction
            # anchor_mats: (r, n, n)
            for i in range(self.r):
                # diag(a[:, i]) @ anchor_mats[i]
                # a[:, i]: (batch, n) -> (batch, n, 1) * (1, n, n)
                perturbation = a[:, i].unsqueeze(-1) * self.anchor_mats[i].unsqueeze(0)
                A = A + perturbation  # (batch, n, n)

        # --- State update ---
        # h_new = A @ h + B @ x
        h_new = torch.bmm(A, h.unsqueeze(-1)).squeeze(-1) + self.B_proj(x)

        return h_new


class GMDRSSMLayer(nn.Module):
    """Full GM-DR-SSM sequence layer with sequential scan.

    Processes a sequence of inputs and produces a sequence of hidden states.
    Uses sequential scan (adequate for MVE with short sequences).
    """

    def __init__(
        self,
        d_model: int,
        n: int = 4,
        kernel_size: int = 3,
        disp_rank: int = 2,
        perturbation_scale: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.step = GMDRSSMStep(
            n=n,
            kernel_size=kernel_size,
            disp_rank=disp_rank,
            d_in=d_model,
            perturbation_scale=perturbation_scale,
        )
        # Output projection: state -> model dim
        self.C_proj = nn.Linear(n, d_model)
        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) input sequence

        Returns:
            y: (batch, seq_len, d_model) output sequence
        """
        batch, seq_len, _ = x.shape

        # Initialize hidden state to zero
        h = torch.zeros(batch, self.n, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            h = self.step(h, x[:, t])
            y_t = self.C_proj(h)  # (batch, d_model)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)

        # Residual connection + LayerNorm
        return self.norm(x + y)


class GMDRSSMClassifier(nn.Module):
    """GM-DR-SSM classifier for group state tracking.

    Architecture:
        Embedding -> [GM-DR-SSM Layer] x L -> MLP Head -> Logits

    Args:
        vocab_size: Number of input tokens (group generators + special tokens)
        num_classes: Number of output classes (group elements)
        d_model: Hidden dimension
        n: State dimension (4 for B_4)
        num_layers: Number of GM-DR-SSM layers
        kernel_size: Size of group convolution kernel |N|
        disp_rank: Displacement rank r
        perturbation_scale: Scale of displacement perturbation epsilon
        max_seq_len: Maximum sequence length (for positional embedding)
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 32,
        n: int = 4,
        num_layers: int = 1,
        kernel_size: int = 3,
        disp_rank: int = 2,
        perturbation_scale: float = 0.1,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList(
            [
                GMDRSSMLayer(
                    d_model=d_model,
                    n=n,
                    kernel_size=kernel_size,
                    disp_rank=disp_rank,
                    perturbation_scale=perturbation_scale,
                )
                for _ in range(num_layers)
            ]
        )

        # MLP head for classification
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, num_classes),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (batch, seq_len) input token indices

        Returns:
            logits: (batch, seq_len, num_classes) classification logits
        """
        batch, seq_len = tokens.shape

        # Embedding
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        x = self.embedding(tokens) + self.pos_embedding(positions)

        # SSM layers
        for layer in self.layers:
            x = layer(x)

        # Classification head (per-position)
        logits = self.head(x)

        return logits
