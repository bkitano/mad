"""Monarch-Gated State Transition SSM - Proposal 006"""
import torch, torch.nn as nn, math

def cayley_transform(skew):
    S = skew - skew.transpose(-2, -1)
    I = torch.eye(S.shape[-1], device=S.device, dtype=S.dtype)
    return torch.linalg.solve(I + S, I - S)

class MonarchSSMLayer(nn.Module):
    def __init__(self, d_model, n):
        super().__init__()
        self.d_model, self.n = d_model, n
        self.b = int(math.isqrt(n))
        assert self.b * self.b == n
        indices = torch.arange(n)
        perm = (indices % self.b) * self.b + (indices // self.b)
        self.register_buffer('perm', perm)
        inv_perm = torch.zeros_like(perm); inv_perm[perm] = torch.arange(n)
        self.register_buffer('inv_perm', inv_perm)
        self.L_skew = nn.Parameter(torch.randn(self.b, self.b, self.b) * 0.1)
        self.R_skew = nn.Parameter(torch.randn(self.b, self.b, self.b) * 0.1)
        self.W_gate = nn.Linear(d_model, 2 * self.b)
        self.W_B = nn.Linear(d_model, n)
        self.C = nn.Linear(n, d_model, bias=False)
        self.D = nn.Parameter(torch.zeros(d_model))
        nn.init.zeros_(self.W_gate.weight); nn.init.constant_(self.W_gate.bias, 2.2)
        nn.init.xavier_normal_(self.W_B.weight, gain=0.1)
        nn.init.xavier_normal_(self.C.weight, gain=0.1)

    def forward(self, u):
        batch, T, _ = u.shape
        h = torch.zeros(batch, self.n, device=u.device, dtype=u.dtype)
        L = cayley_transform(self.L_skew)
        R = cayley_transform(self.R_skew)
        b = self.b
        outputs = []
        for t in range(T):
            u_t = u[:, t, :]
            gates = torch.sigmoid(self.W_gate(u_t))
            alpha, beta = gates[:, :b], gates[:, b:]
            hr = torch.einsum('bij,abj->abi', R, h.view(batch, b, b)) * beta.unsqueeze(-1)
            hp = hr.reshape(batch, self.n)[:, self.perm]
            hl = torch.einsum('bij,abj->abi', L, hp.view(batch, b, b)) * alpha.unsqueeze(-1)
            h = hl.reshape(batch, self.n)[:, self.inv_perm]
            h = h + self.W_B(u_t)
            outputs.append(self.C(h) + self.D * u_t)
        return torch.stack(outputs, dim=1)

class DiagonalSSMLayer(nn.Module):
    def __init__(self, d_model, n):
        super().__init__()
        self.n = n
        self.W_alpha = nn.Linear(d_model, n)
        self.W_B = nn.Linear(d_model, n)
        self.C = nn.Linear(n, d_model, bias=False)
        self.D = nn.Parameter(torch.zeros(d_model))
        nn.init.zeros_(self.W_alpha.weight); nn.init.constant_(self.W_alpha.bias, 2.2)
        nn.init.xavier_normal_(self.W_B.weight, gain=0.1)
        nn.init.xavier_normal_(self.C.weight, gain=0.1)

    def forward(self, u):
        batch, T, _ = u.shape
        h = torch.zeros(batch, self.n, device=u.device, dtype=u.dtype)
        outputs = []
        for t in range(T):
            u_t = u[:, t, :]
            h = torch.sigmoid(self.W_alpha(u_t)) * h + self.W_B(u_t)
            outputs.append(self.C(h) + self.D * u_t)
        return torch.stack(outputs, dim=1)

class DenseSSMLayer(nn.Module):
    def __init__(self, d_model, n):
        super().__init__()
        self.n = n
        self.W_d = nn.Linear(d_model, n)
        self.W_A = nn.Linear(d_model, n * n)
        self.W_B = nn.Linear(d_model, n)
        self.C = nn.Linear(n, d_model, bias=False)
        self.D = nn.Parameter(torch.zeros(d_model))
        nn.init.zeros_(self.W_d.weight); nn.init.constant_(self.W_d.bias, 2.2)
        nn.init.normal_(self.W_A.weight, 0, 0.01); nn.init.normal_(self.W_A.bias, 0, 1.0/n)
        nn.init.xavier_normal_(self.W_B.weight, gain=0.1)
        nn.init.xavier_normal_(self.C.weight, gain=0.1)

    def forward(self, u):
        batch, T, _ = u.shape
        h = torch.zeros(batch, self.n, device=u.device, dtype=u.dtype)
        outputs = []
        for t in range(T):
            u_t = u[:, t, :]
            d_t = torch.sigmoid(self.W_d(u_t))
            A = self.W_A(u_t).view(batch, self.n, self.n)
            A = torch.tanh(A) * (1.0 / math.sqrt(self.n))
            Ad = torch.diag_embed(d_t) + A - torch.diag_embed(A.diagonal(dim1=-2, dim2=-1))
            h = torch.bmm(Ad, h.unsqueeze(-1)).squeeze(-1) + self.W_B(u_t)
            outputs.append(self.C(h) + self.D * u_t)
        return torch.stack(outputs, dim=1)

class SSMClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, n, num_classes, num_layers=2, ssm_type='monarch', dropout=0.1):
        super().__init__()
        self.ssm_type = ssm_type
        self.embedding = nn.Embedding(vocab_size, d_model)
        C = {'monarch': MonarchSSMLayer, 'diagonal': DiagonalSSMLayer, 'dense': DenseSSMLayer}[ssm_type]
        self.layers = nn.ModuleList([C(d_model, n) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model*2),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model*2, num_classes))
    def forward(self, tokens):
        x = self.dropout(self.embedding(tokens))
        for layer, norm in zip(self.layers, self.norms): x = x + layer(norm(x))
        return self.head(x[:, -1, :])
