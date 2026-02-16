"""
Diagonal SSM baseline (S4D-style) for comparison with CC-SSM.
|lambda| <= 1 via sigmoid parameterization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiagonalSSMLayer(nn.Module):
    def __init__(self, d_model, state_dim):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.log_lambda = nn.Parameter(torch.randn(state_dim) * 0.5 + 3.0)
        self.B = nn.Linear(d_model, state_dim, bias=False)
        self.C = nn.Linear(state_dim, d_model, bias=False)
        self.D = nn.Parameter(torch.zeros(d_model))

    def forward(self, u):
        batch_size, seq_len, _ = u.shape
        lam = torch.sigmoid(self.log_lambda)
        Bu = self.B(u)
        h = torch.zeros(batch_size, self.state_dim, device=u.device, dtype=u.dtype)
        outputs = []
        for t in range(seq_len):
            h = lam * h + Bu[:, t]
            y_t = self.C(h) + self.D * u[:, t]
            outputs.append(y_t)
        return torch.stack(outputs, dim=1)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff or int(8 / 3 * d_model)
        d_ff = ((d_ff + 7) // 8) * 8
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class DiagonalSSMBlock(nn.Module):
    def __init__(self, d_model, state_dim, dropout=0.0):
        super().__init__()
        self.ssm_norm = RMSNorm(d_model)
        self.ssm = DiagonalSSMLayer(d_model, state_dim)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.ssm(self.ssm_norm(x)))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DiagonalSSMModel(nn.Module):
    def __init__(self, vocab_size, num_classes, max_seq_len, d_model=64, state_dim=32, num_layers=2, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([DiagonalSSMBlock(d_model, state_dim, dropout) for _ in range(num_layers)])
        self.final_norm = RMSNorm(d_model)
        self.output_head = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, tokens):
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embed(tokens) + self.pos_embed(positions)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.output_head(x)
