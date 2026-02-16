import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
from models.cc_ssm import CCSSMModel
from models.diagonal_ssm import DiagonalSSMModel
from data.generate import DelayedCopyDataset

IGNORE_INDEX = -100

print('=== CC-SSM Smoke Test ===')
model = CCSSMModel(vocab_size=10, num_classes=8, max_seq_len=100, d_model=64, state_dim=32, num_layers=2)
tokens = torch.randint(0, 10, (4, 61))
logits = model(tokens)
print(f'  Input: {tokens.shape}, Output: {logits.shape}')
print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')
layer = model.layers[0].ssm
omega = layer._build_skew_circulant_eigenvalues()
lambdas = layer._cayley_eigenvalues(omega)
mags = lambdas.abs()
print(f'  |lambda|: min={mags.min():.6f}, max={mags.max():.6f}')
assert torch.allclose(mags, torch.ones_like(mags), atol=1e-6), "FAIL: eigenvalues not on unit circle"

print('\n=== DiagSSM Smoke Test ===')
model2 = DiagonalSSMModel(vocab_size=10, num_classes=8, max_seq_len=100, d_model=64, state_dim=32, num_layers=2)
logits2 = model2(tokens)
print(f'  Input: {tokens.shape}, Output: {logits2.shape}')
print(f'  Params: {sum(p.numel() for p in model2.parameters()):,}')

print('\n=== Data Test ===')
ds = DelayedCopyDataset(num_samples=10, vocab_size=8, k=5, delay=50)
tok, tgt = ds[0]
print(f'  Seq len: {ds.seq_len}, targets: {(tgt != IGNORE_INDEX).sum().item()}')

print('\n=== Backward Pass ===')
criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
logits = model(tokens)
targets = torch.full((4, 61), IGNORE_INDEX, dtype=torch.long)
targets[:, 56:61] = torch.randint(0, 8, (4, 5))
loss = criterion(logits.view(-1, 8), targets.view(-1))
loss.backward()
has_nan = any(torch.isnan(p.grad).any().item() for p in model.parameters() if p.grad is not None)
print(f'  Loss: {loss.item():.4f}, NaN in grads: {has_nan}')
assert not has_nan, "FAIL: NaN in gradients"

print('\nAll smoke tests PASSED!')
