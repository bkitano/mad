# MVE 029: Circulant FAVOR+ for Linear Attention

## Hypothesis

Replacing the dense random projection matrix in FAVOR+ with a learnable circulant projection (from the CBE trick) will reduce the feature-map computation from O(md) to O(d log d) while preserving softmax kernel approximation quality.

## Task

**Associative Recall**: Given key-value pairs (k1,v1,...,k8,v8,SEP,kq), output the value vq corresponding to the queried key.

- 5K training sequences, 1K test
- Sequence length 64, vocabulary size 16
- 8 key-value pairs per sequence

## Models Compared

1. **Dense FAVOR+**: Standard FAVOR+ with dense 32x32 random projection (O(d^2))
2. **C-FAVOR+**: Circulant random features via FFT (O(d log d))
3. **ReLU Linear Attention**: No feature map, just ReLU(Q) @ ReLU(K)^T
4. **Softmax Attention**: Standard softmax (quality ceiling)

## Success Criteria

- C-FAVOR+ > 90% accuracy (matching dense FAVOR+)
- C-FAVOR+ feature map faster than dense FAVOR+
- Both FAVOR+ variants > ReLU by 20%+

## Running

### Via Modal (recommended):
```bash
modal run --detach modal_config.py --config config.yaml
```

### Locally (CPU, slower):
```bash
pip install -e .
python train.py --config config.yaml
```

## Files

- `models/attention.py` - All four attention mechanisms
- `data/generate.py` - Associative recall dataset
- `train.py` - Training and evaluation script
- `config.yaml` - Experiment configuration
- `modal_config.py` - Modal cloud deployment
