# MVE 008: Cyclic Reduction vs Prefix Scan for Dense SSM Recurrences

**Proposal**: 026-cyclic-reduction-randmscan-ssm-recurrence

Kernel benchmark comparing cyclic reduction O(Tn^3) vs prefix scan O(Tn^3 log T).

```bash
pip install torch numpy pyyaml
python train.py --config config.yaml
```
