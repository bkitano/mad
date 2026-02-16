"""
MVE 008: Cyclic Reduction vs Prefix Scan for Dense SSM Recurrences

Kernel benchmark comparing two algorithms for h_t = A_t h_{t-1} + b_t.

Success criteria (Proposal 026):
1. GEMM count: CR uses ~log(T)x fewer GEMMs than prefix scan
2. Numerical accuracy: relative inf-norm error < 1e-5
3. Wall-clock: CR >= 2x faster at n=32, T=1024
4. Scaling: Speedup increases with T
"""
import torch
import time
import math
import yaml
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.prefix_scan import prefix_scan, sequential_scan
from models.cyclic_reduction import cyclic_reduction


def gen_data(T, n, device="cpu", dtype=torch.float64, sr=0.95):
    A = torch.randn(T, n, n, device=device, dtype=dtype) * (sr / math.sqrt(n))
    b = torch.randn(T, n, device=device, dtype=dtype)
    return A, b


def test_accuracy(T, n, device="cpu", dtype=torch.float64):
    A, b = gen_data(T, n, device=device, dtype=dtype)
    h_seq = sequential_scan(A, b)
    h_scan, sg = prefix_scan(A, b)
    h_cr, cg = cyclic_reduction(A, b)
    d = max(h_seq.norm(float('inf')).item(), 1e-10)
    return {
        'T': T, 'n': n, 'scan_gemms': sg, 'cr_gemms': cg,
        'scan_err': (h_scan - h_seq).norm(float('inf')).item() / d,
        'cr_err': (h_cr - h_seq).norm(float('inf')).item() / d,
        'cr_scan_err': (h_cr - h_scan).norm(float('inf')).item() / d,
    }


def test_speed(T, n, trials=10, warmup=3, device="cpu"):
    A, b = gen_data(T, n, device=device, dtype=torch.float32)
    for _ in range(warmup):
        prefix_scan(A, b)
        cyclic_reduction(A, b)
    if device != "cpu":
        torch.cuda.synchronize()

    def timeit(fn):
        ts = []
        for _ in range(trials):
            if device != "cpu":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn(A, b)
            if device != "cpu":
                torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return sum(ts) / len(ts)

    sm = timeit(prefix_scan)
    cm = timeit(cyclic_reduction)
    qm = timeit(sequential_scan)
    return {
        'T': T, 'n': n,
        'scan_ms': sm * 1000, 'cr_ms': cm * 1000, 'seq_ms': qm * 1000,
        'cr_vs_scan': sm / cm if cm > 0 else float('inf'),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    cp = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    if os.path.exists(cp):
        with open(cp) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {'device': 'cpu', 'dtype': 'float64', 'n_values': [32],
               'T_values': [64, 128, 256, 512, 1024], 'num_trials': 10, 'warmup': 3}

    dev = cfg.get('device', 'cpu')
    dt = {'float32': torch.float32, 'float64': torch.float64}[cfg.get('dtype', 'float64')]
    ns = cfg.get('n_values', [32])
    Ts = cfg.get('T_values', [64, 128, 256, 512, 1024])
    trials = cfg.get('num_trials', 10)
    warm = cfg.get('warmup', 3)

    print("=" * 80)
    print("MVE 008: Cyclic Reduction vs Prefix Scan for Dense SSM Recurrences")
    print("=" * 80)
    print(f"Device: {dev}, n: {ns}, T: {Ts}\n")

    # Test 1: GEMM Count
    print("TEST 1: GEMM Count")
    print(f"{'T':>6} {'n':>4} {'Scan':>10} {'CR':>10} {'Ratio':>8} {'log2T':>7}")
    print("-" * 50)
    acc_results = []
    for n in ns:
        for T in Ts:
            r = test_accuracy(T, n, device=dev, dtype=dt)
            ratio = r['scan_gemms'] / max(r['cr_gemms'], 1)
            print(f"{T:>6} {n:>4} {r['scan_gemms']:>10} {r['cr_gemms']:>10} "
                  f"{ratio:>7.2f}x {math.log2(T):>6.1f}x")
            acc_results.append(r)
    print()

    # Test 2: Accuracy
    print("TEST 2: Numerical Accuracy (float64)")
    print(f"{'T':>6} {'n':>4} {'ScanErr':>12} {'CRErr':>12} {'CR-Scan':>12} {'OK':>5}")
    print("-" * 55)
    for r in acc_results:
        ok = r['cr_err'] < 1e-5
        print(f"{r['T']:>6} {r['n']:>4} {r['scan_err']:>12.2e} "
              f"{r['cr_err']:>12.2e} {r['cr_scan_err']:>12.2e} "
              f"{'PASS' if ok else 'FAIL':>5}")
    print()

    # Test 3: Wall-clock
    print("TEST 3: Wall-Clock Time (float32)")
    print(f"{'T':>6} {'n':>4} {'Scan(ms)':>10} {'CR(ms)':>10} "
          f"{'Seq(ms)':>10} {'CR/Scan':>9}")
    print("-" * 52)
    wc = []
    for n in ns:
        for T in Ts:
            r = test_speed(T, n, trials=trials, warmup=warm, device=dev)
            print(f"{T:>6} {n:>4} {r['scan_ms']:>10.2f} {r['cr_ms']:>10.2f} "
                  f"{r['seq_ms']:>10.2f} {r['cr_vs_scan']:>8.2f}x")
            wc.append(r)
    print()

    # Test 4: Scaling
    print("TEST 4: Scaling")
    for n in ns:
        print(f"n={n}:")
        prev = None
        for r in wc:
            if r['n'] != n:
                continue
            s = r['cr_vs_scan']
            t = ""
            if prev is not None:
                t = " (+)" if s > prev else " (-)"
            prev = s
            print(f"  T={r['T']:>5}: {s:.2f}x{t}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    pg = next((r for r in acc_results if r['T'] == 1024 and r['n'] == 32), acc_results[-1])
    pw = next((r for r in wc if r['T'] == 1024 and r['n'] == 32), wc[-1])
    print(f"Primary: T={pg['T']}, n={pg['n']}\n")

    gr = pg['scan_gemms'] / max(pg['cr_gemms'], 1)
    lt = math.log2(pg['T'])
    c1 = gr >= 0.8 * lt
    print(f"1. GEMM Ratio: {gr:.2f}x (expect ~{lt:.1f}x) {'PASS' if c1 else 'FAIL'}")

    c2 = pg['cr_err'] < 1e-5
    print(f"2. Accuracy: {pg['cr_err']:.2e} (<1e-5) {'PASS' if c2 else 'FAIL'}")

    sp = pw['cr_vs_scan']
    c3 = sp >= 2.0
    print(f"3. Wallclock: {sp:.2f}x (>=2.0x) {'PASS' if c3 else 'FAIL'}")
    print(f"   Scan={pw['scan_ms']:.2f}ms CR={pw['cr_ms']:.2f}ms Seq={pw['seq_ms']:.2f}ms")

    sps = sorted([(r['T'], r['cr_vs_scan']) for r in wc if r['n'] == ns[0]])
    c4 = len(sps) >= 2 and sps[-1][1] > sps[0][1]
    print(f"4. Scaling: {sps[0][1]:.2f}x->{sps[-1][1]:.2f}x {'PASS' if c4 else 'FAIL'}")

    tot = sum([c1, c2, c3, c4])
    dec = "PROCEED" if tot >= 3 else ("DEBUG" if tot >= 2 else "ABANDON")
    print(f"\nRESULT: {tot}/4 passed | DECISION: {dec}")
    print("=" * 80)

    return {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4, 'decision': dec,
            'acc': acc_results, 'wc': wc}


if __name__ == "__main__":
    main()
