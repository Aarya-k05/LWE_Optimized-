"""
============================================================
ABLATION BENCHMARK: which optimisation buys what
============================================================

Same idea as benchmark.py in the CKKS project.  Every row runs the full 200-step
closed loop and reports the median over --repeats runs.

    1 original            pure-Python big-int loops, sigma = sqrt(18)       (first code)
    2 + sigma = 1         lower injected noise                              (noise)
    3 + NumPy int64       vectorised matrix-vector products                 (latency)
    4 + Numba kernels     compiled, allocation-free kernels                 (latency)
    5 + offline split     pad = S.V[k-1] and (F.S).V[k-1] moved OFF the critical path (latency)
    6 + wide modulus      q = 2^61-1, scale 2^28, CDT noise  == DEFAULT     (NOISE)

Columns
    enc/eval/dec   mean per-call time of the three online phases (includes Python->Numba
                   call overhead, ~1 us per call, the floor of any Python timer)
    call crit.     enc+eval+dec as measured per call
    compute crit.  pure compute on the critical path, dispatch EXCLUDED (a whole replay
                   inside one compiled call); for rows 1-3 (no compiled code) = call crit.
    total          compute crit. + offline work   (total CPU work per step)
    max u err / max x err     max |u_lwe-u_plain| and max ||x_lwe-x_plain||_2
    ct bytes       serialised state ciphertext

Usage:  python benchmark.py [--repeats 5]
============================================================
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from state_space_model import build_plant, design_state_feedback, discretize, N_MASSES
from plaintext_sim import simulate_plaintext
from encrypted_sim import simulate_encrypted
from lwe_core import PROFILES

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

ROWS = [  # (label, profile, backend, split_offline)
    ("1 original (python ints, sigma=sqrt18)", "original", "python", False),
    ("2 + sigma=1", "legacy", "python", False),
    ("3 + NumPy int64", "legacy", "numpy", False),
    ("4 + Numba kernels", "legacy", "numba", False),
    ("5 + offline pad/FS split", "legacy", "numba", True),
    ("6 + wide modulus 2^61-1 (DEFAULT)", "optimized", "numba", True),
]


def run_row(profile, backend, split, Ad, Bd, K, x0, Xp, Up, steps):
    P = PROFILES[profile]
    X, U, t, tr = simulate_encrypted(Ad, Bd, K, x0, steps, P, backend, split, verbose=False)
    enc, ev, dec = (np.mean(t[k]) * 1e3 for k in ("encrypt", "evaluate", "decrypt"))
    call_crit = enc + ev + dec                                       # us
    if backend == "numba":
        k_on, k_full = t["kernel_online_ms"][0] * 1e3, t["kernel_full_ms"][0] * 1e3
        compute_crit = k_on if split else k_full
        total = k_full
    else:
        compute_crit, total = call_crit, call_crit
    return dict(enc=enc, ev=ev, dec=dec, call_crit=call_crit, compute_crit=compute_crit,
                total=total, max_u=float(np.max(np.abs(U - Up))),
                max_x=float(np.max(np.linalg.norm(X - Xp, axis=1))), ct=P.ciphertext_bytes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    A, B, C, D = build_plant()
    K = design_state_feedback(A, B)
    Ad, Bd, _, _ = discretize(A, B, C, D, dt=0.1)
    x0 = np.zeros(A.shape[0]); x0[2 * (N_MASSES - 1)] = 1.0
    steps = 200
    Xp, Up = simulate_plaintext(Ad, Bd, K, x0, steps)

    results = []
    for label, prof, be, sp in ROWS:
        print(f"running {label} ...")
        runs = [run_row(prof, be, sp, Ad, Bd, K, x0, Xp, Up, steps) for _ in range(args.repeats)]
        results.append((label, {k: float(np.median([r[k] for r in runs])) for k in runs[0]}))

    base = results[0][1]["compute_crit"]
    header = ["configuration", "enc us", "eval us", "dec us", "call crit us", "compute crit us",
              "total us", "speed-up", "max u err", "max x err", "ct bytes"]
    table = []
    for label, r in results:
        table.append([label, f"{r['enc']:.3f}", f"{r['ev']:.3f}", f"{r['dec']:.3f}",
                      f"{r['call_crit']:.3f}", f"{r['compute_crit']:.3f}", f"{r['total']:.3f}",
                      f"{base / r['compute_crit']:.1f}x", f"{r['max_u']:.2e}",
                      f"{r['max_x']:.2e}", int(r["ct"])])

    with open(os.path.join(RESULTS_DIR, "benchmark_ablation.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(table)
    with open(os.path.join(RESULTS_DIR, "benchmark_ablation.md"), "w", encoding="utf-8") as f:
        f.write(f"LWE ablation, 18 states, 200 steps, median of {args.repeats} runs\n\n")
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for row in table:
            row = list(row); row[5] = f"**{row[5]}**"
            f.write("| " + " | ".join(str(c) for c in row) + " |\n")

    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    names = [r[0].split(" (")[0] for r in results]
    ax = axs[0]
    ax.bar(range(len(results)), [r[1]["compute_crit"] for r in results], color="#4472C4")
    ax.set_yscale("log"); ax.set_ylabel("critical-path compute per step (us, log)")
    ax.set_title("Latency (dispatch excluded)")
    ax = axs[1]
    ax.bar(range(len(results)), [r[1]["max_u"] for r in results], color="#C00000")
    ax.set_yscale("log"); ax.set_ylabel("max |u_lwe - u_plain| (log)")
    ax.set_title("Control error (noise)")
    for a in axs:
        a.set_xticks(range(len(results))); a.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        a.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(RESULTS_DIR, "benchmark_ablation.png"), dpi=150)
    plt.close(fig)

    print(open(os.path.join(RESULTS_DIR, "benchmark_ablation.md"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
