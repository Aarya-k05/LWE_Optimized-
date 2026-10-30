#!/usr/bin/env python3
"""
============================================================
ABSOLUTE AND RELATIVE NOISE ANALYSIS (LWE vs CKKS)
============================================================

Consumes results/lwe_stream_results.csv produced by compare_and_plot.py, so the
trajectories are exactly the ones in the main run (same seed, same results).
If the file is missing it runs the simulation first.  CKKS is scored with the
IDENTICAL formulas from reference/ckks_results/control_data.csv.

Two kinds of "error" are tracked for both state and control:
  * trajectory error : LWE closed loop vs the plaintext closed loop (as in CKKS)
  * cipher noise     : decoded value vs the true value on the SAME state (LWE only)

>>> THE RELATIVE-NOISE CRITERIA ARE ALL IN `metrics()` AND `FLOOR_FRACTION` BELOW.
>>> Change those two places and every number and the graph follow.

Why several "relative" definitions?  The plant is a regulator: x and u decay
toward zero, while the injected noise floor does not. Dividing each step's
error by that step's (tiny) signal therefore blows up late in the run for ANY
scheme. So we report four views:

  peak-relative     max_k err_k / max_k |signal_k|          (headline, stable)
  horizon-relative  ||err||_2 / ||signal||_2 over all steps  (energy ratio)
  per-step, floored max_k err_k / max(|signal_k|, 1e-3*peak) (worst step)
  per-step, raw     max_k err_k / |signal_k|                 (ill-conditioned;
                    shown only for transparency, with the step it occurs at)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "results"
RESULTS_CSV = OUTPUT_DIR / "lwe_stream_results.csv"
CKKS_CSV = ROOT / "reference" / "ckks_results" / "control_data.csv"
FLOOR_FRACTION = 1e-3
N_STATES = 18


def load_lwe() -> dict[str, np.ndarray]:
    if not RESULTS_CSV.exists():
        import compare_and_plot
        compare_and_plot.main(["--skip-extras"])
    with RESULTS_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "u_plain": np.array([float(r["u_plain"]) for r in rows]),
        "u_lwe": np.array([float(r["u_lwe"]) for r in rows]),
        "x_plain": np.array(
            [[float(r[f"x_plain_{i}"]) for i in range(N_STATES)] for r in rows]
        ),
        "x_lwe": np.array(
            [[float(r[f"x_lwe_{i}"]) for i in range(N_STATES)] for r in rows]
        ),
        "state_noise": np.array([float(r["state_noise_l2"]) for r in rows]),
        "control_noise": np.abs([float(r["control_noise_abs"]) for r in rows]),
    }


def load_ckks_errors() -> dict[str, np.ndarray] | None:
    """Recorded CKKS per-step errors (same plant / plaintext baseline as the LWE run)."""
    if not CKKS_CSV.exists():
        return None
    with CKKS_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "control": np.array([float(r["control_error"]) for r in rows]),
        "state": np.array([float(r["state_error"]) for r in rows]),
    }


def metrics(err: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    """err and signal are non-negative magnitudes per step (L2 norm or |.|)."""
    peak = float(np.max(signal))
    floor = FLOOR_FRACTION * peak
    raw = err / np.maximum(signal, np.finfo(float).tiny)
    floored = err / np.maximum(signal, floor)
    return {
        "abs_max": float(np.max(err)),
        "abs_rms": float(np.sqrt(np.mean(err**2))),
        "signal_peak": peak,
        "rel_peak": float(np.max(err) / peak),
        "rel_horizon": float(np.linalg.norm(err) / np.linalg.norm(signal)),
        "rel_step_floored_max": float(np.max(floored)),
        "rel_step_floored_median": float(np.median(floored)),
        "rel_step_raw_max": float(np.max(raw)),
        "rel_step_raw_max_at": int(np.argmax(raw)),
    }


def fmt(name: str, m: dict[str, float]) -> list[str]:
    return [
        f"  {name}",
        f"    absolute  max={m['abs_max']:.4e}   rms={m['abs_rms']:.4e}   (signal peak {m['signal_peak']:.4e})",
        f"    relative  peak-rel={m['rel_peak']:.4e}   horizon-rel={m['rel_horizon']:.4e}",
        f"              per-step floored: max={m['rel_step_floored_max']:.4e}  median={m['rel_step_floored_median']:.4e}",
        f"              per-step raw max={m['rel_step_raw_max']:.4e} at step {m['rel_step_raw_max_at']} (ill-conditioned)",
    ]


def main() -> None:
    d = load_lwe()
    x_norm = np.linalg.norm(d["x_plain"], axis=1)
    u_abs = np.abs(d["u_plain"])
    lwe_state_err = np.linalg.norm(d["x_lwe"] - d["x_plain"], axis=1)
    lwe_ctrl_err = np.abs(d["u_lwe"] - d["u_plain"])

    results = {
        "LWE state": metrics(lwe_state_err, x_norm),
        "LWE control": metrics(lwe_ctrl_err, u_abs),
        "LWE state (cipher noise only)": metrics(d["state_noise"], x_norm),
        "LWE control (cipher noise only)": metrics(d["control_noise"], u_abs),
    }
    ckks = load_ckks_errors()
    if ckks is not None and len(ckks["state"]) == len(x_norm):
        results["CKKS state"] = metrics(ckks["state"], x_norm)
        results["CKKS control"] = metrics(ckks["control"], u_abs)

    # Per-step CSV (absolute + relative, floored and raw).
    floor_x = FLOOR_FRACTION * x_norm.max()
    floor_u = FLOOR_FRACTION * u_abs.max()
    with (OUTPUT_DIR / "lwe_relative_noise_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        cols = [
            "step", "state_norm_plain", "state_error_abs", "state_error_rel_floored",
            "state_error_rel_raw", "u_plain", "control_error_abs",
            "control_error_rel_floored", "control_error_rel_raw",
        ]
        w = csv.writer(handle)
        w.writerow(cols)
        for k in range(len(x_norm)):
            w.writerow([
                k, x_norm[k], lwe_state_err[k],
                lwe_state_err[k] / max(x_norm[k], floor_x),
                lwe_state_err[k] / max(x_norm[k], np.finfo(float).tiny),
                d["u_plain"][k], lwe_ctrl_err[k],
                lwe_ctrl_err[k] / max(u_abs[k], floor_u),
                lwe_ctrl_err[k] / max(u_abs[k], np.finfo(float).tiny),
            ])

    # Summary.
    lines = [
        "LWE stream cipher - absolute and relative noise",
        "================================================",
        f"Steps: {len(x_norm)}   floor = {FLOOR_FRACTION:g} x peak signal",
        "State error = ||x_scheme - x_plain||_2 ; signal = ||x_plain||_2",
        "Control error = |u_scheme - u_plain| ; signal = |u_plain|",
        "",
    ]
    for name, m in results.items():
        lines += fmt(name, m)
        lines.append("")
    if "CKKS state" in results:
        lines.append("LWE / CKKS ratio (>1 means LWE noisier)")
        for kind in ("state", "control"):
            a, b = results[f"LWE {kind}"], results[f"CKKS {kind}"]
            lines.append(
                f"  {kind}: abs-max x{a['abs_max'] / b['abs_max']:.3g}   "
                f"peak-rel x{a['rel_peak'] / b['rel_peak']:.3g}   "
                f"horizon-rel x{a['rel_horizon'] / b['rel_horizon']:.3g}"
            )
        lines.append("")
    lines += [
        "Note: absolute noise comes from integer noise injected every step (F.e / 2^(s+g) for",
        "control, e / 2^s per state entry), so it does not decay while the",
        "signal does. Per-step relative error therefore rises as the regulator",
        "converges; peak-relative and horizon-relative are the stable figures.",
        "Same plaintext baseline is assumed for CKKS (verified by",
        "compare_ckks_lwe.py). No security claim: custom scheme, not standard HE.",
    ]
    (OUTPUT_DIR / "lwe_relative_noise_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    # Figure.
    t = np.arange(len(x_norm)) * 0.1
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    eps = 1e-16   # display floor: step 0 has exactly zero state error (same x0)
    panels = [
        (ax[0, 0], "State: absolute error", lwe_state_err, ckks["state"] if ckks else None, "||dx||_2"),
        (ax[0, 1], "State: relative error (floored)", lwe_state_err / np.maximum(x_norm, floor_x),
         ckks["state"] / np.maximum(x_norm, floor_x) if ckks else None, "||dx|| / max(||x||, floor)"),
        (ax[1, 0], "Control: absolute error", lwe_ctrl_err, ckks["control"] if ckks else None, "|du|"),
        (ax[1, 1], "Control: relative error (floored)", lwe_ctrl_err / np.maximum(u_abs, floor_u),
         ckks["control"] / np.maximum(u_abs, floor_u) if ckks else None, "|du| / max(|u|, floor)"),
    ]
    for a, title, lwe_v, ckks_v, ylabel in panels:
        a.semilogy(t, np.maximum(lwe_v, eps), label="LWE stream cipher")
        if ckks_v is not None:
            a.semilogy(t, np.maximum(ckks_v, eps), label="CKKS")
        a.set_title(title)
        a.set_xlabel("time (s)")
        a.set_ylabel(ylabel)
        a.grid(True, alpha=0.3, which="both")
        a.legend()
    fig.suptitle("Absolute vs relative noise (relative uses a 1e-3 x peak floor)", fontweight="bold")
    fig.savefig(OUTPUT_DIR / "lwe_relative_noise.png", dpi=200)
    plt.close(fig)

    print("\n".join(lines))


if __name__ == "__main__":
    main()
