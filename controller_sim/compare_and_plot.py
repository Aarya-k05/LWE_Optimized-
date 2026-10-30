"""
============================================================
COMPARE PLAINTEXT vs LWE-ENCRYPTED CONTROLLER
============================================================

Runs both controllers and produces the same set of deliverables as the CKKS
project, plus the LWE-specific files that the previous LWE code produced:

    results/
      --- same names / formats as the CKKS project -------------------------
        control_data.csv         all 18 states, plain vs encrypted, u, errors, timing
        execution_trace.csv      per-step sensor -> controller -> actuator trace
        state_response.png       first / middle / last mass
        control_signal.png       u(k) plaintext vs LWE
        error_vs_step.png        ||x_plain - x_lwe||_2   (log)
        timing_breakdown.png     encrypt / evaluate / decrypt (+ offline)
        summary.txt              numerical summary (CKKS section layout)
      --- same names as the previous LWE code ------------------------------
        lwe_stream_results.csv   (same columns, new noise columns appended at the end)
        lwe_stream_summary.txt
        lwe_stream_four_graphs.png / .pdf
      --- noise + comparison (separate scripts, run at the end) -------------
        lwe_relative_noise.png / _results.csv / _summary.txt   (relative_noise.py)
        ckks_vs_lwe_comparison.csv / ckks_vs_lwe_four_graphs.png / .pdf
                                                                (compare_ckks_lwe.py)

Usage:
    python compare_and_plot.py                    # optimized profile (default)
    python compare_and_plot.py --profile legacy   # reproduces the previous LWE code exactly
    python compare_and_plot.py --skip-extras      # skip relative-noise + CKKS comparison
============================================================
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from state_space_model import build_plant, design_state_feedback, discretize, N_MASSES
from plaintext_sim import simulate_plaintext
from encrypted_sim import simulate_encrypted
from lwe_core import PROFILES, run_self_checks

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_csv(path, headers, rows):
    """Write a simple CSV file that can be opened in Excel."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="optimized", choices=["optimized", "legacy"],
                    help="optimized = wide modulus, low noise (default); "
                         "legacy = previous LWE code, bit-identical")
    ap.add_argument("--skip-extras", action="store_true",
                    help="do not run relative_noise.py / compare_ckks_lwe.py afterwards")
    args = ap.parse_args(argv)
    P = PROFILES[args.profile]

    # ---------------------------------------------------------
    # BUILD PLANT + CONTROLLER   (identical to the CKKS project)
    # ---------------------------------------------------------
    A, B, C, D = build_plant()
    K = design_state_feedback(A, B)
    n = A.shape[0]
    dt = 0.1
    Ad, Bd, Cd, Dd = discretize(A, B, C, D, dt=dt)

    x0 = np.zeros(n)
    x0[2 * (N_MASSES - 1)] = 1.0   # displace the last mass by 1m, rest at 0
    steps = 200
    t = np.arange(steps + 1) * dt

    print("=" * 72)
    print(f"STATE-SPACE MODEL  ({N_MASSES} masses -> {n} states)")
    print("=" * 72)
    print("State feedback gain K:\n", K)

    # ---------------------------------------------------------
    # PLAINTEXT BASELINE
    # ---------------------------------------------------------
    print("\n" + "=" * 72)
    print("1) PLAINTEXT BASELINE")
    print("=" * 72)
    X_plain, U_plain = simulate_plaintext(Ad, Bd, K, x0, steps)
    print(f"Initial state : {X_plain[0]}")
    print(f"Final state   : {X_plain[-1]}")
    print(f"Max |u|       : {np.max(np.abs(U_plain)):.9f}")

    # ---------------------------------------------------------
    # ENCRYPTED RUN
    # ---------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"2) LWE ENCRYPTED CONTROLLER  (profile: {P.name}, q=2^{P.q_bits}-1, "
          f"scale=2^{P.scale_bits}, gain=2^{P.gain_bits}, sigma={P.sigma})")
    print("=" * 72)
    checks = run_self_checks(P, K)
    X_enc, U_enc, timings, trace = simulate_encrypted(Ad, Bd, K, x0, steps, P, verbose=True)

    # ---------------------------------------------------------
    # ACCURACY
    # ---------------------------------------------------------
    state_error = np.linalg.norm(X_plain - X_enc, axis=1)
    control_error = np.abs(U_plain - U_enc)
    max_state_error = float(np.max(state_error))
    max_control_error = float(np.max(control_error))
    state_valid = all(r["state_ok"] for r in trace)
    control_valid = all(r["control_ok"] for r in trace)
    overall_ok = state_valid and control_valid and all(checks.values())

    enc = np.array(timings["encrypt"]); ev = np.array(timings["evaluate"])
    dec = np.array(timings["decrypt"]); off = np.array(timings["offline"])
    crit = enc + ev + dec
    fused = np.array(timings["fused"])

    print("\n" + "=" * 72)
    print("3) PLAINTEXT vs LWE ACCURACY")
    print("=" * 72)
    print(f"Final plaintext state  : {X_plain[-1]}")
    print(f"Final encrypted state  : {X_enc[-1]}")
    print(f"Maximum state error    : {max_state_error:.6e}")
    print(f"Maximum control error  : {max_control_error:.6e}")
    print(f"Critical-path latency  : {crit.mean()*1e3:.3f} us/step (3 dispatches), "
          f"{fused.mean()*1e3:.3f} us/step fused;  offline {off.mean()*1e3:.3f} us/step")
    print("\nInterpretation:")
    print("The LWE controller is exact modular arithmetic; its only error is the")
    print("injected integer noise e (and fixed-point rounding), divided by the scale.")
    print("Self-checks:", ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in checks.items()))

    # ---------------------------------------------------------
    # SAVE ACTUAL NUMERICAL DATA  (CKKS-format CSVs)
    # ---------------------------------------------------------
    state_headers = []
    for i in range(N_MASSES):
        state_headers += [f"pos{i+1}_plain", f"vel{i+1}_plain",
                          f"pos{i+1}_encrypted", f"vel{i+1}_encrypted"]
    control_rows = []
    for k in range(steps):
        tr = trace[k]
        row = [k, f"{t[k]:.6f}"]
        for i in range(N_MASSES):
            row += [f"{X_plain[k, 2*i]:.12f}", f"{X_plain[k, 2*i+1]:.12f}",
                    f"{X_enc[k, 2*i]:.12f}", f"{X_enc[k, 2*i+1]:.12f}"]
        row += [f"{U_plain[k]:.12f}", f"{U_enc[k]:.12f}",
                f"{state_error[k]:.12e}", f"{control_error[k]:.12e}",
                f"{tr['encrypt_ms']:.6f}", f"{tr['evaluate_ms']:.6f}", f"{tr['decrypt_ms']:.6f}",
                tr["enc_x_bytes"], 0]
        control_rows.append(row)
    save_csv(os.path.join(RESULTS_DIR, "control_data.csv"),
             ["step", "time_s"] + state_headers + [
                 "u_plain", "u_encrypted", "state_error", "control_error",
                 "encrypt_ms", "he_dot_ms", "decrypt_ms",
                 "enc_x_serialized_bytes", "enc_u_serialized_bytes"], control_rows)

    trace_state_headers = [f"x{i+1}" for i in range(n)]
    trace_rows = []
    for tr in trace:
        row = [tr["step"], f"{tr['time_s']:.6f}"]
        row += [f"{v:.12f}" for v in tr["state_before"]]
        row += [f"{tr['u_encrypted']:.12f}", f"{tr['u_plain_check']:.12f}",
                f"{tr['control_error']:.12e}"]
        row += [f"{v:.12f}" for v in tr["state_after"]]
        row += [f"{tr['state_change_norm']:.12e}", f"{tr['encrypt_ms']:.6f}",
                f"{tr['evaluate_ms']:.6f}", f"{tr['decrypt_ms']:.6f}",
                f"{tr['offline_ms']:.6f}", tr["enc_x_bytes"], 0,
                f"{tr['state_noise_l2']:.12e}", f"{tr['state_noise_linf']:.12e}",
                f"{tr['control_noise']:.12e}", tr["e_linf"]]
        trace_rows.append(row)
    save_csv(os.path.join(RESULTS_DIR, "execution_trace.csv"),
             (["step", "time_s"] + [f"sensor_{h}" for h in trace_state_headers]
              + ["u_from_encrypted_controller", "u_plain_validation", "control_error"]
              + [f"next_{h}" for h in trace_state_headers]
              + ["state_change_norm", "encrypt_ms", "he_dot_ms", "decrypt_ms", "offline_ms",
                 "enc_x_serialized_bytes", "enc_u_serialized_bytes",
                 "state_noise_l2", "state_noise_linf", "control_noise_abs", "e_linf"]),
             trace_rows)

    # ---- legacy-named LWE results (same schema as before + noise columns at the end) ----
    legacy_rows = []
    for k in range(steps):
        tr = trace[k]
        r = {"step": k, "time_s": k * dt, "u_plain": U_plain[k], "u_lwe": U_enc[k],
             "state_error_l2": state_error[k], "control_error_abs": control_error[k],
             "encrypt_ms": tr["encrypt_ms"], "evaluation_ms": tr["evaluate_ms"],
             "recovery_ms": tr["decrypt_ms"],
             "total_ms": tr["encrypt_ms"] + tr["evaluate_ms"] + tr["decrypt_ms"],
             "state_recovery_pass": tr["state_ok"], "control_recovery_pass": tr["control_ok"],
             "enc_x_serialized_bytes": tr["enc_x_bytes"]}
        r.update({f"x_plain_{i}": float(v) for i, v in enumerate(X_plain[k])})
        r.update({f"x_lwe_{i}": float(v) for i, v in enumerate(X_enc[k])})
        r.update({"offline_ms": tr["offline_ms"],
                  "total_with_offline_ms": r["total_ms"] + tr["offline_ms"],
                  "state_noise_l2": tr["state_noise_l2"],
                  "state_noise_linf": tr["state_noise_linf"],
                  "control_noise_abs": abs(tr["control_noise"]), "e_linf": tr["e_linf"]})
        legacy_rows.append(r)
    with open(os.path.join(RESULTS_DIR, "lwe_stream_results.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(legacy_rows[0]))
        w.writeheader()
        w.writerows(legacy_rows)

    # ---------------------------------------------------------
    # PLOTS  (CKKS-format figures)
    # ---------------------------------------------------------
    rep_indices = sorted(set([0, N_MASSES // 2, N_MASSES - 1]))
    rep_labels = {0: "Mass 1 (fixed end)", N_MASSES - 1: "Mass N (actuated end)"}
    fig, axs = plt.subplots(len(rep_indices), 1, figsize=(8, 3 * len(rep_indices)), sharex=True)
    for ax, i in zip(axs, rep_indices):
        ax.plot(t, X_plain[:, 2*i], label="Plaintext controller", lw=2)
        ax.plot(t, X_enc[:, 2*i], "--", label="Encrypted controller (LWE)", lw=2)
        ax.set_ylabel("Position (m)")
        ax.set_title(rep_labels.get(i, f"Mass {i+1} (middle)"))
        ax.legend(); ax.grid(True)
    axs[-1].set_xlabel("Time (s)")
    fig.suptitle("Closed-Loop State Response (representative masses along the chain)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "state_response.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t[:-1], U_plain, label="Plaintext u(k)", lw=2)
    ax.plot(t[:-1], U_enc, "--", label="Encrypted u(k)", lw=2)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Control input u(k)")
    ax.set_title("Control Signal: Plaintext vs LWE")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "control_signal.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(t, state_error + 1e-16)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State error, ||x_plain - x_enc|| (log scale)")
    ax.set_title(f"LWE Noise-Induced Error (full {n}-state vector norm)")
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "error_vs_step.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ops = ["encrypt", "evaluate", "decrypt", "offline\n(off critical path)"]
    means = [enc.mean(), ev.mean(), dec.mean(), off.mean()]
    bars = ax.bar(ops, means, color=["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5"])
    bars[3].set_hatch("//")
    ax.set_ylabel("Average time per control step (ms)")
    ax.set_title(f"LWE Controller Timing Breakdown ({n} states)")
    for i, v in enumerate(means):
        ax.text(i, v, f"{v:.5f} ms", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "timing_breakdown.png"), dpi=150)
    plt.close(fig)

    # ---- legacy four-graph figure (same panels as before) ----
    ct_kb = P.ciphertext_bytes / 1024.0
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle("LWE custom stream-cipher arithmetic — "
                 + ("VALIDATION FAILED" if not overall_ok else "recovery passed; comparison provisional"),
                 fontsize=15, fontweight="bold", color="darkred" if not overall_ok else "darkgreen")
    ax = axes[0, 0]
    labels = ["Encrypt", "Eval/control\n(not HE)", "Decrypt/\nrecovery", "TOTAL\n(critical path)"]
    vals = [enc.mean(), ev.mean(), dec.mean(), crit.mean()]
    bars = ax.bar(labels, vals, color=["#4472C4", "#ED7D31", "#70AD47", "#5B9BD5"])
    ax.set_title(f"LWE Stream Cipher — Timing (average of {steps} steps)")
    ax.set_ylabel("Milliseconds per step"); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(top=max(vals) * 1.3)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.6g}", (b.get_x() + b.get_width()/2, b.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    ax.text(0.99, 0.97, f"Offline pad pre-compute: {off.mean():.6g} ms/step (not on critical path)\n"
            "Setup/key generation excluded; secret-key recurrence cancellation.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax = axes[0, 1]
    bars = ax.bar(["State ciphertext\nto controller", "Encrypted control/result\nto actuator"],
                  [ct_kb, 0], color=["#4472C4", "#A5A5A5"])
    ax.set_title("Ciphertext Size"); ax.set_ylabel("Serialized size (KB; 1024 bytes/KB)")
    ax.grid(axis="y", alpha=0.3)
    ax.annotate(f"{ct_kb:.6g} KB", (bars[0].get_x() + bars[0].get_width()/2, ct_kb),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
    ax.text(bars[1].get_x() + bars[1].get_width()/2, max(ct_kb*0.08, 1e-5), "N/A",
            ha="center", va="bottom", fontsize=9)
    ax = axes[1, 0]
    bar = ax.bar(["Maximum |u_LWE - u_plain|"], [max_control_error], color="#C00000")[0]
    ax.set_title("Maximum Control Error"); ax.set_ylabel("Absolute error"); ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log"); ax.set_ylim(bottom=max_control_error / 10.0)
    ax.annotate(f"{max_control_error:.9g}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    ax.text(0.98, 0.96, f"Maximum state error (L2): {max_state_error:.9g}\n"
            f"State recovery: {'PASS' if state_valid else 'FAIL'}; "
            f"control recovery: {'PASS' if control_valid else 'FAIL'}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"})
    ax = axes[1, 1]
    sel = slice(19, 200)
    ks = np.arange(steps)
    ax.plot(ks[sel], U_plain[sel], "-o", markersize=3, linewidth=1, label="Plaintext")
    ax.plot(ks[sel], U_enc[sel], "--s", markersize=3, linewidth=1,
            label=f"LWE report equation (Q{P.gain_bits} gain encoding)")
    ax.set_title("Control Signal After Transient"); ax.set_xlabel("Control step")
    ax.set_ylabel("Control input u(k)"); ax.grid(True, alpha=0.35); ax.legend()
    fig.savefig(os.path.join(RESULTS_DIR, "lwe_stream_four_graphs.png"), dpi=300)
    fig.savefig(os.path.join(RESULTS_DIR, "lwe_stream_four_graphs.pdf"))
    plt.close(fig)

    # ---------------------------------------------------------
    # SUMMARIES
    # ---------------------------------------------------------
    cn = np.abs([r["control_noise"] for r in trace])
    sn = np.array([r["state_noise_linf"] for r in trace])
    half = steps // 2
    pm = lambda b: "PASS" if b else "FAIL"
    cfg = (f"Modulus q: 2^{P.q_bits}-1 = {P.q}\n"
           f"Secret matrix: 18x18, uniform in [0, q//{P.secret_div})\n"
           f"Noise: {'exact discrete Gaussian (CDT)' if P.sampler == 'cdt' else 'rounded Box-Muller'}, "
           f"sigma = {P.sigma:g}, seed {P.seed}\n"
           f"State scale: 2^{P.scale_bits};  gain scale: 2^{P.gain_bits}  "
           f"(control scale 2^{P.scale_bits + P.gain_bits})\n"
           f"Recoverable |u| range without wrap-around: +-{P.max_control:.6g}  "
           f"(measured max |u| = {np.max(np.abs(U_enc)):.6g})\n"
           f"Ciphertext: {P.ciphertext_bytes} bytes per state vector (18 x "
           f"{8 if P.q_bits > 32 else 4}-byte words)\n"
           "Recurrence: V[k] = S.V[k-1] + m[k] + e[k] (mod q), V[-1] = 0\n"
           "Control: F.V[k] - (F.S).V[k-1] (mod q), centred decode")
    timing_txt = (f"Encrypt (online):           {enc.mean():.6f} ms/step\n"
                  f"Evaluate:                   {ev.mean():.6f} ms/step\n"
                  f"Decrypt/recover:            {dec.mean():.6f} ms/step\n"
                  f"Critical path (3 calls):    {crit.mean():.6f} ms/step  (median {np.median(crit):.6f})\n"
                  f"Critical path (fused call): {fused.mean():.6f} ms/step\n"
                  f"Offline pre-compute:        {off.mean():.6f} ms/step  (idle part of the 100 ms period)\n"
                  f"Pure compute, critical path (dispatch excluded): {timings['kernel_online_ms'][0]:.6f} ms/step\n"
                  f"Pure compute, everything incl. offline:          {timings['kernel_full_ms'][0]:.6f} ms/step\n"
                  f"Total work per step:        {crit.mean() + off.mean():.6f} ms/step")
    noise_txt = (f"State noise  ||decoded - true||_inf : max {sn.max():.4e}, mean {sn.mean():.4e}\n"
                 f"Control noise |u_lwe - (-Kx)| same state: max {cn.max():.4e}, rms {np.sqrt(np.mean(cn**2)):.4e}\n"
                 f"Noise growth check (rms 2nd half / rms 1st half): control "
                 f"{np.sqrt(np.mean(cn[half:]**2)) / np.sqrt(np.mean(cn[:half]**2)):.3f}, "
                 f"state {np.sqrt(np.mean(sn[half:]**2)) / np.sqrt(np.mean(sn[:half]**2)):.3f}"
                 "  (~1 => no accumulation => no bootstrapping needed)")
    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\nLWE RECURSIVE STREAM-CIPHER STATE-FEEDBACK CONTROL SYSTEM\n" + "=" * 72 + "\n\n")
        f.write("1. PLANT\n")
        f.write(f"{N_MASSES}-mass chain: mass 1 anchored to a wall, mass {N_MASSES}\n"
                "driven by a single actuator; identical spring/damper between\n"
                "each neighbouring pair (m=1 kg, c=1.0 Ns/m, k=2 N/m).\n"
                f"State vector (n={n}): [pos1, vel1, ..., pos{N_MASSES}, vel{N_MASSES}]\n\n")
        f.write("2. CONTROLLER\nControl law: u(k) = -K x(k)   (LQR, same K as the CKKS project)\n"
                f"K =\n{np.array2string(K, precision=4, max_line_width=100)}\n\n")
        f.write(f"3. DISCRETIZATION\nSampling time: {dt} s (ZOH)\nAd shape: {Ad.shape}, Bd shape: {Bd.shape}\n\n")
        f.write(f"4. LWE CONFIGURATION  (profile: {P.name})\n{cfg}\n\n")
        f.write(f"5. SIMULATION\nControl cycles: {steps}\nInitial state: {x0}\n"
                f"Final plaintext state: {X_plain[-1]}\nFinal encrypted-controller state: {X_enc[-1]}\n\n")
        f.write(f"6. ACCURACY\nMaximum state error (full {n}-dim vector norm): {max_state_error:.12e}\n"
                f"Maximum control error: {max_control_error:.12e}\n\n")
        f.write(f"7. TIMING\n{timing_txt}\n\n8. NOISE (absolute)\n{noise_txt}\n\n")
        f.write("9. VALIDATION\n" + "\n".join(f"{k}: {pm(v)}" for k, v in checks.items())
                + f"\nEvery step state recovered within noise bound: {pm(state_valid)}"
                f"\nEvery step control inside the non-wrapping range: {pm(control_valid)}"
                f"\nOverall: {pm(overall_ok)}\n\n")
        f.write("10. FILES\ncontrol_data.csv, execution_trace.csv, state_response.png, "
                "control_signal.png, error_vs_step.png, timing_breakdown.png, "
                "lwe_stream_*.{csv,txt,png,pdf}, lwe_relative_noise.*, ckks_vs_lwe_*\n\n"
                "No security claim: custom secret-key recurrence, n=18, not standard HE.\n")

    with open(os.path.join(RESULTS_DIR, "lwe_stream_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Fresh LWE-based recursive stream-cipher simulation\n"
                "===================================================\n"
                f"State dimension: {n} (expected 18)\nControl steps: {steps} (expected 200)\n"
                f"Input gain K: {np.array2string(K, precision=12, separator=', ')}\n\n"
                f"Cryptographic construction\n--------------------------\n{cfg}\n\n"
                f"Measured per-step timing (mean over {steps} recorded steps)\n"
                "--------------------------------------------------------\n"
                f"{timing_txt}\n"
                "Setup/key generation (RNG + S + F@S precompute, excluded from per-step): "
                f"{timings['setup_ms'][0]:.9g} ms.\n\n"
                f"Serialized sizes\n-----------------\nState ciphertext payload: "
                f"{P.ciphertext_bytes} bytes ({ct_kb:.9g} KB). No network transfer is implemented.\n\n"
                "Numerical validation\n--------------------\n"
                + "\n".join(f"{k}: {pm(v)}" for k, v in checks.items())
                + f"\nState recovery every step: {pm(state_valid)}\n"
                f"Control recovery every step: {pm(control_valid)}\nOverall recovery validation: {pm(overall_ok)}\n"
                f"Maximum state error max_k ||x_LWE-x_plain||_2: {max_state_error:.12g}\n"
                f"Maximum control error max_k |u_LWE-u_plain|: {max_control_error:.12g}\n\n"
                f"Noise\n-----\n{noise_txt}\n\n"
                "Caveat\n------\nStandard LWE / homomorphic-encryption security: NOT CLAIMED. "
                "Evaluation uses secret-key modular recurrence cancellation (F.V[k] - (F.S).V[k-1]).\n")

    print(f"\nResults written to {os.path.abspath(RESULTS_DIR)}")

    if not args.skip_extras:
        import relative_noise
        relative_noise.main()
        try:
            import compare_ckks_lwe
            compare_ckks_lwe.main()
        except Exception as exc:     # comparison is optional
            print(f"[compare_ckks_lwe skipped: {exc}]")


if __name__ == "__main__":
    main()
