"""
============================================================
ENCRYPTED CLOSED-LOOP SIMULATION
LWE recursive stream-cipher state-feedback controller
============================================================

This file is the actual execution/demo of the encrypted controller.
(It plays the same role as encrypted_sim.py in the CKKS project.)

At every control step:

    1. The simulated plant has a real state x(k).
    2. SENSOR   : encrypts x(k):          V[k] = pad + round(x*2^s) + e[k]  (mod q)
    3. CONTROLLER: evaluates               r = F.V[k] - (F.S).V[k-1]        (mod q)
       using only ciphertexts (V[k], V[k-1]) and the public matrix F.S.
    4. ACTUATOR : decodes r -> u(k) ~= -K x(k).   Only u is used to drive the plant.
    5. The plant is advanced with that decoded u, and the loop repeats.
    6. OFF THE CRITICAL PATH, in the idle part of the period, the sensor/controller
       pre-compute pad = S.V[k] and F.S.V[k] for the NEXT step (see lwe_core.py, O1).

Timing columns (ms/step), same spirit as the CKKS "enc / HE / dec" columns:
    encrypt   step 2   (online part only: 18 additions mod q)
    evaluate  step 3   (18 modular multiplications)
    decrypt   step 4   (centred decode of u and of the 18 states)
    offline   step 6   (NOT on the critical path; reported separately)
    critical = encrypt + evaluate + decrypt        total = critical + offline

Noise columns (absolute; the relative version is built from these by
relative_noise.py):
    state_noise_l2/linf : decoded state (m+e)/2^s  minus the true state x(k)
    control_noise       : u_lwe(k) minus (-K.x(k)) evaluated on the SAME state
                          -> isolates cipher noise from closed-loop divergence
    e_linf              : largest injected integer noise this step

Important:
    Software proof-of-concept.  The plant is simulated in plaintext, the
    controller computation is the part done on ciphertext-side quantities.
    No security claim is made (see lwe_core.py).
============================================================
"""

import time

import numpy as np

from lwe_core import LWEStreamCipher, LWEParams, sample_noise_bank, N_STATES

_NS = 1e-6   # ns -> ms


def simulate_encrypted(Ad, Bd, K, x0, steps, params: LWEParams, backend="numba",
                       split_offline=True, verbose=True):
    """Run the closed loop through the LWE controller.

    split_offline=True  : pad / F.S.V pre-computation is off the critical path (default)
    split_offline=False : it is done inside the encrypt window (used by the ablation)

    Returns X_enc (steps+1, n), U_enc (steps,), timings (dict of lists, ms), trace (list).
    """
    n = Ad.shape[0]
    assert n == N_STATES
    t_setup = time.perf_counter_ns()
    cipher = LWEStreamCipher(params, K, backend)          # keygen + JIT load (setup)
    setup_ms = (time.perf_counter_ns() - t_setup) * _NS
    # Noise is state independent: sample every step's noise in one call, up front.
    noise_bank = sample_noise_bank(cipher.rng, steps, params)

    X = np.zeros((steps + 1, n))
    U = np.zeros(steps)
    X[0] = x0
    timings = {"encrypt": [], "evaluate": [], "decrypt": [], "offline": [], "fused": [],
               "setup_ms": [setup_ms]}
    trace = []
    Kf = np.asarray(K).reshape(n)

    if split_offline:
        cipher.precompute()          # V[-1] = 0 -> pad = 0 (setup)

    for k in range(steps):
        x = X[k]
        e = noise_bank[k]

        # ---- 2. SENSOR: encrypt -------------------------------------------
        t0 = time.perf_counter_ns()
        if not split_offline:
            cipher.precompute()
        cipher.encrypt(x, e)
        t1 = time.perf_counter_ns()
        # ---- 3. CONTROLLER: evaluate on ciphertexts ------------------------
        residue = cipher.evaluate()
        t2 = time.perf_counter_ns()
        # ---- 4. ACTUATOR: decrypt ------------------------------------------
        state_hat, u = cipher.recover(residue)
        t3 = time.perf_counter_ns()
        state_hat = state_hat.copy()
        ct_bytes = len(cipher.serialize())          # outside the timers
        ct_copy = cipher.ct.copy()

        # ---- 6. OFFLINE: next step's pad / F.S.V ---------------------------
        cipher.advance()
        t4 = time.perf_counter_ns()
        if split_offline:
            cipher.precompute()
        t5 = time.perf_counter_ns()

        # ---- bookkeeping / validation (never timed) ------------------------
        u_plain_same_state = float(-Kf @ x)
        unwrapped = sum(int(a) * int(b) for a, b in zip(
            cipher.gain_int, np.rint(x * params.scale).astype(np.int64) + e))
        control_ok = abs(unwrapped) <= params.q // 2 and \
            abs(unwrapped - residue_to_signed(residue, params.q)) == 0
        state_noise = state_hat - x
        X[k + 1] = Ad @ x + Bd[:, 0] * u
        U[k] = u

        enc_ms = (t1 - t0) * _NS
        ev_ms = (t2 - t1) * _NS
        dec_ms = (t3 - t2) * _NS
        off_ms = (t5 - t4) * _NS if split_offline else 0.0
        if not split_offline:                       # precompute lived in encrypt window
            off_ms = 0.0
        for key, val in (("encrypt", enc_ms), ("evaluate", ev_ms),
                         ("decrypt", dec_ms), ("offline", off_ms)):
            timings[key].append(val)

        trace.append({
            "step": k, "time_s": k * 0.1,
            "state_before": x.copy(), "state_after": X[k + 1].copy(),
            "u_encrypted": u, "u_plain_check": u_plain_same_state,
            "control_error": abs(u - u_plain_same_state),
            "state_change_norm": float(np.linalg.norm(X[k + 1] - x)),
            "encrypt_ms": enc_ms, "evaluate_ms": ev_ms, "decrypt_ms": dec_ms,
            "offline_ms": off_ms,
            "enc_x_bytes": ct_bytes,
            "state_noise_l2": float(np.linalg.norm(state_noise)),
            "state_noise_linf": float(np.max(np.abs(state_noise))),
            "control_noise": u - u_plain_same_state,
            "e_linf": int(np.max(np.abs(e))),
            "state_ok": bool(np.all(np.abs(state_noise) <=
                                    (np.abs(e) + 1.0) / params.scale + 1e-12)),
            "control_ok": bool(control_ok),
            "ciphertext": ct_copy,
        })

        if verbose and (k < 3 or k % 20 == 0 or k == steps - 1):
            print(f"[step {k:3d}] x[16]={x[16]: .6f}  V[0..2]={ct_copy[:3]}  "
                  f"u_enc={u: .9f}  u_plain={u_plain_same_state: .9f}  "
                  f"|du|={abs(u - u_plain_same_state):.2e}  "
                  f"enc={enc_ms*1e3:.2f}us eval={ev_ms*1e3:.2f}us dec={dec_ms*1e3:.2f}us "
                  f"offline={off_ms*1e3:.2f}us")

    # Secondary measurement: fused single dispatch, replayed from the same inputs.
    if backend == "numba":
        cipher.reset()
        cipher.precompute()
        for k in range(steps):
            t0 = time.perf_counter_ns()
            ct, _, u_f = cipher.fused_step(X[k], noise_bank[k])
            timings["fused"].append((time.perf_counter_ns() - t0) * _NS)
            assert u_f == U[k], "fused replay diverged from three-phase run"
            cipher.advance()
            cipher.precompute()
    if backend == "numba":
        # dispatch-free compute cost per step (whole replay inside one compiled call)
        on, full = cipher.kernel_timing(X[:steps], noise_bank)
        timings["kernel_online_ms"], timings["kernel_full_ms"] = [on * _NS], [full * _NS]
    return X, U, timings, trace


def residue_to_signed(residue, q):
    r = int(residue) % q
    return r if r <= q // 2 else r - q


if __name__ == "__main__":
    from state_space_model import build_plant, design_state_feedback, discretize, N_MASSES
    from lwe_core import PROFILES
    A, B, C, D = build_plant()
    K = design_state_feedback(A, B)
    Ad, Bd, _, _ = discretize(A, B, C, D, dt=0.1)
    x0 = np.zeros(A.shape[0])
    x0[2 * (N_MASSES - 1)] = 1.0
    simulate_encrypted(Ad, Bd, K, x0, 200, PROFILES["optimized"])
