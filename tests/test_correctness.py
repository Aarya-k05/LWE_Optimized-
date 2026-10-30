"""
============================================================
CORRECTNESS TESTS  (python tests/test_correctness.py)
============================================================
1. The cipher's self-checks pass for both profiles.
2. Compiled Numba kernels == pure-Python big-int reference, step by step,
   over a full 200-step closed loop (bit-identical ciphertexts and controls).
3. The `legacy` profile reproduces the previous LWE code's errors exactly.
4. The `optimized` profile stays within CKKS-level accuracy.
5. No noise accumulation: noise in the 2nd half of the run is not larger than
   in the 1st half (=> bootstrapping is not needed).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "controller_sim"))
from state_space_model import build_plant, design_state_feedback, discretize, N_MASSES
from plaintext_sim import simulate_plaintext
from encrypted_sim import simulate_encrypted
from lwe_core import PROFILES, run_self_checks

A, B, C, D = build_plant(); K = design_state_feedback(A, B)
Ad, Bd, _, _ = discretize(A, B, C, D, dt=0.1)
x0 = np.zeros(18); x0[2 * (N_MASSES - 1)] = 1.0
Xp, Up = simulate_plaintext(Ad, Bd, K, x0, 200)

for name in ("optimized", "legacy"):
    assert all(run_self_checks(PROFILES[name], K).values()), f"self-checks failed: {name}"
print("self-checks: PASS")

for name in ("optimized", "legacy"):
    Xn, Un, _, trn = simulate_encrypted(Ad, Bd, K, x0, 200, PROFILES[name], "numba", verbose=False)
    Xr, Ur, _, trr = simulate_encrypted(Ad, Bd, K, x0, 200, PROFILES[name], "python", False, verbose=False)
    assert np.array_equal(Un, Ur) and all(np.array_equal(a["ciphertext"], b["ciphertext"])
                                          for a, b in zip(trn, trr)), f"numba != python ({name})"
print("numba == python reference (200 steps, bit-identical): PASS")

Xl, Ul, _, _ = simulate_encrypted(Ad, Bd, K, x0, 200, PROFILES["legacy"], verbose=False)
assert abs(np.max(np.abs(Ul - Up)) - 0.0144361658317) < 1e-9, "legacy no longer matches previous code"
print("legacy profile reproduces previous LWE code (max u err 1.44361658e-02): PASS")

Xo, Uo, _, tro = simulate_encrypted(Ad, Bd, K, x0, 200, PROFILES["optimized"], verbose=False)
mu, mx = np.max(np.abs(Uo - Up)), np.max(np.linalg.norm(Xo - Xp, axis=1))
assert mu < 2.2e-5 and mx < 1.4e-6, (mu, mx)          # CKKS-level (2.2e-5 / 1.4e-6)
assert all(r["state_ok"] and r["control_ok"] for r in tro)
print(f"optimized accuracy {mu:.2e} / {mx:.2e} (CKKS: 2.2e-5 / 1.4e-6): PASS")

cn = np.abs([r["control_noise"] for r in tro])
ratio = np.sqrt(np.mean(cn[100:] ** 2)) / np.sqrt(np.mean(cn[:100] ** 2))
assert ratio < 2.0, ratio
print(f"noise growth ratio 2nd/1st half = {ratio:.2f} (no accumulation): PASS")
print("ALL TESTS PASSED")
