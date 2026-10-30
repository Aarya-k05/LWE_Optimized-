# Encrypted State-Feedback Control Simulation — LWE Stream Cipher (optimized)

Software simulation of a state-feedback controller whose computation is carried out on
**LWE recursive stream-cipher ciphertexts**. This is the LWE counterpart of the CKKS project and
follows the same structure, file names, result files and plots.

    Enc(x) · (-K) = Enc(u)        i.e.   F·V[k] − (F·S)·V[k−1]  ≈  −K·x[k]

**Plant: a 9-mass spring-damper chain (18 states)**, one actuator on the last mass, full state
available to the controller — identical to the CKKS project (same `state_space_model.py`,
same LQR gain `K`, same `x0`, `dt = 0.1 s`, 200 steps).

## Prerequisites and installation

Python 3.9+.

```bash
pip install numpy scipy matplotlib numba
```

## Running

```bash
cd controller_sim

python compare_and_plot.py                  # full run: plaintext vs LWE, all CSVs/plots/summary,
                                            #   relative-noise graph, CKKS comparison  (default profile)
python compare_and_plot.py --profile legacy # reproduces the previous LWE code bit-for-bit
python compare_and_plot.py --skip-extras    # only the main run (no relative-noise / CKKS comparison)

python encrypted_sim.py                     # only the encrypted controller, step-by-step trace printed
python benchmark.py --repeats 5             # ablation table: which optimisation buys what
python relative_noise.py                    # (re)build the relative-noise graph from results/
python compare_ckks_lwe.py                  # (re)build the CKKS-vs-LWE comparison from results/
python ../tests/test_correctness.py         # correctness tests
```

The first run compiles the Numba kernels (a few seconds, then cached on disk).
All outputs go to `results/`.

## Project structure

```
controller_sim/
├── state_space_model.py    # Plant (9-mass chain) + LQR gain          (identical to CKKS project)
├── plaintext_sim.py        # Baseline (unencrypted) controller        (identical to CKKS project)
├── lwe_core.py             # THE CIPHER: parameters, kernels, 5-method API, self-checks  <-- read this
├── encrypted_sim.py        # Closed loop through the cipher + timing + noise tracking
├── compare_and_plot.py     # Run both, write every CSV / plot / summary
├── benchmark.py            # Ablation table (CKKS benchmark.py analogue)
├── relative_noise.py       # Absolute + relative noise, separate graph
└── compare_ckks_lwe.py     # CKKS-vs-LWE comparison CSV + four graphs
tests/test_correctness.py
reference/                  # CKKS project zip + its control_data.csv (used by the comparison)
baseline_results/           # pre-optimisation stats: original code and previous "v2" code
results/                    # all generated output
```

`lwe_core.py` has the same banner-style documentation as the CKKS files; the entire scheme is
five methods: `precompute → encrypt → evaluate → recover → advance`.

## How it works

```
Simulated state x(k)
    ↓  SENSOR     V[k] = pad + round(x·2^s) + e[k]                (mod q)      encrypt
Ciphertext V[k]   (18 words)
    ↓  CONTROLLER r = F·V[k] − (F·S)·V[k−1]                       (mod q)      evaluate
Residue r         (uses only ciphertexts + public matrix F·S, never S)
    ↓  ACTUATOR   u = centred(r) / 2^(s+g)                                     decrypt
Control u  →  update plant  →  next state
    ·  OFFLINE (idle part of the 100 ms period): pad = S·V[k], (F·S)·V[k] for the next step
```

Algorithm (unchanged from the report / previous code): `V[k] = S·V[k−1] + m[k] + e[k] (mod q)`,
`m = round(x·2^s)`, `F = round(−K·2^g)`, `V[−1] = 0`, decryption subtracts `S·V[k−1]` exactly.

## What changed vs the previous LWE code

| # | change | why | effect |
|---|---|---|---|
| O1 | **pad and (F·S)·V[k−1] computed offline** | They depend only on the *previous* ciphertext, so they are computed before x[k] exists (same idea as the CKKS Enc(0) pre-computation) | online compute 747 → ~200 ns |
| O2 | **modulus q = 2^61 − 1, scale 2^28, gain 2^22** (was 2^31 − 1, 2^16, 2^8) | Mersenne reduction is shift+add; limb-split 64-bit multiply keeps it exact. σ stays 1, so noise/message drops ~4096× | **max u error 1.44e-2 → 3.3e-6** |
| O3 | exact discrete-Gaussian noise (inverse-CDF table), all 200 steps sampled in one call | `round(Box-Muller)` has variance σ²+1/12 and a wrong tail | slightly lower, cleaner noise |
| O4 | fused / bundled Numba kernels (one int64 workspace) | a Numba call costs ~0.15 µs per array argument; bundling 9 args → 6 gave 1.24 → 0.77 µs per call | lower call latency |
| O5 | full-range uniform secret S | the old `S < q/1000` restriction only existed to dodge int64 overflow | removes an ad-hoc restriction |

Not changed: the recurrence, the report equation (3.11), the plant, `K`, `x0`, the result files.

## Results (this machine; absolute times vary by CPU)

Ablation (`benchmark.py`, 200 steps, median of 5):

| configuration | call crit. µs | compute crit. µs | total µs | max u err | max x err | ct bytes |
|---|---|---|---|---|---|---|
| 1 original (Python ints, σ=√18) | 122.6 | **122.6** | 122.6 | 6.12e-02 | 3.10e-03 | 72 |
| 2 + σ = 1 | 129.5 | **129.5** | 129.5 | 1.44e-02 | 7.25e-04 | 72 |
| 3 + NumPy int64 | 39.3 | **39.3** | 39.3 | 1.44e-02 | 7.25e-04 | 72 |
| 4 + Numba kernels | 4.7 | **0.75** | 0.75 | 1.44e-02 | 7.25e-04 | 72 |
| 5 + offline pad/FS split | 3.2 | **0.19** | 0.75 | 1.44e-02 | 7.25e-04 | 72 |
| 6 + wide modulus (**default**) | 2.9 | **0.22** | 1.22 | **3.29e-06** | **2.16e-07** | 144 |

* *call crit.* = encrypt+evaluate+decrypt timed per Python call (includes ≈1 µs Python→Numba dispatch per call — the floor of any Python timer; this is what the previous code reported).
* *compute crit.* = pure compute on the critical path, dispatch excluded (whole replay inside one compiled call) — what an embedded implementation would see.

LWE vs CKKS (same plant/baseline, `ckks_vs_lwe_comparison.csv`): max control error **3.3e-6 vs 2.1e-5**, max state error **2.2e-7 vs 1.3e-6** (LWE ≈ 6× lower); critical path ≈ µs vs ≈ ms for CKKS.

**Honest trade-offs**
* The previous code was already at the Python dispatch floor (~3 µs per step), so *per-call* latency improves only modestly; the real latency gain is in compute (O1) and in noise (O2).
* The wide modulus costs ≈ 0.47 µs more *total* work per step (324 limb-split multiplies instead of 324 plain ones) — still off the critical path.
* σ = 1 with q = 2^61 − 1 makes the LWE instance *easier* to attack than before; n = 18 was already far below secure sizes. **No security claim is made** (as in the original). If a security margin matters, raise σ: noise scales linearly with σ, and σ ≈ 16 would still be ~250× quieter than the old code.
* `max_control` (largest |u| without modular wrap) is 1024 for the default profile (was 64); the run measures max |u| = 56.9. The simulator flags any step that would wrap.

## Noise tracking

Absolute noise is recorded per step (`lwe_stream_results.csv`, `execution_trace.csv`):
`state_noise_l2`, `state_noise_linf`, `control_noise_abs` (cipher noise on the same state), `e_linf`,
plus the trajectory errors vs plaintext (`state_error_l2`, `control_error_abs`) as in the CKKS project.
`summary.txt §8` reports a growth check (rms of 2nd half / 1st half ≈ 0.99 → no accumulation).

**Relative noise** has its own graph (`results/lwe_relative_noise.png`, `lwe_relative_noise_results.csv`,
`lwe_relative_noise_summary.txt`). The criteria currently implemented — peak-relative, horizon-relative,
per-step floored (1e-3 × peak) and per-step raw — live in `metrics()` and `FLOOR_FRACTION` in
`relative_noise.py`; edit those two places once you fix the exact criteria and everything follows.

## Techniques considered

| technique | used? | reasoning |
|---|---|---|
| Bootstrapping | **no, not needed** | Noise does not accumulate: the decryptor subtracts `S·V[k−1]` exactly, so each step contains only the fresh `e[k]` (measured flat). Encrypted-control literature reaches the same conclusion for recursive LWE/Ring-LWE controllers: Kim–Shim–Han (arXiv:1912.07362), Jang et al. (arXiv:2406.14372), Lee et al. (arXiv:2307.03451). |
| NTT | **no** | NTT accelerates *polynomial* (Ring-LWE) products. Here `S` is an unstructured 18×18 matrix (324 multiplies); a size-32 negacyclic NTT route needs ≈ 3 transforms + 32 pointwise products and would change the scheme to Ring-LWE. Pays off from n ≳ 256. |
| Mersenne reduction, limb-split multiply, lazy folding | yes (O2) | Crandall (Mersenne primes); Harvey, *Faster arithmetic for NTTs* (J. Symb. Comput. 2014). |
| CDT discrete-Gaussian sampling | yes (O3) | e.g. Ducas–Durmus–Lepoint–Lyubashevsky, CRYPTO 2013. |

The three arXiv papers were found and checked via search while preparing this; the Crandall / Harvey /
Ducas et al. references are standard ones cited from memory — please verify them before quoting in a report.

## Files in `results/`

`control_data.csv`, `execution_trace.csv`, `state_response.png`, `control_signal.png`,
`error_vs_step.png`, `timing_breakdown.png`, `summary.txt` — same names/format as the CKKS project
(extra noise columns appended at the end).
`lwe_stream_results.csv`, `lwe_stream_summary.txt`, `lwe_stream_four_graphs.png/.pdf` — same names/columns as
the previous LWE code (noise/offline columns appended at the end).
`lwe_relative_noise.*`, `ckks_vs_lwe_comparison.csv`, `ckks_vs_lwe_four_graphs.png/.pdf`,
`benchmark_ablation.csv/.md/.png`.

## Notes

* Same caveat as before: custom secret-key recurrence, not standard HE; evaluation is by modular recurrence cancellation; n = 18; no security claim.
* The plant is simulated in plaintext; only the controller computation goes through the cipher.
* Control steps cannot be parallelised (`x(k+1)` needs `u(k)`); only the offline pre-computation is independent of the state.
* The CKKS-vs-LWE comparison remains marked PROVISIONAL (gain encoding is not specified by the report; no network transfer measured; run environments not proven identical), as in the previous version.
* `--profile legacy` reproduces the previous code exactly (max u error 1.4436e-2, max state error 7.254e-4).
