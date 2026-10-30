"""
============================================================
LWE RECURSIVE STREAM-CIPHER  --  CORE (parameters, kernels, cipher)
============================================================

This file contains the ENTIRE cryptographic part of the project.  Everything
else (plant, plots, CSVs) only calls the five methods of `LWEStreamCipher`:

    cipher.precompute()      OFFLINE   pad = S.V[k-1],  fsv = (F.S).V[k-1]
    cipher.encrypt(x, e)     ONLINE    V[k] = pad + round(x*2^s) + e   (mod q)
    cipher.evaluate()        ONLINE    r = F.V[k] - fsv                 (mod q)
    cipher.recover(r)        ONLINE    x_hat = V[k] - pad,  u = centred(r)/2^(s+g)
    cipher.advance()         bookkeeping: V[k-1] <- V[k]

THE ALGORITHM (unchanged from the report / the previous LWE code)
-----------------------------------------------------------------
    S      secret 18x18 matrix mod q            F = round(-K * 2^g)   integer gain row
    V[-1]  = 0                                  FS = (F.S) mod q      public eval key

    V[k] = S.V[k-1] + m[k] + e[k]   (mod q)     m[k] = round(x[k] * 2^s)   fixed point
                                                e[k] = small integer Gaussian noise

    state  :  m[k]+e[k] = V[k] - S.V[k-1]
    control:  F.V[k] - (F.S).V[k-1] = F.(m[k]+e[k])        (mod q)   <-- report eq. (3.11)
              u[k] = centred_residue / 2^(s+g)  ~=  -K.x[k]

The controller side only ever needs V[k], V[k-1] and the matrix FS = F.S.
It never needs S itself (this is what "evaluation" means in this scheme).

WHAT IS *NOT* CLAIMED
---------------------
Same caveat as the original code: this is a custom secret-key recurrence, not a
standard homomorphic-encryption scheme, and n = 18 is far below any secure LWE
dimension.  No security claim is made.  Latency / noise are the subject here.

WHAT WAS OPTIMISED  (each item is measured in benchmark.py)
-----------------------------------------------------------
 O1  Pad / FS pre-computation OFF the critical path.
     The recurrence means S.V[k-1] and FS.V[k-1] depend only on the PREVIOUS
     ciphertext, so they are computed in the idle part of the 100 ms sampling
     period, before x[k] exists.  The online path becomes 18 additions + 18
     modular multiplications.  (Same idea as the Enc(0) pool in the CKKS code.)
 O2  Mersenne modulus q = 2^61 - 1 with limb-split 64-bit multiplication.
     Reduction mod a Mersenne prime is shift+add (2^61 = 1 mod q), no division.
     The wide modulus lets the fixed-point scale grow from 2^16 to 2^28 while the
     injected noise stays sigma = 1, so noise / message drops by ~4096x.
     (Crandall 1992 Mersenne reduction; Harvey 2014 lazy/limb arithmetic.)
 O3  Exact discrete-Gaussian noise via an inverse-CDF table (CDT sampler),
     sampled for all 200 steps in one vectorised call (noise is state-independent).
     The old "round(Box-Muller)" sampler has variance sigma^2 + 1/12 and the wrong tail.
 O4  Fused, allocation-free Numba kernels (no Python big-int loops, no temporaries).
 O5  Uniform full-range secret S (old code limited S < q/1000 only to dodge int64
     overflow; the 128-bit-safe multiply removes that restriction).

WHAT WAS CONSIDERED AND DELIBERATELY *NOT* USED
-----------------------------------------------
 * Bootstrapping: not needed.  The ciphertext "noise" never accumulates: the
   decryptor subtracts S.V[k-1] EXACTLY, so each step only contains the fresh e[k]
   (measured flat over 200 steps, see results/lwe_relative_noise.png and the
   noise-growth line in summary.txt).  The recursive-LWE encrypted-control papers
   (Kim-Shim-Han, IEEE TAC 2020 / arXiv:1912.07362; Jang et al. arXiv:2406.14372)
   reach the same conclusion: closed-loop stability, not bootstrapping, tames error.
 * NTT: an NTT speeds up polynomial (Ring-LWE) products.  Here S is an unstructured
   18x18 matrix: 324 multiplications.  A size-32 negacyclic NTT route would need
   ~3 transforms x 80 butterflies + 32 pointwise products, no saving at n = 18,
   and it would change the scheme to Ring-LWE.  It pays off for n >= ~256.

Only the mathematics above is "the algorithm"; everything else is engineering.
============================================================
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numba as nb
import numpy as np

N_STATES = 18


# ----------------------------------------------------------------------------
# 1. PARAMETERS
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class LWEParams:
    """All tunable numbers in one place.

    q_bits      modulus q = 2^q_bits - 1 (Mersenne prime: 31 or 61)
    scale_bits  s : state fixed-point scale 2^s            (noise / message ratio)
    gain_bits   g : gain-row fixed-point scale 2^g         (gain quantisation)
    sigma       std-dev of the integer noise e (units of 1/2^s)
    secret_div  S entries are uniform in [0, q // secret_div)   (1 = full range)
    sampler     "cdt" (exact discrete Gaussian) or "box_muller" (legacy rounding)
    """
    name: str
    q_bits: int
    scale_bits: int
    gain_bits: int
    sigma: float
    secret_div: int
    sampler: str
    seed: int = 42

    @property
    def q(self) -> int:
        return (1 << self.q_bits) - 1

    @property
    def scale(self) -> float:
        return float(1 << self.scale_bits)

    @property
    def gain_scale(self) -> float:
        return float(1 << self.gain_bits)

    @property
    def control_scale(self) -> float:
        return self.scale * self.gain_scale

    @property
    def max_control(self) -> float:
        """Largest |u| that can be recovered without modular wrap-around."""
        return (self.q // 2) / self.control_scale

    @property
    def ciphertext_bytes(self) -> int:
        return N_STATES * (8 if self.q_bits > 32 else 4)


PROFILES = {
    # DEFAULT.  Wide modulus, large scale: low noise, still microseconds.
    "optimized": LWEParams("optimized", 61, 28, 22, 1.0, 1, "cdt"),
    # The previous (already Numba-optimised) code: reproduces it bit-for-bit.
    "legacy": LWEParams("legacy", 31, 16, 8, 1.0, 1000, "box_muller"),
    # The very first code (sigma = sqrt(18)); used only as ablation row 1.
    "original": LWEParams("original", 31, 16, 8, math.sqrt(N_STATES), 1000, "box_muller"),
}


# ----------------------------------------------------------------------------
# 2. NOISE SAMPLING  (state independent -> sampled up-front, off the critical path)
# ----------------------------------------------------------------------------
def sample_noise_bank(rng: np.random.Generator, steps: int, p: LWEParams) -> np.ndarray:
    """Return a (steps, 18) int64 array of integer noise e[k]."""
    if p.sampler == "box_muller":            # legacy: rounded continuous Gaussian
        u = np.maximum(rng.random((steps, N_STATES, 2)), np.finfo(np.float64).tiny)
        z = np.sqrt(-2.0 * np.log(u[:, :, 0])) * np.cos(2.0 * np.pi * u[:, :, 1])
        return np.rint(p.sigma * z).astype(np.int64)
    # CDT: P[e = k] proportional to exp(-k^2 / (2 sigma^2)),  |k| <= tail
    tail = int(math.ceil(12 * p.sigma)) + 1
    ks = np.arange(-tail, tail + 1)
    pmf = np.exp(-(ks.astype(np.float64) ** 2) / (2.0 * p.sigma ** 2))
    cdf = np.cumsum(pmf / pmf.sum())
    cdf[-1] = 1.0
    idx = np.searchsorted(cdf, rng.random((steps, N_STATES)), side="right")
    return (idx - tail).astype(np.int64)


# ----------------------------------------------------------------------------
# 3. MODULAR ARITHMETIC KERNELS (Numba)
# ----------------------------------------------------------------------------
@nb.njit(cache=True, nogil=True)
def _mulmod_m61(a, b):
    """a*b mod (2^61-1) for 0 <= a,b < 2^61, using only 64-bit words.

    Split a = a1*2^31 + a0 and b = b1*2^31 + b0 (a1,b1 < 2^30), then
        a*b = a1*b1*2^62 + (a0*b1 + a1*b0)*2^31 + a0*b0 ,   and 2^61 = 1 (mod q)
    so 2^62 -> 2 and mid*2^31 -> (mid >> 30) + ((mid & (2^30-1)) << 31).
    Every partial sum is < 2^64 (bounds in the comments), so nothing overflows.
    """
    m61 = np.uint64(2305843009213693951)
    m31 = np.uint64(2147483647)
    m30 = np.uint64(1073741823)
    a = np.uint64(a)
    b = np.uint64(b)
    a0 = a & m31
    a1 = a >> np.uint64(31)
    b0 = b & m31
    b1 = b >> np.uint64(31)
    lo = a0 * b0                      # < 2^62
    mid = a0 * b1 + a1 * b0           # < 2^62
    hi = a1 * b1                      # < 2^60
    r = (hi << np.uint64(1)) + (mid >> np.uint64(30)) + ((mid & m30) << np.uint64(31)) + lo
    r = (r & m61) + (r >> np.uint64(61))      # fold: result < 2^61 + 8
    if r >= m61:
        r -= m61
    return np.int64(r)


@nb.njit(cache=True, nogil=True)
def _mulmod_m31(a, b):
    """a*b mod (2^31-1) for 0 <= a,b < 2^31 (product < 2^62 fits int64)."""
    m31 = np.int64(2147483647)
    p = a * b
    p = (p & m31) + (p >> 31)
    p = (p & m31) + (p >> 31)
    if p >= m31:
        p -= m31
    return p


def _build_kernels(q_bits: int):
    """Compile the five per-step kernels for q = 2^q_bits - 1."""
    Q = (1 << q_bits) - 1
    HALF = Q // 2
    mm = _mulmod_m61 if q_bits == 61 else _mulmod_m31

    @nb.njit(cache=True, nogil=True)
    def addmod(a, b):
        s = a + b
        return s - Q if s >= Q else s

    @nb.njit(cache=True, nogil=True)
    def dot_mod(vec, v):
        acc = np.int64(0)
        for j in range(vec.shape[0]):
            acc = addmod(acc, mm(vec[j], v[j]))
        return acc

    # Workspace layout (all int64, n = 18):   ws = [ pad(n) | V[k](n) | fsv(1) | V[k-1](n) ]
    # Bundling everything into ONE array matters: a Numba call costs ~0.15 us per
    # array argument (unboxing), which is more than the arithmetic itself here.

    @nb.njit(cache=True, nogil=True)
    def precompute(secret, fs, ws):
        """OFFLINE: pad = S.V[k-1] mod q ; fsv = (F.S).V[k-1] mod q."""
        n = secret.shape[0]
        prev = ws[2 * n + 1:]
        for i in range(n):
            ws[i] = dot_mod(secret[i], prev)
        ws[2 * n] = dot_mod(fs, prev)

    @nb.njit(cache=True, nogil=True)
    def encrypt(x, noise, ws, consts):
        """ONLINE: V[k] = pad + round(x * 2^s) + e  (mod q)."""
        n = x.shape[0]
        for i in range(n):
            m = np.int64(np.rint(x[i] * consts[0])) + noise[i]
            ws[n + i] = addmod(ws[i], m % Q)

    @nb.njit(cache=True, nogil=True)
    def evaluate(gain_mod, ws):
        """ONLINE: report eq. (3.11):  F.V[k] - (F.S).V[k-1]  (mod q)."""
        n = gain_mod.shape[0]
        return (dot_mod(gain_mod, ws[n:2 * n]) - ws[2 * n]) % Q

    @nb.njit(cache=True, nogil=True)
    def recover(ws, residue, state_out, consts):
        """ONLINE: centred decode of the state (V - pad) and of the control residue."""
        n = state_out.shape[0]
        for i in range(n):
            r = (ws[n + i] - ws[i]) % Q
            if r > HALF:
                r -= Q
            state_out[i] = r / consts[0]
        c = residue % Q
        if c > HALF:
            c -= Q
        return c / consts[1]

    @nb.njit(cache=True, nogil=True)
    def fused(x, noise, ws, gain_mod, state_out, consts):
        """encrypt + evaluate + recover in ONE native call (one dispatch)."""
        encrypt(x, noise, ws, consts)
        res = evaluate(gain_mod, ws)
        return recover(ws, res, state_out, consts)

    @nb.njit(cache=True, nogil=True)
    def replay(secret, fs, gain_mod, ws, X, noise, state_out, u_out, consts, with_offline):
        """Run X.shape[0] steps INSIDE one native call (no Python dispatch per step).

        Used only for timing: time(replay)/steps is the pure compute cost per step.
        with_offline=False -> online path only; True -> also pad/FS pre-compute + advance.
        """
        n = secret.shape[0]
        for k in range(X.shape[0]):
            u_out[k] = fused(X[k], noise[k], ws, gain_mod, state_out, consts)
            if with_offline:
                for i in range(n):
                    ws[2 * n + 1 + i] = ws[n + i]
                precompute(secret, fs, ws)

    return precompute, encrypt, evaluate, recover, fused, replay


_KERNELS: dict = {}


def kernels(q_bits: int):
    if q_bits not in _KERNELS:
        _KERNELS[q_bits] = _build_kernels(q_bits)
    return _KERNELS[q_bits]


# ----------------------------------------------------------------------------
# 4. PURE-PYTHON / NUMPY REFERENCE ARITHMETIC (readable ground truth, ablation rows)
# ----------------------------------------------------------------------------
def centered(value: int, q: int) -> int:
    """Canonical residue -> signed representative in [-(q-1)/2, (q-1)/2]."""
    r = int(value) % q
    return r if r <= q // 2 else r - q


def _py_dot(a, b, q):
    return sum(int(x) * int(y) for x, y in zip(a, b)) % q


def _np_dot(a: np.ndarray, v: np.ndarray, q: int) -> np.ndarray:
    """Exact (a @ v) mod q for a, v < 2^31 using int64: split v into 16-bit halves."""
    lo, hi = v & 0xFFFF, v >> 16
    return ((a @ lo) % q + (((a @ hi) % q) << 16)) % q


# ----------------------------------------------------------------------------
# 5. THE CIPHER
# ----------------------------------------------------------------------------
class LWEStreamCipher:
    """Sensor-side encryptor + controller-side evaluator + actuator-side decryptor.

    backend  "numba"  (default, fast)
             "numpy"  (q = 2^31-1 only; vectorised int64)
             "python" (arbitrary-precision ints; slowest, used as ground truth)
    """

    def __init__(self, params: LWEParams, K: np.ndarray, backend: str = "numba"):
        if backend == "numpy" and params.q_bits != 31:
            raise ValueError("numpy backend supports q = 2^31-1 only")
        self.p, self.backend, self.q = params, backend, params.q
        n = N_STATES
        # ---- key generation (setup, excluded from per-step timing) ----
        self.rng = np.random.default_rng(params.seed)
        self.secret = self.rng.integers(0, self.q // params.secret_div, size=(n, n),
                                        dtype=np.int64)
        self.gain_int = np.rint(-np.asarray(K).reshape(n) * params.gain_scale).astype(np.int64)
        self.gain_mod = np.ascontiguousarray(self.gain_int % self.q)
        self.fs = np.array([_py_dot(self.gain_int, self.secret[:, j], self.q)
                            for j in range(n)], dtype=np.int64)       # (F.S) mod q
        self.reset()
        if backend == "numba":
            self._k = kernels(params.q_bits)
            self._warm()

    def reset(self):
        n = N_STATES
        # one int64 workspace [pad | V[k] | fsv | V[k-1]]; the named arrays are VIEWS into it
        self.ws = np.zeros(3 * n + 1, dtype=np.int64)
        self.pad, self.ct, self.fsv = self.ws[:n], self.ws[n:2 * n], self.ws[2 * n:2 * n + 1]
        self.prev_ct = self.ws[2 * n + 1:]                # V[k-1] (zeros at start)
        self.consts = np.array([self.p.scale, self.p.control_scale])
        self.state_hat = np.zeros(n, dtype=np.float64)

    def _warm(self):
        """Trigger JIT load once so it is not billed to step 0 (reported separately)."""
        z = np.zeros(N_STATES, dtype=np.float64)
        zi = np.zeros(N_STATES, dtype=np.int64)
        pre, enc, ev, rec, fus, _ = self._k
        pre(self.secret, self.fs, self.ws)
        enc(z, zi, self.ws, self.consts)
        rec(self.ws, ev(self.gain_mod, self.ws), self.state_hat, self.consts)
        fus(z, zi, self.ws, self.gain_mod, self.state_hat, self.consts)
        self.reset()

    # ---- OFFLINE ------------------------------------------------------------
    def precompute(self):
        q = self.q
        if self.backend == "numba":
            self._k[0](self.secret, self.fs, self.ws)
        elif self.backend == "numpy":
            self.pad[:] = _np_dot(self.secret, self.prev_ct, q)
            self.fsv[:] = int(_np_dot(self.fs[None, :], self.prev_ct, q)[0])
        else:
            self.pad[:] = [_py_dot(r, self.prev_ct, q) for r in self.secret]
            self.fsv[:] = _py_dot(self.fs, self.prev_ct, q)

    # ---- ONLINE -------------------------------------------------------------
    def encrypt(self, x: np.ndarray, noise: np.ndarray) -> np.ndarray:
        if self.backend == "numba":
            self._k[1](x, noise, self.ws, self.consts)
        else:
            m = np.rint(x * self.p.scale).astype(np.int64)
            self.ct[:] = (self.pad + m + noise) % self.q
        return self.ct

    def evaluate(self) -> int:
        if self.backend == "numba":
            return int(self._k[2](self.gain_mod, self.ws))
        if self.backend == "numpy":
            return int((_np_dot(self.gain_mod[None, :], self.ct, self.q)[0]
                        - int(self.fsv[0])) % self.q)
        return (_py_dot(self.gain_mod, self.ct, self.q) - int(self.fsv[0])) % self.q

    def recover(self, residue: int):
        p = self.p
        if self.backend == "numba":
            u = self._k[3](self.ws, residue, self.state_hat, self.consts)
            return self.state_hat, u
        st = np.array([centered(int(c) - int(a), self.q) for c, a in zip(self.ct, self.pad)],
                      dtype=np.float64) / p.scale
        self.state_hat = st
        return st, centered(residue, self.q) / p.control_scale

    def fused_step(self, x, noise):
        """Single-dispatch encrypt+evaluate+recover (numba only)."""
        u = self._k[4](x, noise, self.ws, self.gain_mod, self.state_hat, self.consts)
        return self.ct, self.state_hat, u

    def kernel_timing(self, X, noise_bank, reps: int = 200):
        """Pure compute cost per step (ns), dispatch excluded.  Returns (online, full)."""
        import time
        saved = self.ws.copy()
        u_out = np.empty(X.shape[0])
        X = np.ascontiguousarray(X[:X.shape[0]])
        out = []
        for with_off in (False, True):
            best = []
            for _ in range(reps):
                t0 = time.perf_counter_ns()
                self._k[5](self.secret, self.fs, self.gain_mod, self.ws, X, noise_bank,
                           self.state_hat, u_out, self.consts, with_off)
                best.append((time.perf_counter_ns() - t0) / X.shape[0])
            out.append(float(np.median(best)))
        self.ws[:] = saved
        return tuple(out)

    def advance(self):
        self.prev_ct[:] = self.ct

    def serialize(self) -> bytes:
        fmt = ">%d%s" % (N_STATES, "Q" if self.p.q_bits > 32 else "I")
        return struct.pack(fmt, *(int(v) for v in self.ct))


# ----------------------------------------------------------------------------
# 6. SELF-CHECKS  (these are the PASS/FAIL lines in summary.txt)
# ----------------------------------------------------------------------------
def run_self_checks(p: LWEParams, K: np.ndarray) -> dict[str, bool]:
    q, n = p.q, N_STATES
    res: dict[str, bool] = {}
    rng = np.random.default_rng(7)

    # fixed-point encode/decode round trip
    x = rng.uniform(-5, 5, n)
    res["fixed_point"] = bool(np.all(np.abs(np.rint(x * p.scale) / p.scale - x)
                                     <= 0.5 / p.scale + 1e-15))

    # centred modular interval boundaries
    h = q // 2
    res["modular_wraparound"] = (centered(h, q) == h and centered(-h, q) == -h
                                 and centered(h + 1, q) == -h and centered(-h - 1, q) == h)

    # recurrence + report equation, against pure-python big-int ground truth
    ref = LWEStreamCipher(p, K, "python")
    fast = LWEStreamCipher(p, K, "numba")
    ok_state = ok_report = ok_kernel = True
    for _ in range(500):
        x = rng.uniform(-0.1, 0.1, n)      # keeps |u| inside even the legacy +-64 range
        e = rng.integers(-4, 5, n, dtype=np.int64)
        for c in (ref, fast):
            c.prev_ct[:] = rng.integers(0, q, n, dtype=np.int64) if c is ref else ref.prev_ct
            c.precompute()
            c.encrypt(x, e)
        r_ref, r_fast = ref.evaluate(), fast.evaluate()
        st_ref, u_ref = ref.recover(r_ref)
        st_fast, u_fast = fast.recover(r_fast)
        m_plus_e = np.rint(x * p.scale).astype(np.int64) + e
        ok_state &= bool(np.array_equal(np.rint(st_ref * p.scale).astype(np.int64), m_plus_e))
        exp = sum(int(a) * int(b) for a, b in zip(ref.gain_int, m_plus_e))
        ok_report &= (r_ref == exp % q) and (abs(exp) <= h) and centered(r_ref, q) == exp
        ok_kernel &= bool(np.array_equal(ref.ct, fast.ct) and r_ref == r_fast
                          and np.array_equal(st_ref, st_fast) and u_ref == u_fast)
    res["recursive_state"], res["report_equation"], res["kernel_equivalence"] = (
        bool(ok_state), bool(ok_report), bool(ok_kernel))

    # signed control recovery: +, -, 0, large
    ok = True
    for u_target in (12.5, -0.125, 0.0, 50.0, -50.0):
        xs = np.zeros(n)
        xs[16] = u_target / float(-K[0, 16]) if u_target else 0.0
        c = LWEStreamCipher(p, K, "python")
        c.precompute()
        c.encrypt(xs, np.zeros(n, dtype=np.int64))
        _, u = c.recover(c.evaluate())
        tol = 0.5 * abs(xs[16]) / p.gain_scale + 0.5 * np.sum(np.abs(K)) / p.scale
        ok &= abs(u - u_target) <= tol + 1e-9
    res["signed_control"] = bool(ok)
    return res
