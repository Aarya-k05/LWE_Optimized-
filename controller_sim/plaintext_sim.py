"""
============================================================
PLAINTEXT (UNENCRYPTED) CLOSED-LOOP SIMULATION
============================================================
Baseline: the controller computes u(k) = -K x(k) directly in the
clear. This is what a normal digital controller does, and is the
reference we compare the encrypted controller against.

Nothing here changed for the move from 2 states to N states: this
function was already written generically in terms of n = Ad.shape[0]
and a scalar u, so it works unmodified for the 18-state chain model.
============================================================
"""

import numpy as np
from state_space_model import build_plant, design_state_feedback, discretize, N_MASSES


def simulate_plaintext(Ad, Bd, K, x0, steps, r=0.0):
    """
    Regulates the plant state to the reference r (default 0) using
    u(k) = K (r - x1(k))  ... implemented here as full state feedback
    u(k) = -K x(k) (regulator form), matching e(k)=r(k)-y(k), u(k)=Ke(k).
    """
    n = Ad.shape[0]
    x = np.array(x0, dtype=float).reshape(n, 1)

    X = np.zeros((steps + 1, n))
    U = np.zeros(steps)
    X[0] = x.flatten()

    for k in range(steps):
        u = -K @ x            # controller: u(k) = -K x(k)
        u_val = float(u[0, 0])
        x = Ad @ x + Bd * u_val
        X[k + 1] = x.flatten()
        U[k] = u_val

    return X, U


if __name__ == "__main__":
    A, B, C, D = build_plant()
    K = design_state_feedback(A, B)
    Ad, Bd, Cd, Dd = discretize(A, B, C, D)

    n = A.shape[0]
    x0 = np.zeros(n)
    x0[2 * (N_MASSES - 1)] = 1.0   # displace the last mass by 1m, rest at 0
    X, U = simulate_plaintext(Ad, Bd, K, x0, steps=200)

    print("Final state:", X[-1])
    print("Max control effort |u|:", np.max(np.abs(U)))
