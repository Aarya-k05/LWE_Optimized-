"""
============================================================
STATE-SPACE MODEL + STATE FEEDBACK CONTROLLER DESIGN
============================================================
This extends the original 2-state "dummy example of a control
system" to an N_MASSES-mass chain, giving a 2*N_MASSES state
system while keeping the exact same generic model used before:

    Error:            e(k) = r(k) - y(k)
    Controller:        u(k) = K e(k)              (state feedback: u = -Kx)
    State update:     x(k+1) = A x(k) + B u(k)
    Output:            y(k) = C x(k) + D u(k)

Plant chosen: a chain of N_MASSES masses connected in series by
springs and dampers (mass 1 is anchored to a fixed wall, mass
N_MASSES is free). A single actuator force u is applied to the
LAST mass in the chain, and the LAST mass's position is measured.
This keeps the controller SISO (one input, one output, one gain
row) -- identical in spirit to the original mass-spring-damper --
it's just N_MASSES of them daisy-chained together instead of one.

    wall --[k,c]-- m1 --[k,c]-- m2 --[k,c]-- ... --[k,c]-- mN <-- u(t)

Each mass contributes 2 states (position, velocity), interleaved
exactly like the original 2-state model:

    x = [pos_1, vel_1, pos_2, vel_2, ..., pos_N, vel_N]^T

so state indices 0,1 are "mass 1" (equivalent to the whole state
vector in the original 2-state demo), and so on down the chain.

Set N_MASSES = 8 for a 16-state system, 9 for 18-state, etc.
============================================================
"""

import numpy as np
from scipy import signal
from scipy.linalg import solve_continuous_are

# ------------------------------------------------------------
# Number of masses in the chain. n_states = 2 * N_MASSES.
# 9 masses -> 18 states (used by default here).
# ------------------------------------------------------------
N_MASSES = 9


def build_plant(n_masses=N_MASSES, m=1.0, c=1.0, k=2.0):
    """
    Continuous-time state-space model of an n_masses-mass chain,
    each mass connected to its neighbours (and mass 1 to a fixed
    wall) by identical springs (k) and dampers (c).

    Single actuator: force u applied to the LAST mass.
    Single sensor:   position of the LAST mass.

    This is the direct generalization of build_plant() from the
    2-state version: with n_masses=1 it reduces to exactly the
    original m*x'' + c*x' + k*x = u mass-spring-damper.
    """
    n = 2 * n_masses
    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    C = np.zeros((1, n))
    D = np.zeros((1, 1))

    for i in range(n_masses):
        pos_idx, vel_idx = 2 * i, 2 * i + 1

        # dx_i/dt = v_i
        A[pos_idx, vel_idx] = 1.0

        k_total = 0.0
        c_total = 0.0

        # connection to the left: previous mass, or the fixed wall
        # if this is the first mass
        k_total += k
        c_total += c
        if i > 0:
            A[vel_idx, 2 * (i - 1)] += k / m
            A[vel_idx, 2 * (i - 1) + 1] += c / m
        # (i == 0 case: the wall doesn't have states, so there's no
        #  cross term -- its effect is already inside k_total/c_total)

        # connection to the right: next mass, if it exists
        if i < n_masses - 1:
            k_total += k
            c_total += c
            A[vel_idx, 2 * (i + 1)] += k / m
            A[vel_idx, 2 * (i + 1) + 1] += c / m

        A[vel_idx, pos_idx] += -k_total / m
        A[vel_idx, vel_idx] += -c_total / m

    # Actuator: force applied directly to the last mass
    B[2 * (n_masses - 1) + 1, 0] = 1.0 / m

    # Sensor: position of the last mass
    C[0, 2 * (n_masses - 1)] = 1.0

    return A, B, C, D


def design_state_feedback(A, B, Q=None, R=None):
    """
    Design the state feedback gain K such that u(k) = -K x(k)
    stabilizes and regulates the plant, using the Linear Quadratic
    Regulator (LQR).

    Why LQR instead of manual pole placement here:
    a long single-actuator mass chain has very unevenly
    "controllable" modes -- some modes need enormous gain to move
    at all, so hand-picking N desired poles (as in the 2-state
    version) either fails or produces huge, numerically ugly gains
    that erode CKKS accuracy. LQR instead asks for a good balance
    between control effort and state error, and produces a gain
    matrix that stays well-scaled even as the number of states
    grows. It reduces to the same idea as pole placement -- it
    still returns a single K such that u = -Kx -- just computed by
    solving the Riccati equation instead of specifying poles by hand.
    """
    n = A.shape[0]

    if Q is None:
        # Weight every mass's position more heavily than its
        # velocity, so the controller prioritizes returning each
        # mass to its rest position.
        Q = np.eye(n)
        for i in range(n // 2):
            Q[2 * i, 2 * i] = 50.0
            Q[2 * i + 1, 2 * i + 1] = 1.0

    if R is None:
        R = np.array([[0.01]])

    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K


def discretize(A, B, C, D, dt=0.1):
    """Zero-order-hold discretization, matching x(k+1)=A_d x(k)+B_d u(k)."""
    sysd = signal.cont2discrete((A, B, C, D), dt, method='zoh')
    Ad, Bd, Cd, Dd, _ = sysd
    return Ad, Bd, Cd, Dd


if __name__ == "__main__":
    A, B, C, D = build_plant()
    K = design_state_feedback(A, B)
    Ad, Bd, Cd, Dd = discretize(A, B, C, D)

    print(f"n_masses = {N_MASSES}  ->  n_states = {A.shape[0]}")
    print("Open-loop eigenvalues (poles):\n", np.linalg.eigvals(A))
    print("\nState feedback gain K:\n", K)
    print("Closed-loop eigenvalues (A-BK):\n",
          np.linalg.eigvals(A - B @ K))
    print(f"\nDiscretized (dt=0.1s):")
    print("Ad shape:", Ad.shape, " Bd shape:", Bd.shape)
