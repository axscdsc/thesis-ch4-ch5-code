import argparse
import json
import math
import os
from dataclasses import asdict, dataclass

import numpy as np

try:
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
except Exception as exc:
    LinearNDInterpolator = None
    NearestNDInterpolator = None
    SCIPY_IMPORT_ERROR = exc
else:
    SCIPY_IMPORT_ERROR = None

from clean_fiplug_core import make_clean_params


@dataclass
class PDESettings:
    q: float
    gamma: float
    J: int
    Qmax: int
    Nt: int
    n_pi: int
    delta_min: float
    delta_max: float
    delta_points: int
    outdir: str
    seed: int
    drift_eps: float


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def make_ou_grid_np(J, eta):
    if J == 1:
        theta = np.array([0.0], dtype=float)
        L = np.zeros((1, 1), dtype=float)
        return theta, L

    std_u = math.sqrt(1.0 / (2.0 * eta))
    theta = np.linspace(-3.0 * std_u, 3.0 * std_u, J).astype(float)
    L = make_ou_generator_np(theta, eta)
    return theta, L


def make_ou_generator_np(theta, eta):
    J = len(theta)
    L = np.zeros((J, J), dtype=float)

    if J == 1:
        return L

    h = float(theta[1] - theta[0])
    for i in range(J):
        x = float(theta[i])
        drift = -eta * x
        rate_up = 0.5 / (h * h) + max(drift, 0.0) / h
        rate_dn = 0.5 / (h * h) + max(-drift, 0.0) / h

        if i == 0:
            L[i, i + 1] = rate_up + rate_dn
        elif i == J - 1:
            L[i, i - 1] = rate_up + rate_dn
        else:
            L[i, i + 1] = rate_up
            L[i, i - 1] = rate_dn

    for i in range(J):
        L[i, i] = -np.sum(L[i, :])
    return L


def make_simplex_grid(n_pi):
    pts = []
    ij = []
    for i in range(n_pi + 1):
        for j in range(n_pi + 1 - i):
            pi1 = i / n_pi
            pi2 = j / n_pi
            pi3 = 1.0 - pi1 - pi2
            pts.append([pi1, pi2, pi3])
            ij.append((i, j))
    return np.asarray(pts, dtype=float), ij


def project_simplex_3(pi):
    pi = np.asarray(pi, dtype=float)
    pi = np.maximum(pi, 1e-12)
    s = float(np.sum(pi))
    if s <= 0:
        return np.array([1/3, 1/3, 1/3], dtype=float)
    return pi / s


def posterior_jump(pi, lam_vec):
    numer = pi * lam_vec
    denom = float(np.sum(numer))
    if denom <= 1e-14:
        return pi.copy()
    return project_simplex_3(numer / denom)


def state_intensity_coefficients(theta, sigma, q_loading, gamma, varphi, psi):
    c = gamma * sigma * q_loading * theta
    ell_a = varphi + psi * np.exp(-c)
    ell_b = varphi + psi * np.exp(+c)
    return ell_a, ell_b


def nearest_pi_index(pi, pi_grid):
    pi = project_simplex_3(pi)
    d2 = np.sum((pi_grid - pi[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))


class ValueInterpolator:
    def __init__(self, pi_grid, values):
        self.pi_grid = pi_grid
        self.coords = pi_grid[:, :2]
        self.values = np.asarray(values, dtype=float)

        if LinearNDInterpolator is None:
            self.linear = None
            self.nearest = None
        else:
            self.linear = LinearNDInterpolator(self.coords, self.values)
            self.nearest = NearestNDInterpolator(self.coords, self.values)

    def __call__(self, pi):
        pi = project_simplex_3(pi)
        x = np.array([pi[0], pi[1]], dtype=float)

        if self.linear is None:
            idx = nearest_pi_index(pi, self.pi_grid)
            return float(self.values[idx])

        val = self.linear(x)
        val = float(np.asarray(val).reshape(-1)[0])
        if not np.isfinite(val):
            val = float(self.nearest(x))
        return val


def build_interpolators_for_time(V_next, pi_grid):
    return [ValueInterpolator(pi_grid, V_next[iq]) for iq in range(V_next.shape[0])]


def q_to_index(Q, q_grid):
    Q = int(Q)
    if Q <= int(q_grid[0]):
        return 0
    if Q >= int(q_grid[-1]):
        return len(q_grid) - 1
    return int(Q - int(q_grid[0]))


def belief_drift(pi, L, ell_a, ell_b):
    pi = project_simplex_3(pi)
    hat_a = float(np.dot(pi, ell_a))
    hat_b = float(np.dot(pi, ell_b))
    pred = pi @ L
    b = pred + pi * (hat_a - ell_a) + pi * (hat_b - ell_b)
    # Ensure it is tangent to the simplex numerically.
    b = b - np.mean(b)
    return b


def directional_belief_derivative(interp_Q, pi, b, eps):
    pi = project_simplex_3(pi)
    pi_plus = project_simplex_3(pi + eps * b)
    v0 = interp_Q(pi)
    v1 = interp_Q(pi_plus)
    return (v1 - v0) / eps


def bellman_update_ct_hjb(
    V_next,
    interpolators,
    iq,
    Q,
    pi,
    pi_grid,
    theta,
    L,
    ell_a,
    ell_b,
    controls,
    mm,
    dt,
    q_grid,
    drift_eps,
):
    alpha = float(getattr(mm, "alpha", 0.001))
    phi_inventory = float(
        getattr(mm, "running_phi",
        getattr(mm, "phi_inventory",
        getattr(mm, "phi", 0.1)))
    )
    k = float(getattr(mm, "k", 1.0))
    mu = float(getattr(mm, "mu", 0.0))
    eta = float(getattr(mm, "eta", 10.0))
    sigma = float(getattr(mm, "sigma", 1.0))
    q_loading = float(getattr(mm, "qconst", getattr(mm, "q", 0.6)))

    Qmin = int(q_grid[0])
    Qmax = int(q_grid[-1])

    pi = project_simplex_3(pi)

    # Semi-Lagrangian CTMC prediction only.
    pi_ctmc = project_simplex_3(pi + dt * (pi @ L))

    # Base continuation value at the predicted belief.
    V_base = interpolators[iq](pi_ctmc)

    m_pi = float(np.dot(pi_ctmc, theta))
    running = -phi_inventory * (Q ** 2) + Q * (mu - eta * sigma * q_loading * m_pi)

    hat_a = float(np.dot(pi_ctmc, ell_a))
    hat_b = float(np.dot(pi_ctmc, ell_b))

    Gamma_a = posterior_jump(pi_ctmc, ell_a)
    Gamma_b = posterior_jump(pi_ctmc, ell_b)

    ask_allowed = Q > Qmin
    bid_allowed = Q < Qmax

    iq_a = q_to_index(Q - 1, q_grid)
    iq_b = q_to_index(Q + 1, q_grid)

    V_a_nofill = interpolators[iq](Gamma_a)
    V_b_nofill = interpolators[iq](Gamma_b)

    V_a_fill = interpolators[iq_a](Gamma_a) if ask_allowed else V_a_nofill
    V_b_fill = interpolators[iq_b](Gamma_b) if bid_allowed else V_b_nofill

    best_H = -1.0e100
    best_da = float(controls[0])
    best_db = float(controls[0])

    for da in controls:
        pfill_a = math.exp(-k * float(da)) if ask_allowed else 0.0
        pfill_a = min(max(pfill_a, 0.0), 1.0)

        for db in controls:
            pfill_b = math.exp(-k * float(db)) if bid_allowed else 0.0
            pfill_b = min(max(pfill_b, 0.0), 1.0)

            ask_term = hat_a * (
                pfill_a * (float(da) + V_a_fill - V_base)
                + (1.0 - pfill_a) * (V_a_nofill - V_base)
            )

            bid_term = hat_b * (
                pfill_b * (float(db) + V_b_fill - V_base)
                + (1.0 - pfill_b) * (V_b_nofill - V_base)
            )

            H = running + ask_term + bid_term

            if H > best_H:
                best_H = H
                best_da = float(da)
                best_db = float(db)

    V_new = V_base + dt * best_H
    return V_new, best_da, best_db, best_H



def solve(settings):
    if settings.J != 3:
        raise ValueError("This solver is currently implemented only for J=3.")

    if SCIPY_IMPORT_ERROR is not None:
        print("WARNING: scipy interpolation unavailable; falling back to nearest interpolation.")
        print("scipy import error:", SCIPY_IMPORT_ERROR)

    mm = make_clean_params(
        q=settings.q,
        gamma=settings.gamma,
        q_bar=50,
        fixed_psi=None,
        Nt_for_recalib=settings.Nt,
    )

    T = float(getattr(mm, "T", 1.0))
    eta = float(getattr(mm, "eta", 10.0))
    sigma = float(getattr(mm, "sigma", 1.0))
    varphi = float(getattr(mm, "varphi", getattr(mm, "phi", 15.0)))
    psi = float(getattr(mm, "psi", 15.0))
    alpha = float(getattr(mm, "alpha", 0.001))
    q_loading = float(getattr(mm, "qconst", getattr(mm, "q", settings.q)))

    theta, L = make_ou_grid_np(settings.J, eta)
    ell_a, ell_b = state_intensity_coefficients(
        theta=theta,
        sigma=sigma,
        q_loading=q_loading,
        gamma=settings.gamma,
        varphi=varphi,
        psi=psi,
    )

    pi_grid, pi_ij = make_simplex_grid(settings.n_pi)
    q_grid = np.arange(-settings.Qmax, settings.Qmax + 1, dtype=int)
    controls = np.linspace(settings.delta_min, settings.delta_max, settings.delta_points)

    nQ = len(q_grid)
    nPi = len(pi_grid)
    nT = settings.Nt + 1
    dt = T / settings.Nt

    V = np.zeros((nT, nQ, nPi), dtype=float)
    delta_a = np.zeros((settings.Nt, nQ, nPi), dtype=float)
    delta_b = np.zeros((settings.Nt, nQ, nPi), dtype=float)
    H_star = np.zeros((settings.Nt, nQ, nPi), dtype=float)

    # Terminal reduced value.
    for iq, Q in enumerate(q_grid):
        V[-1, iq, :] = -alpha * (Q ** 2)

    for n in range(settings.Nt - 1, -1, -1):
        if n % max(1, settings.Nt // 10) == 0:
            print(f"Backward HJB step {n}/{settings.Nt}", flush=True)

        V_next = V[n + 1]
        interpolators = build_interpolators_for_time(V_next, pi_grid)

        for iq, Q in enumerate(q_grid):
            for ip, pi in enumerate(pi_grid):
                val, da, db, H = bellman_update_ct_hjb(
                    V_next=V_next,
                    interpolators=interpolators,
                    iq=iq,
                    Q=int(Q),
                    pi=pi,
                    pi_grid=pi_grid,
                    theta=theta,
                    L=L,
                    ell_a=ell_a,
                    ell_b=ell_b,
                    controls=controls,
                    mm=mm,
                    dt=dt,
                    q_grid=q_grid,
                    drift_eps=settings.drift_eps,
                )
                V[n, iq, ip] = val
                delta_a[n, iq, ip] = da
                delta_b[n, iq, ip] = db
                H_star[n, iq, ip] = H

    ensure_dir(settings.outdir)

    np.save(os.path.join(settings.outdir, "value.npy"), V)
    np.save(os.path.join(settings.outdir, "delta_a.npy"), delta_a)
    np.save(os.path.join(settings.outdir, "delta_b.npy"), delta_b)
    np.save(os.path.join(settings.outdir, "H_star.npy"), H_star)
    np.save(os.path.join(settings.outdir, "q_grid.npy"), q_grid)
    np.save(os.path.join(settings.outdir, "pi_grid.npy"), pi_grid)
    np.save(os.path.join(settings.outdir, "theta.npy"), theta)
    np.save(os.path.join(settings.outdir, "generator.npy"), L)
    np.save(os.path.join(settings.outdir, "lambda_a.npy"), ell_a)
    np.save(os.path.join(settings.outdir, "lambda_b.npy"), ell_b)

    metadata = {
        "settings": asdict(settings),
        "scheme": "semi_lagrangian_ctmc_only_hjb_v3",
        "T": T,
        "eta": eta,
        "sigma": sigma,
        "varphi": varphi,
        "psi": psi,
        "alpha": alpha,
        "q_loading": q_loading,
        "nQ": nQ,
        "nPi": nPi,
        "controls": controls.tolist(),
        "dt": dt,
        "theta": theta.tolist(),
        "generator": L.tolist(),
        "lambda_a": ell_a.tolist(),
        "lambda_b": ell_b.tolist(),
    }
    with open(os.path.join(settings.outdir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved PDE v2 solution to:", settings.outdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--J", type=int, default=3)
    parser.add_argument("--Qmax", type=int, default=5)
    parser.add_argument("--Nt", type=int, default=20)
    parser.add_argument("--n_pi", type=int, default=8)
    parser.add_argument("--delta_min", type=float, default=0.01)
    parser.add_argument("--delta_max", type=float, default=3.0)
    parser.add_argument("--delta_points", type=int, default=7)
    parser.add_argument("--drift_eps", type=float, default=1e-3)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    settings = PDESettings(
        q=args.q,
        gamma=args.gamma,
        J=args.J,
        Qmax=args.Qmax,
        Nt=args.Nt,
        n_pi=args.n_pi,
        delta_min=args.delta_min,
        delta_max=args.delta_max,
        delta_points=args.delta_points,
        outdir=args.outdir,
        seed=args.seed,
        drift_eps=args.drift_eps,
    )
    solve(settings)


if __name__ == "__main__":
    main()
