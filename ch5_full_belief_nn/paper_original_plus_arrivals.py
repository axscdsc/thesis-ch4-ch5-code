import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Avoid requiring LaTeX in VS Code / Colab.
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.unicode_minus"] = False


BASE_PARAMS_ENV = {
    "k": 1,
    "phi": 0.1,        # running inventory penalty in the paper code
    "alpha": 0.001,
    "q_bar": 50,
    "T": 1,
    "N_t": 1000,
    "N_u": 100,
    "U_max": 10,
    "sigma": 1,
    "eta": 10,
    "qconst": 0.6,
    "mu": 0,
    "Q_0": 0,
    "X_0": 0,
    "S_0": 100,
    "U_0": 0,
    "varphi": 15,
    "total_arrivals": 30,
    "gamma": 1,
    "matrixCJ": [],
}

TABLE1_Q_VALUES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
TABLE1_GAMMA_VALUES = np.array([0.0, 1.0, 2.0, 3.0])

TABLE1_Q_MEANS = {
    "FI":  [21.34, 21.34, 21.34, 21.33, 21.31, 21.30],
    "CJP": [21.34, 21.32, 21.27, 21.18, 21.06, 20.91],
    "PI":  [21.34, 21.32, 21.28, 21.20, 21.14, 21.30],
}
TABLE1_GAMMA_MEANS = {
    "FI":  [21.46, 21.33, 21.17, 21.00],
    "CJP": [21.34, 21.18, 21.01, 20.82],
    "PI":  [21.36, 21.20, 21.03, 20.85],
}


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def mean_sd(x: np.ndarray) -> Tuple[float, float]:
    return float(np.mean(x)), float(np.sqrt(np.var(x)))


def import_original_utils():
    """Import src.utils from the original paper code folder."""
    try:
        import src.utils as utils
        from importlib import reload
        reload(utils)
        return utils
    except Exception as exc:
        msg = (
            "Could not import `src.utils`.\n\n"
            "Please put this file in the same folder as the original paper code, with:\n"
            "    src/utils.py\n"
            "available. The original notebook imports it using `import src.utils as utils`.\n\n"
            f"Original error: {repr(exc)}"
        )
        raise ImportError(msg)

@dataclass
class PaperPoint:
    x: float
    FI_mean: float
    FI_sd: float
    CJP_mean: float
    CJP_sd: float
    PI_mean: float
    PI_sd: float
    psi: float
    varphi: float
    qconst: float
    gamma: float


def build_base_mm(utils):
    params_env = BASE_PARAMS_ENV.copy()
    return utils.mm_with_fads(**params_env)


def run_original_paper_point(utils, params_env_base: dict, sims: int, seed: int) -> Tuple[PaperPoint, object]:
    mm_aux = utils.mm_with_fads(**params_env_base)
    np.random.seed(seed)
    PnL_val = mm_aux.PnL(sims)

    FI_mean, FI_sd = mean_sd(PnL_val[0])
    CJP_mean, CJP_sd = mean_sd(PnL_val[1])
    PI_mean, PI_sd = mean_sd(PnL_val[2])

    point = PaperPoint(
        x=float(params_env_base.get("qconst", np.nan)),
        FI_mean=FI_mean, FI_sd=FI_sd,
        CJP_mean=CJP_mean, CJP_sd=CJP_sd,
        PI_mean=PI_mean, PI_sd=PI_sd,
        psi=float(getattr(mm_aux, "psi", np.nan)),
        varphi=float(getattr(mm_aux, "varphi", params_env_base.get("varphi", np.nan))),
        qconst=float(getattr(mm_aux, "qconst", params_env_base.get("qconst", np.nan))),
        gamma=float(getattr(mm_aux, "gamma", params_env_base.get("gamma", np.nan))),
    )
    return point, mm_aux


def original_q_sweep(utils, sims: int, seed: int, q_values: np.ndarray) -> Tuple[List[PaperPoint], List[object]]:
    mm_base = build_base_mm(utils)
    params_env_cop = BASE_PARAMS_ENV.copy()
    params_env_cop["matrixCJ"] = np.array(mm_base.get_delta_CJ_matrix())

    points = []
    mms = []
    for q in q_values:
        params_aux = params_env_cop.copy()
        params_aux["qconst"] = float(q)
        params_aux["gamma"] = 1.0
        point, mm_aux = run_original_paper_point(utils, params_aux, sims=sims, seed=seed)

        # Make the stored metadata and the mm object explicit and consistent.
        point.x = float(q)
        point.qconst = float(q)
        point.gamma = 1.0
        mm_aux.qconst = float(q)
        mm_aux.gamma = 1.0

        points.append(point)
        mms.append(mm_aux)
        print(
            f"q={q:.2f} | FI={point.FI_mean:.5f} CJP={point.CJP_mean:.5f} "
            f"PI={point.PI_mean:.5f} psi={point.psi:.5f} gamma={point.gamma:.2f}"
        )
    return points, mms


def original_gamma_sweep(utils, sims: int, seed: int, gamma_values: np.ndarray) -> Tuple[List[PaperPoint], List[object]]:
    mm_base = build_base_mm(utils)
    params_env_cop = BASE_PARAMS_ENV.copy()
    params_env_cop["matrixCJ"] = np.array(mm_base.get_delta_CJ_matrix())
    params_env_cop["qconst"] = 0.6

    points = []
    mms = []
    for gamma in gamma_values:
        params_aux = params_env_cop.copy()
        params_aux["qconst"] = 0.6
        params_aux["gamma"] = float(gamma)
        point, mm_aux = run_original_paper_point(utils, params_aux, sims=sims, seed=seed)

        # Make the stored metadata and the mm object explicit and consistent.
        point.x = float(gamma)
        point.qconst = 0.6
        point.gamma = float(gamma)
        mm_aux.qconst = 0.6
        mm_aux.gamma = float(gamma)

        points.append(point)
        mms.append(mm_aux)
        print(
            f"gamma={gamma:.2f} | FI={point.FI_mean:.5f} CJP={point.CJP_mean:.5f} "
            f"PI={point.PI_mean:.5f} psi={point.psi:.5f} q={point.qconst:.2f}"
        )
    return points, mms

def stationary_distribution_from_generator(L: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(L.T)
    idx = int(np.argmin(np.abs(eigvals)))
    pi = np.real(eigvecs[:, idx])
    pi = np.maximum(pi, 0.0)

    if pi.sum() <= 1e-14:
        pi = np.abs(np.real(eigvecs[:, idx]))

    return pi / pi.sum()


def generator_from_grid(theta: np.ndarray, eta: float) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    J = len(theta)
    L = np.zeros((J, J), dtype=float)

    for i in range(J):
        b = -eta * theta[i]

        if i == 0:
            h = theta[1] - theta[0]
            r = max(b / h + 0.5 / (h * h), 0.0)
            L[i, i + 1] = r

        elif i == J - 1:
            h = theta[-1] - theta[-2]
            l = max(-b / h + 0.5 / (h * h), 0.0)
            L[i, i - 1] = l

        else:
            hm = theta[i] - theta[i - 1]
            hp = theta[i + 1] - theta[i]

            r = (1.0 + b * hm) / (hp * (hp + hm))
            l = (1.0 - b * hp) / (hm * (hp + hm))

            L[i, i + 1] = max(r, 0.0)
            L[i, i - 1] = max(l, 0.0)

        L[i, i] = -np.sum(L[i, :])

    return L

def create_ctmc_discretization(J: int, eta: float):
    if J == 1:
        theta = np.array([0.0])
        L = np.array([[0.0]])
        pi0 = np.array([1.0])
        return theta, L, pi0

    std_u = np.sqrt(1.0 / (2.0 * eta))
    theta = np.linspace(-3.0 * std_u, 3.0 * std_u, J)

    L = generator_from_grid(theta, eta)
    pi = stationary_distribution_from_generator(L)

    return theta, L, pi
  
def solve_A_coeff(T, Nt, alpha, phi, varphi, psi, k):
    dt = T / Nt
    kappa = 4.0 * np.exp(-1.0) * (varphi + psi) * k
    A = np.empty(Nt + 1)
    A[Nt] = -alpha
    for i in range(Nt - 1, -1, -1):
        A_next = A[i + 1]
        # A'(t) = phi - kappa A(t)^2, integrated backward.
        A[i] = A_next - dt * (phi - kappa * A_next ** 2)
    return A


def solve_B_coeff(T, Nt, eta, mu, sigma, q, gamma, phi, varphi, psi, k, A):
    dt = T / Nt
    kappa = 4.0 * np.exp(-1.0) * (varphi + psi) * k
    b0 = np.zeros(Nt + 1)
    b1 = np.zeros(Nt + 1)

    for i in range(Nt - 1, -1, -1):
        A_next = A[i + 1]
        b0_next = b0[i + 1]
        b1_next = b1[i + 1]

        # Chosen to match the reduced quadratic approximation convention used in our ARR plug-in.
        b0_prime = -mu - kappa * A_next * b0_next
        b1_prime = (
            eta * sigma * q
            + eta * b1_next
            - kappa * A_next * b1_next
            - 4.0 * np.exp(-1.0) * psi * q * sigma * gamma * A_next
            - 4.0 * np.exp(-1.0) * k * gamma * q * sigma * psi * A_next ** 2
        )
        b0[i] = b0_next - dt * b0_prime
        b1[i] = b1_next - dt * b1_prime
    return b0, b1


def quote_from_coeff(Q, U_est, A, b0, b1, k, t_idx):
    B = b0[t_idx] + b1[t_idx] * U_est
    rho_a = 1.0 / k + (2.0 * Q - 1.0) * A[t_idx] + B
    rho_b = 1.0 / k - (2.0 * Q + 1.0) * A[t_idx] - B
    return np.maximum(rho_a, 1e-8), np.maximum(rho_b, 1e-8)


def simulate_arrivals_filter_from_mm(mm_aux, sims: int, Nt: int, J: int, seed: int, verbose: bool = True) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)

    T = float(getattr(mm_aux, "T", BASE_PARAMS_ENV["T"]))
    dt = T / Nt
    k = float(getattr(mm_aux, "k", BASE_PARAMS_ENV["k"]))
    phi = float(getattr(mm_aux, "phi", BASE_PARAMS_ENV["phi"]))
    alpha = float(getattr(mm_aux, "alpha", BASE_PARAMS_ENV["alpha"]))
    q_bar = int(getattr(mm_aux, "q_bar", BASE_PARAMS_ENV["q_bar"]))
    Q_min, Q_max = -q_bar, q_bar
    sigma = float(getattr(mm_aux, "sigma", BASE_PARAMS_ENV["sigma"]))
    eta = float(getattr(mm_aux, "eta", BASE_PARAMS_ENV["eta"]))
    qconst = float(getattr(mm_aux, "qconst", BASE_PARAMS_ENV["qconst"]))
    pconst = np.sqrt(max(1.0 - qconst ** 2, 0.0))
    mu = float(getattr(mm_aux, "mu", BASE_PARAMS_ENV["mu"]))
    S0 = float(getattr(mm_aux, "S_0", BASE_PARAMS_ENV["S_0"]))
    U0 = float(getattr(mm_aux, "U_0", BASE_PARAMS_ENV["U_0"]))
    Q0 = float(getattr(mm_aux, "Q_0", BASE_PARAMS_ENV["Q_0"]))
    X0 = float(getattr(mm_aux, "X_0", BASE_PARAMS_ENV["X_0"]))
    varphi = float(getattr(mm_aux, "varphi", BASE_PARAMS_ENV["varphi"]))
    psi = float(getattr(mm_aux, "psi", np.nan))
    gamma = float(getattr(mm_aux, "gamma", BASE_PARAMS_ENV["gamma"]))

    if not np.isfinite(psi):
        raise ValueError("Could not read mm_aux.psi. ARR needs the paper-rescaled psi from the original code.")

    A = solve_A_coeff(T, Nt, alpha, phi, varphi, psi, k)
    b0, b1 = solve_B_coeff(T, Nt, eta, mu, sigma, qconst, gamma, phi, varphi, psi, k, A)

    theta, L, pi0 = create_ctmc_discretization(J=J, eta=eta)
    trans = expm(L * dt)

    S = np.full(sims, S0, dtype=float)
    U = np.full(sims, U0, dtype=float)
    Q = np.full(sims, Q0, dtype=float)
    X = np.full(sims, X0, dtype=float)
    int_Q2 = np.zeros(sims, dtype=float)
    pi = np.tile(pi0, (sims, 1))

    iterator = tqdm(range(Nt), desc="ARR simulation") if verbose else range(Nt)
    for it in iterator:
        S_prev = S.copy()
        U_prev = U.copy()

        U_est = pi @ theta
        rho_a, rho_b = quote_from_coeff(Q, U_est, A, b0, b1, k, it)

        allow_a = (Q > Q_min).astype(float)
        allow_b = (Q < Q_max).astype(float)

        fad_true = gamma * sigma * qconst * U
        lam_a = (varphi * np.exp(-k * rho_a) + psi * np.exp(-k * rho_a - fad_true)) * allow_a
        lam_b = (varphi * np.exp(-k * rho_b) + psi * np.exp(-k * rho_b + fad_true)) * allow_b

        dNa = rng.poisson(np.maximum(lam_a * dt, 0.0)).astype(float)
        dNb = rng.poisson(np.maximum(lam_b * dt, 0.0)).astype(float)

        # Cap jumps to preserve inventory bounds.
        dNa = np.minimum(dNa, np.maximum(Q - Q_min, 0.0))
        dNb = np.minimum(dNb, np.maximum(Q_max - Q, 0.0))

        X += dNa * (S_prev + rho_a) - dNb * (S_prev - rho_b)
        Q += dNb - dNa
        int_Q2 += Q ** 2 * dt

        # Update true state and price.
        dB = rng.normal(0.0, np.sqrt(dt), sims)
        dZ = rng.normal(0.0, np.sqrt(dt), sims)
        U = U_prev - eta * U_prev * dt + dB
        S = S_prev + mu * dt + sigma * (qconst * (U - U_prev) + pconst * dZ)

        # Vectorized over paths and states.
        pi_pred = pi @ trans
        rho_a_col = rho_a[:, None]
        rho_b_col = rho_b[:, None]
        allow_a_col = allow_a[:, None]
        allow_b_col = allow_b[:, None]
        fad_j = gamma * sigma * qconst * theta[None, :]

        lam_a_j = (varphi * np.exp(-k * rho_a_col) + psi * np.exp(-k * rho_a_col - fad_j)) * allow_a_col
        lam_b_j = (varphi * np.exp(-k * rho_b_col) + psi * np.exp(-k * rho_b_col + fad_j)) * allow_b_col
        lam_a_j = np.maximum(lam_a_j, 1e-12)
        lam_b_j = np.maximum(lam_b_j, 1e-12)

        log_like = (
            dNa[:, None] * np.log(lam_a_j * dt) - lam_a_j * dt
            + dNb[:, None] * np.log(lam_b_j * dt) - lam_b_j * dt
        )
        log_like -= np.max(log_like, axis=1, keepdims=True)
        weights = pi_pred * np.exp(log_like)
        denom = weights.sum(axis=1, keepdims=True)
        bad = denom[:, 0] <= 1e-14
        pi = weights / np.maximum(denom, 1e-14)
        if np.any(bad):
            pi[bad, :] = pi0

    perf = X + Q * S - alpha * Q ** 2 - phi * int_Q2
    return mean_sd(perf)

def points_to_dict(points: List[PaperPoint]) -> dict:
    return {
        "x": [p.x for p in points],
        "FI_mean": [p.FI_mean for p in points],
        "FI_sd": [p.FI_sd for p in points],
        "CJP_mean": [p.CJP_mean for p in points],
        "CJP_sd": [p.CJP_sd for p in points],
        "PI_mean": [p.PI_mean for p in points],
        "PI_sd": [p.PI_sd for p in points],
        "psi": [p.psi for p in points],
        "varphi": [p.varphi for p in points],
        "qconst": [p.qconst for p in points],
        "gamma": [p.gamma for p in points],
    }


def save_json(obj: dict, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_sweep(x, series: Dict[str, List[float]], xlabel: str, title: str, outfile: str):
    markers = {"FI": "o", "CJP": "^", "PI": "s", "ARR": "D"}
    plt.figure(figsize=(6, 4))
    for name, y in series.items():
        plt.plot(x, y, marker=markers.get(name, "o"), linewidth=1.8, markersize=5, label=name)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel("average performance", fontsize=13)
    plt.title(title, fontsize=13)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def run_q(mode: str, sims_paper: int, sims_arr: int, Nt_arr: int, J: int, seed: int, outdir: str):
    utils = import_original_utils()
    ensure_dir(outdir)
    print("\n" + "=" * 70)
    print("Q SWEEP: original paper code" + (" + ARR" if "arrivals" in mode else ""))
    print("=" * 70)

    points, mms = original_q_sweep(utils, sims=sims_paper, seed=seed, q_values=TABLE1_Q_VALUES)
    data = points_to_dict(points)

    series = {
        "FI": data["FI_mean"],
        "CJP": data["CJP_mean"],
        "PI": data["PI_mean"],
    }

    if "arrivals" in mode:
        arr_mean = []
        arr_sd = []
        for q, mm_aux in zip(TABLE1_Q_VALUES, mms):
            # Explicitly enforce the q-sweep convention for ARR: q varies and gamma=1.
            mm_aux.qconst = float(q)
            mm_aux.gamma = 1.0
            print(
                f"\nARR q={q:.2f}, using original-code psi={float(getattr(mm_aux, 'psi')):.5f}, "
                f"gamma={float(getattr(mm_aux, 'gamma')):.2f}"
            )
            m, s = simulate_arrivals_filter_from_mm(mm_aux, sims=sims_arr, Nt=Nt_arr, J=J, seed=seed, verbose=True)
            arr_mean.append(m)
            arr_sd.append(s)
            print(f"ARR q={q:.2f} | mean={m:.5f}, sd={s:.5f}")
        data["ARR_mean"] = arr_mean
        data["ARR_sd"] = arr_sd
        series["ARR"] = arr_mean

    save_json(data, os.path.join(outdir, "q_sweep_results.json"))
    plot_sweep(TABLE1_Q_VALUES, series, "q", "q sweep: original FI/CJP/PI" + (" + ARR" if "arrivals" in mode else ""), os.path.join(outdir, "q_sweep.png"))

    # Also save paper Table 1 reference for quick visual check.
    ref_series = dict(TABLE1_Q_MEANS)
    plot_sweep(TABLE1_Q_VALUES, ref_series, "q", "Table 1 reference: q sweep", os.path.join(outdir, "q_sweep_table1_reference.png"))


def run_gamma(mode: str, sims_paper: int, sims_arr: int, Nt_arr: int, J: int, seed: int, outdir: str):
    utils = import_original_utils()
    ensure_dir(outdir)
    print("\n" + "=" * 70)
    print("GAMMA SWEEP: original paper code" + (" + ARR" if "arrivals" in mode else ""))
    print("=" * 70)

    points, mms = original_gamma_sweep(utils, sims=sims_paper, seed=seed, gamma_values=TABLE1_GAMMA_VALUES)
    data = points_to_dict(points)

    series = {
        "FI": data["FI_mean"],
        "CJP": data["CJP_mean"],
        "PI": data["PI_mean"],
    }

    if "arrivals" in mode:
        arr_mean = []
        arr_sd = []
        for gamma, mm_aux in zip(TABLE1_GAMMA_VALUES, mms):
            # Explicitly enforce the gamma-sweep convention for ARR: q=0.6 and gamma varies.
            mm_aux.qconst = 0.6
            mm_aux.gamma = float(gamma)
            print(
                f"\nARR gamma={gamma:.2f}, using original-code psi={float(getattr(mm_aux, 'psi')):.5f}, "
                f"q={float(getattr(mm_aux, 'qconst')):.2f}"
            )
            m, s = simulate_arrivals_filter_from_mm(mm_aux, sims=sims_arr, Nt=Nt_arr, J=J, seed=seed, verbose=True)
            arr_mean.append(m)
            arr_sd.append(s)
            print(f"ARR gamma={gamma:.2f} | mean={m:.5f}, sd={s:.5f}")
        data["ARR_mean"] = arr_mean
        data["ARR_sd"] = arr_sd
        series["ARR"] = arr_mean

    save_json(data, os.path.join(outdir, "gamma_sweep_results.json"))
    plot_sweep(TABLE1_GAMMA_VALUES, series, "gamma", "gamma sweep: original FI/CJP/PI" + (" + ARR" if "arrivals" in mode else ""), os.path.join(outdir, "gamma_sweep.png"))

    ref_series = dict(TABLE1_GAMMA_MEANS)
    plot_sweep(TABLE1_GAMMA_VALUES, ref_series, "gamma", "Table 1 reference: gamma sweep", os.path.join(outdir, "gamma_sweep_table1_reference.png"))


def build_parser():
    parser = argparse.ArgumentParser(description="Original paper code + arrivals-filter extension")
    parser.add_argument("--mode", type=str, default="q_plus_arrivals",
                        choices=["q_paper", "gamma_paper", "q_plus_arrivals", "gamma_plus_arrivals", "all"])
    parser.add_argument("--sims_paper", type=int, default=100000,
                        help="Monte Carlo paths for original paper FI/CJP/PI via mm_aux.PnL")
    parser.add_argument("--sims_arr", type=int, default=10000,
                        help="Monte Carlo paths for ARR extension")
    parser.add_argument("--Nt_arr", type=int, default=1000,
                        help="Time steps for ARR extension")
    parser.add_argument("--J", type=int, default=7,
                        help="Number of CTMC states for arrivals filter")
    parser.add_argument("--seed", type=int, default=36457656)
    parser.add_argument("--outdir", type=str, default="outputs_original_plus_arrivals")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.mode == "q_paper":
        run_q(args.mode, args.sims_paper, args.sims_arr, args.Nt_arr, args.J, args.seed, args.outdir)
    elif args.mode == "gamma_paper":
        run_gamma(args.mode, args.sims_paper, args.sims_arr, args.Nt_arr, args.J, args.seed, args.outdir)
    elif args.mode == "q_plus_arrivals":
        run_q(args.mode, args.sims_paper, args.sims_arr, args.Nt_arr, args.J, args.seed, args.outdir)
    elif args.mode == "gamma_plus_arrivals":
        run_gamma(args.mode, args.sims_paper, args.sims_arr, args.Nt_arr, args.J, args.seed, args.outdir)
    elif args.mode == "all":
        run_q("q_plus_arrivals", args.sims_paper, args.sims_arr, args.Nt_arr, args.J, args.seed, os.path.join(args.outdir, "q"))
        run_gamma("gamma_plus_arrivals", args.sims_paper, args.sims_arr, args.Nt_arr, args.J, args.seed, os.path.join(args.outdir, "gamma"))
