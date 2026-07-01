from __future__ import annotations

import csv
import importlib
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.stats import norm


BASE_PARAMS_ENV: Dict = {
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
    "varphi": 15,     # uninformed baseline arrival coefficient
    "total_arrivals": 30,
    "gamma": 1,
    "matrixCJ": [],
}



@dataclass
class CleanParams:
    k: float = 1.0
    phi: float = 0.1          # running inventory penalty
    alpha: float = 0.001
    q_bar: int = 50
    T: float = 1.0
    sigma: float = 1.0
    eta: float = 10.0
    qconst: float = 0.6
    mu: float = 0.0
    Q_0: float = 0.0
    X_0: float = 0.0
    S_0: float = 100.0
    U_0: float = 0.0
    varphi: float = 15.0     # uninformed baseline arrival coefficient
    total_arrivals: float = 30.0
    gamma: float = 1.0       # raw gamma, not multiplied by q sigma
    psi: float = 0.0


def recalibrated_psi(T: float, Nt: int, eta: float, qconst: float, sigma: float,
                     gamma: float, varphi: float, total_arrivals: float) -> float:
    """Paper-style recalibration of psi to keep total expected arrivals comparable."""
    t_grid = np.linspace(0.0, T, Nt)
    var_u = (1.0 - np.exp(-2.0 * eta * t_grid)) / (2.0 * eta)
    effective_gamma = gamma * qconst * sigma
    aux_integral = T * float(np.mean(np.exp(0.5 * effective_gamma ** 2 * var_u)))
    return (total_arrivals * T - varphi * T) / aux_integral


def make_clean_params(
    q: float,
    gamma: float,
    q_bar: int = 50,
    fixed_psi: Optional[float] = None,
    Nt_for_recalib: int = 1000,
    eta: float = 10.0,
    sigma: float = 1.0,
) -> CleanParams:
    p = CleanParams(
        qconst=float(q),
        gamma=float(gamma),
        q_bar=int(q_bar),
        eta=float(eta),
        sigma=float(sigma),
    )
    p.psi = float(fixed_psi) if fixed_psi is not None else recalibrated_psi(
        p.T, Nt_for_recalib, p.eta, p.qconst, p.sigma, p.gamma, p.varphi, p.total_arrivals
    )
    return p


TABLE_Q_VALUES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
TABLE_GAMMA_VALUES = np.array([0.0, 1.0, 2.0, 3.0])


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def mean_sd(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.sqrt(np.var(x)))


def import_original_utils():
    """Import src.utils from the current project folder."""
    import src.utils as utils
    importlib.reload(utils)
    return utils


def build_base_params_with_matrix(utils, q_bar: Optional[int] = None) -> Dict:
    params = BASE_PARAMS_ENV.copy()
    if q_bar is not None:
        params["q_bar"] = int(q_bar)
    mm_base = utils.mm_with_fads(**params)
    params["matrixCJ"] = np.array(mm_base.get_delta_CJ_matrix())
    return params


def make_mm(utils, q: float, gamma: float, q_bar: Optional[int] = None, fixed_psi: Optional[float] = None):
    """Build an original-paper mm_with_fads object.

    Important: src.utils.mm_with_fads stores an *effective* gamma internally,
    gamma_eff = raw_gamma * qconst * sigma.  Therefore we do not overwrite
    mm.gamma after construction.  This helper is kept only for original-paper
    compatibility checks; the clean FI-plug-in simulations use CleanParams.
    """
    params = build_base_params_with_matrix(utils, q_bar=q_bar)
    params["qconst"] = float(q)
    params["gamma"] = float(gamma)
    mm = utils.mm_with_fads(**params)
    if q_bar is not None:
        mm.q_bar = int(q_bar)
    if fixed_psi is not None:
        mm.psi = float(fixed_psi)
    return mm


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_sided_pvalue_from_z(z: float) -> float:
    return 2.0 * (1.0 - normal_cdf(abs(float(z))))


def benjamini_hochberg_mask(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float).ravel()
    m = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    threshold = alpha * np.arange(1, m + 1) / m
    passed = sorted_p <= threshold
    reject = np.zeros(m, dtype=bool)
    if np.any(passed):
        kmax = int(np.max(np.where(passed)[0]))
        reject[order[: kmax + 1]] = True
    return reject


def save_csv_rows(path: str, rows: List[Dict]):
    ensure_dir(os.path.dirname(path) or ".")
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_csv_matrix(path: str, q_values: np.ndarray, gamma_values: np.ndarray, matrix: np.ndarray):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gamma\\q"] + [f"{q:.6g}" for q in q_values])
        for i, g in enumerate(gamma_values):
            writer.writerow([f"{g:.6g}"] + [f"{matrix[i, j]:.10g}" for j in range(len(q_values))])



def _lazy_pyplot():
    # Keep matplotlib out of the import path.  Some clusters build the font cache
    # slowly on first import; importing only when plots are requested makes the
    # numerical jobs safer.
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt

def plot_lines(x: np.ndarray, series: Dict[str, Iterable[float]], xlabel: str, ylabel: str, title: str, outfile: str):
    markers = {"FI": "o", "CJP": "^", "PI": "s", "ARR": "D", "PI-FIplug": "s", "ARR-FIplug": "D"}
    plt = _lazy_pyplot()
    plt.figure(figsize=(6.5, 4.3))
    for name, y in series.items():
        plt.plot(x, list(y), marker=markers.get(name, "o"), linewidth=1.8, markersize=5, label=name)
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=13)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


def plot_heatmap(q_values: np.ndarray, gamma_values: np.ndarray, matrix: np.ndarray,
                 outfile: str, title: str, colorbar_label: str,
                 cmap: str = "RdBu_r", symmetric: bool = True):
    if symmetric:
        vmax = float(np.max(np.abs(matrix)))
        if vmax < 1e-14:
            vmax = 1e-14
        vmin = -vmax
    else:
        vmin = float(np.min(matrix))
        vmax = float(np.max(matrix))
        if abs(vmax - vmin) < 1e-14:
            vmax = vmin + 1e-14

    plt = _lazy_pyplot()
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[q_values[0], q_values[-1], gamma_values[0], gamma_values[-1]],
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, fontsize=14)
    ax.set_xlabel(r"$q$", fontsize=18)
    ax.set_ylabel(r"$\gamma$", fontsize=18)
    ax.set_title(title, fontsize=15)
    if len(q_values) <= 11:
        ax.set_xticks(q_values)
    if len(gamma_values) <= 13:
        ax.set_yticks(gamma_values)
    ax.tick_params(axis="both", labelsize=12)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()


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
    if J == 1:
        return np.zeros((1, 1))
    L = np.zeros((J, J), dtype=float)
    for i in range(J):
        b = -eta * theta[i]
        if i == 0:
            h = theta[1] - theta[0]
            r = max(b / h + 0.5 / (h * h), 0.0)
            L[i, i + 1] = r
        elif i == J - 1:
            h = theta[-1] - theta[-2]
            ell = max(-b / h + 0.5 / (h * h), 0.0)
            L[i, i - 1] = ell
        else:
            hm = theta[i] - theta[i - 1]
            hp = theta[i + 1] - theta[i]
            r = (1.0 + b * hm) / (hp * (hp + hm))
            ell = (1.0 - b * hp) / (hm * (hp + hm))
            L[i, i + 1] = max(r, 0.0)
            L[i, i - 1] = max(ell, 0.0)
        L[i, i] = -np.sum(L[i, :])
    return L


def create_ctmc_discretization(J: int, eta: float, grid: str = "equidistant") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if J == 1:
        theta = np.array([0.0])
        return theta, np.zeros((1, 1)), np.array([1.0])

    std_u = math.sqrt(1.0 / (2.0 * eta))
    if grid == "equidistant":
        theta = np.linspace(-3.0 * std_u, 3.0 * std_u, J)
        L = generator_from_grid(theta, eta)
        pi0 = stationary_distribution_from_generator(L)
        return theta, L, pi0

    if grid == "equal_probability":
        probs = np.linspace(0.0, 1.0, J + 1)
        edges = std_u * norm.ppf(probs)
        theta = np.zeros(J)
        for j in range(J):
            a = edges[j] / std_u
            b = edges[j + 1] / std_u
            denom = norm.cdf(b) - norm.cdf(a)
            theta[j] = std_u * (norm.pdf(a) - norm.pdf(b)) / denom
        L = generator_from_grid(theta, eta)
        pi0 = np.ones(J) / J
        return theta, L, pi0

    raise ValueError("grid must be 'equidistant' or 'equal_probability'")


def solve_A_coeff(T: float, Nt: int, alpha: float, phi: float, varphi: float, psi: float, k: float) -> np.ndarray:
    """Approximate FI coefficient A(t), terminal A(T)=-alpha."""
    dt = T / Nt
    kappa = 4.0 * math.exp(-1.0) * (varphi + psi) * k
    A = np.empty(Nt + 1)
    A[Nt] = -alpha
    for i in range(Nt - 1, -1, -1):
        A_next = A[i + 1]
        # A'(t) = phi - kappa A(t)^2, integrated backward.
        A[i] = A_next - dt * (phi - kappa * A_next ** 2)
    return A


def solve_B_coeff(T: float, Nt: int, eta: float, mu: float, sigma: float, q: float,
                  gamma: float, phi: float, varphi: float, psi: float, k: float,
                  A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Approximate FI coefficient B(t,u)=b0(t)+b1(t)u."""
    dt = T / Nt
    kappa = 4.0 * math.exp(-1.0) * (varphi + psi) * k
    b0 = np.zeros(Nt + 1)
    b1 = np.zeros(Nt + 1)
    for i in range(Nt - 1, -1, -1):
        A_next = A[i + 1]
        b0_next = b0[i + 1]
        b1_next = b1[i + 1]
        b0_prime = -mu - kappa * A_next * b0_next
        b1_prime = (
            eta * sigma * q
            + eta * b1_next
            - kappa * A_next * b1_next
            - 4.0 * math.exp(-1.0) * psi * q * sigma * gamma * A_next
            - 4.0 * math.exp(-1.0) * k * gamma * q * sigma * psi * A_next ** 2
        )
        b0[i] = b0_next - dt * b0_prime
        b1[i] = b1_next - dt * b1_prime
    return b0, b1


def quote_from_fi_coeff(Q: np.ndarray, U_est: np.ndarray, A: np.ndarray,
                        b0: np.ndarray, b1: np.ndarray, k: float, t_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Common FI plug-in quote map evaluated at a filtered estimate U_est."""
    B = b0[t_idx] + b1[t_idx] * U_est
    rho_a = 1.0 / k + (2.0 * Q - 1.0) * A[t_idx] + B
    rho_b = 1.0 / k - (2.0 * Q + 1.0) * A[t_idx] - B
    return np.maximum(rho_a, 1e-8), np.maximum(rho_b, 1e-8)


def coefficients_from_mm(mm_aux, Nt: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = float(getattr(mm_aux, "T", 1.0))
    k = float(getattr(mm_aux, "k", 1.0))
    phi = float(getattr(mm_aux, "phi", 0.1))
    alpha = float(getattr(mm_aux, "alpha", 0.001))
    eta = float(getattr(mm_aux, "eta", 10.0))
    sigma = float(getattr(mm_aux, "sigma", 1.0))
    qconst = float(getattr(mm_aux, "qconst", 0.6))
    mu = float(getattr(mm_aux, "mu", 0.0))
    varphi = float(getattr(mm_aux, "varphi", 15.0))
    psi = float(getattr(mm_aux, "psi"))
    gamma = float(getattr(mm_aux, "gamma", 1.0))
    A = solve_A_coeff(T, Nt, alpha, phi, varphi, psi, k)
    b0, b1 = solve_B_coeff(T, Nt, eta, mu, sigma, qconst, gamma, phi, varphi, psi, k, A)
    return A, b0, b1

def kalman_riccati_array(T: float, Nt: int, eta: float, qconst: float) -> np.ndarray:
    """
    Riccati equation for price-based filter.

    With dY_t=(dS_t-mu dt)/sigma=q dU_t+p dZ_t and dU_t=-eta U_t dt+dB_t,
    the observation noise is correlated with the state noise.  The scalar
    Riccati equation is
        P' = -2 eta P + 1 - q^2 (1 - eta P)^2.
    """
    dt = T / Nt
    P = np.zeros(Nt + 1)
    for i in range(Nt):
        p = P[i]
        dP = -2.0 * eta * p + 1.0 - (qconst ** 2) * (1.0 - eta * p) ** 2
        P[i + 1] = max(p + dt * dP, 0.0)
    return P


def kalman_update(Uhat: np.ndarray, dY: np.ndarray, P_t: float, dt: float, eta: float, qconst: float) -> np.ndarray:
    K = qconst * (1.0 - eta * P_t)
    return Uhat - eta * Uhat * dt + K * (dY + qconst * eta * Uhat * dt)


@dataclass
class SimResult:
    mean: float
    sd: float
    paths: Optional[np.ndarray] = None

def simulate_common_fiplug_strategy(
    mm_aux,
    strategy: str,
    sims: int,
    Nt: int,
    J: int,
    seed: int,
    grid: str = "equidistant",
    return_paths: bool = False,
    verbose: bool = False,
) -> SimResult:

    if strategy not in {"pi", "arr"}:
        raise ValueError("strategy must be 'pi' or 'arr'")

    rng = np.random.default_rng(seed)

    T = float(getattr(mm_aux, "T", 1.0))
    dt = T / Nt
    k = float(getattr(mm_aux, "k", 1.0))
    phi = float(getattr(mm_aux, "phi", 0.1))
    alpha = float(getattr(mm_aux, "alpha", 0.001))
    q_bar = int(getattr(mm_aux, "q_bar", 50))
    Q_min, Q_max = -q_bar, q_bar
    sigma = float(getattr(mm_aux, "sigma", 1.0))
    eta = float(getattr(mm_aux, "eta", 10.0))
    qconst = float(getattr(mm_aux, "qconst", 0.6))
    pconst = math.sqrt(max(1.0 - qconst ** 2, 0.0))
    mu = float(getattr(mm_aux, "mu", 0.0))
    S0 = float(getattr(mm_aux, "S_0", 100.0))
    U0 = float(getattr(mm_aux, "U_0", 0.0))
    Q0 = float(getattr(mm_aux, "Q_0", 0.0))
    X0 = float(getattr(mm_aux, "X_0", 0.0))
    varphi = float(getattr(mm_aux, "varphi", 15.0))
    psi = float(getattr(mm_aux, "psi"))
    gamma = float(getattr(mm_aux, "gamma", 1.0))

    A, b0, b1 = coefficients_from_mm(mm_aux, Nt)
    P = kalman_riccati_array(T, Nt, eta, qconst)
    theta, L, pi0 = create_ctmc_discretization(J, eta, grid=grid)
    trans = expm(L * dt)

    S = np.full(sims, S0, dtype=float)
    U = np.full(sims, U0, dtype=float)
    Q = np.full(sims, Q0, dtype=float)
    X = np.full(sims, X0, dtype=float)
    int_Q2 = np.zeros(sims, dtype=float)
    Uhat_PI = np.full(sims, U0, dtype=float)
    pi = np.tile(pi0, (sims, 1))

    loop = range(Nt)
    if verbose:
        from tqdm import tqdm
        loop = tqdm(loop, desc=f"{strategy.upper()}-FIplug")

    for it in loop:
        S_prev = S.copy()
        U_prev = U.copy()

        if strategy == "pi":
            U_est = Uhat_PI
        else:
            U_est = pi @ theta

        rho_a, rho_b = quote_from_fi_coeff(Q, U_est, A, b0, b1, k, it)

        allow_a = (Q > Q_min).astype(float)
        allow_b = (Q < Q_max).astype(float)
        fad_true = gamma * sigma * qconst * U
        lam_a = (varphi * np.exp(-k * rho_a) + psi * np.exp(-k * rho_a - fad_true)) * allow_a
        lam_b = (varphi * np.exp(-k * rho_b) + psi * np.exp(-k * rho_b + fad_true)) * allow_b

        dNa = rng.poisson(np.maximum(lam_a * dt, 0.0)).astype(float)
        dNb = rng.poisson(np.maximum(lam_b * dt, 0.0)).astype(float)
        dNa = np.minimum(dNa, np.maximum(Q - Q_min, 0.0))
        dNb = np.minimum(dNb, np.maximum(Q_max - Q, 0.0))

        X += dNa * (S_prev + rho_a) - dNb * (S_prev - rho_b)
        Q += dNb - dNa
        int_Q2 += Q ** 2 * dt

        dB = rng.normal(0.0, math.sqrt(dt), sims)
        dZ = rng.normal(0.0, math.sqrt(dt), sims)
        U = U_prev - eta * U_prev * dt + dB
        dS = mu * dt + sigma * (qconst * (U - U_prev) + pconst * dZ)
        S = S_prev + dS
        dY = (dS - mu * dt) / sigma

        # Price filter update.  We update it in both strategies for diagnostics;
        # only the PI strategy uses it for quotes.
        Uhat_PI = kalman_update(Uhat_PI, dY, P[it], dt, eta, qconst)

        # Arrival filter update.  We update it in both strategies for diagnostics;
        # only the ARR strategy uses it for quotes.
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
    m, s = mean_sd(perf)
    return SimResult(mean=m, sd=s, paths=perf if return_paths else None)

def simulate_filter_mse(
    mm_aux,
    sims: int,
    Nt: int,
    J: int,
    seed: int,
    grid: str = "equidistant",
    quote_policy: str = "zero_fiplug",
    return_path_diffs: bool = False,
) -> Dict[str, object]:

    if quote_policy not in {"zero_fiplug", "arr_fiplug", "pi_fiplug", "constant"}:
        raise ValueError("Unknown quote_policy")

    rng = np.random.default_rng(seed)

    T = float(getattr(mm_aux, "T", 1.0))
    dt = T / Nt
    k = float(getattr(mm_aux, "k", 1.0))
    q_bar = int(getattr(mm_aux, "q_bar", 50))
    Q_min, Q_max = -q_bar, q_bar
    sigma = float(getattr(mm_aux, "sigma", 1.0))
    eta = float(getattr(mm_aux, "eta", 10.0))
    qconst = float(getattr(mm_aux, "qconst", 0.6))
    pconst = math.sqrt(max(1.0 - qconst ** 2, 0.0))
    mu = float(getattr(mm_aux, "mu", 0.0))
    U0 = float(getattr(mm_aux, "U_0", 0.0))
    Q0 = float(getattr(mm_aux, "Q_0", 0.0))
    varphi = float(getattr(mm_aux, "varphi", 15.0))
    psi = float(getattr(mm_aux, "psi"))
    gamma = float(getattr(mm_aux, "gamma", 1.0))

    A, b0, b1 = coefficients_from_mm(mm_aux, Nt)
    P = kalman_riccati_array(T, Nt, eta, qconst)
    theta, L, pi0 = create_ctmc_discretization(J, eta, grid=grid)
    trans = expm(L * dt)

    U = np.full(sims, U0, dtype=float)
    Q = np.full(sims, Q0, dtype=float)
    Uhat_PI = np.full(sims, U0, dtype=float)
    pi = np.tile(pi0, (sims, 1))

    mse_pi_path = np.zeros(sims, dtype=float)
    mse_arr_path = np.zeros(sims, dtype=float)

    for it in range(Nt):
        U_prev = U.copy()
        Uhat_ARR = pi @ theta

        if quote_policy == "zero_fiplug":
            U_for_quote = np.zeros(sims)
            rho_a, rho_b = quote_from_fi_coeff(Q, U_for_quote, A, b0, b1, k, it)
        elif quote_policy == "arr_fiplug":
            rho_a, rho_b = quote_from_fi_coeff(Q, Uhat_ARR, A, b0, b1, k, it)
        elif quote_policy == "pi_fiplug":
            rho_a, rho_b = quote_from_fi_coeff(Q, Uhat_PI, A, b0, b1, k, it)
        else:
            rho_a = np.full(sims, 1.0 / k)
            rho_b = np.full(sims, 1.0 / k)

        allow_a = (Q > Q_min).astype(float)
        allow_b = (Q < Q_max).astype(float)
        fad_true = gamma * sigma * qconst * U
        lam_a = (varphi * np.exp(-k * rho_a) + psi * np.exp(-k * rho_a - fad_true)) * allow_a
        lam_b = (varphi * np.exp(-k * rho_b) + psi * np.exp(-k * rho_b + fad_true)) * allow_b
        dNa = rng.poisson(np.maximum(lam_a * dt, 0.0)).astype(float)
        dNb = rng.poisson(np.maximum(lam_b * dt, 0.0)).astype(float)
        dNa = np.minimum(dNa, np.maximum(Q - Q_min, 0.0))
        dNb = np.minimum(dNb, np.maximum(Q_max - Q, 0.0))
        Q += dNb - dNa

        dB = rng.normal(0.0, math.sqrt(dt), sims)
        dZ = rng.normal(0.0, math.sqrt(dt), sims)
        U = U_prev - eta * U_prev * dt + dB
        dS = mu * dt + sigma * (qconst * (U - U_prev) + pconst * dZ)
        dY = (dS - mu * dt) / sigma

        Uhat_PI = kalman_update(Uhat_PI, dY, P[it], dt, eta, qconst)

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

        Uhat_ARR_post = pi @ theta
        mse_pi_path += (Uhat_PI - U) ** 2
        mse_arr_path += (Uhat_ARR_post - U) ** 2

    mse_pi_path /= Nt
    mse_arr_path /= Nt
    diff_path = mse_arr_path - mse_pi_path
    out = {
        "MSE_PI": float(np.mean(mse_pi_path)),
        "MSE_ARR": float(np.mean(mse_arr_path)),
        "MSE_ARR_minus_PI": float(np.mean(diff_path)),
        "sd_diff_path": float(np.sqrt(np.var(diff_path))),
        "psi": float(psi),
        "qconst": float(qconst),
        "gamma": float(gamma),
        "quote_policy": quote_policy,
        "grid": grid,
        "J": int(J),
    }
    if return_path_diffs:
        out["diff_path"] = diff_path
    return out
