import argparse
import json
import math
import os
from typing import Dict, Iterable, List

import numpy as np
from scipy.linalg import expm
from tqdm import tqdm

from clean_fiplug_core import (
    ensure_dir,
    make_clean_params,
    save_csv_rows,
    plot_lines,
    mean_sd,
    coefficients_from_mm,
    kalman_riccati_array,
    create_ctmc_discretization,
    quote_from_fi_coeff,
    kalman_update,
)


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def simulate_full_arrival_fiplug_strategy(
    mm_aux,
    strategy: str,
    sims: int,
    Nt: int,
    J: int,
    seed: int,
    grid: str = "equidistant",
) -> Dict[str, object]:
    if strategy not in {"pi", "arr_fill", "arr_full"}:
        raise ValueError("strategy must be pi, arr_fill, or arr_full")

    rng = np.random.default_rng(seed)

    T = float(getattr(mm_aux, "T", 1.0))
    dt = T / Nt
    k = float(getattr(mm_aux, "k", 1.0))
    phi_inventory = float(getattr(mm_aux, "phi", 0.1))
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
    pi_fill = np.tile(pi0, (sims, 1))
    pi_full = np.tile(pi0, (sims, 1))

    for it in range(Nt):
        S_prev = S.copy()
        U_prev = U.copy()

        Uhat_fill = pi_fill @ theta
        Uhat_full = pi_full @ theta

        if strategy == "pi":
            U_est = Uhat_PI
        elif strategy == "arr_fill":
            U_est = Uhat_fill
        else:
            U_est = Uhat_full

        rho_a, rho_b = quote_from_fi_coeff(Q, U_est, A, b0, b1, k, it)

        # Full market-order arrivals M. These do not depend on quotes.
        fad_true = gamma * sigma * qconst * U
        lam_M_a = varphi + psi * np.exp(-fad_true)
        lam_M_b = varphi + psi * np.exp(fad_true)

        dMa = rng.poisson(np.maximum(lam_M_a * dt, 0.0)).astype(float)
        dMb = rng.poisson(np.maximum(lam_M_b * dt, 0.0)).astype(float)

        # Quote-based thinning M -> N.
        allow_a = (Q > Q_min).astype(float)
        allow_b = (Q < Q_max).astype(float)

        p_fill_a = np.clip(np.exp(-k * rho_a) * allow_a, 0.0, 1.0)
        p_fill_b = np.clip(np.exp(-k * rho_b) * allow_b, 0.0, 1.0)

        dNa = rng.binomial(dMa.astype(int), p_fill_a).astype(float)
        dNb = rng.binomial(dMb.astype(int), p_fill_b).astype(float)

        dNa = np.minimum(dNa, np.maximum(Q - Q_min, 0.0))
        dNb = np.minimum(dNb, np.maximum(Q_max - Q, 0.0))

        X += dNa * (S_prev + rho_a) - dNb * (S_prev - rho_b)
        Q += dNb - dNa
        int_Q2 += Q ** 2 * dt

        # Hidden fad and price process.
        dB = rng.normal(0.0, math.sqrt(dt), sims)
        dZ = rng.normal(0.0, math.sqrt(dt), sims)

        U = U_prev - eta * U_prev * dt + dB
        dS = mu * dt + sigma * (qconst * (U - U_prev) + pconst * dZ)
        S = S_prev + dS
        dY = (dS - mu * dt) / sigma

        Uhat_PI = kalman_update(Uhat_PI, dY, P[it], dt, eta, qconst)

        fad_j = gamma * sigma * qconst * theta[None, :]

        # ARR-full filter update from M.
        pi_pred_full = pi_full @ trans
        lam_M_a_j = np.maximum(varphi + psi * np.exp(-fad_j), 1e-12)
        lam_M_b_j = np.maximum(varphi + psi * np.exp(fad_j), 1e-12)

        log_like_full = (
            dMa[:, None] * np.log(lam_M_a_j * dt) - lam_M_a_j * dt
            + dMb[:, None] * np.log(lam_M_b_j * dt) - lam_M_b_j * dt
        )
        log_like_full -= np.max(log_like_full, axis=1, keepdims=True)

        weights_full = pi_pred_full * np.exp(log_like_full)
        denom_full = weights_full.sum(axis=1, keepdims=True)
        pi_full = weights_full / np.maximum(denom_full, 1e-14)

        bad_full = denom_full[:, 0] <= 1e-14
        if np.any(bad_full):
            pi_full[bad_full, :] = pi0

        # ARR-fill filter update from N.
        pi_pred_fill = pi_fill @ trans

        rho_a_col = rho_a[:, None]
        rho_b_col = rho_b[:, None]
        allow_a_col = allow_a[:, None]
        allow_b_col = allow_b[:, None]

        lam_N_a_j = (
            varphi * np.exp(-k * rho_a_col)
            + psi * np.exp(-k * rho_a_col - fad_j)
        ) * allow_a_col

        lam_N_b_j = (
            varphi * np.exp(-k * rho_b_col)
            + psi * np.exp(-k * rho_b_col + fad_j)
        ) * allow_b_col

        lam_N_a_j = np.maximum(lam_N_a_j, 1e-12)
        lam_N_b_j = np.maximum(lam_N_b_j, 1e-12)

        log_like_fill = (
            dNa[:, None] * np.log(lam_N_a_j * dt) - lam_N_a_j * dt
            + dNb[:, None] * np.log(lam_N_b_j * dt) - lam_N_b_j * dt
        )
        log_like_fill -= np.max(log_like_fill, axis=1, keepdims=True)

        weights_fill = pi_pred_fill * np.exp(log_like_fill)
        denom_fill = weights_fill.sum(axis=1, keepdims=True)
        pi_fill = weights_fill / np.maximum(denom_fill, 1e-14)

        bad_fill = denom_fill[:, 0] <= 1e-14
        if np.any(bad_fill):
            pi_fill[bad_fill, :] = pi0

    perf = X + Q * S - alpha * Q ** 2 - phi_inventory * int_Q2
    mean, sd = mean_sd(perf)
    return {
        "strategy": strategy,
        "mean": float(mean),
        "sd": float(sd),
        "se": float(sd / math.sqrt(float(sims))),
        "psi": float(psi),
        "qconst": float(qconst),
        "gamma": float(gamma),
        "grid": grid,
        "J": int(J),
    }


def run_sweep(
    name: str,
    values: Iterable[float],
    make_params,
    sims: int,
    Nt: int,
    J: int,
    seed: int,
    grid: str,
    outdir: str,
    no_plots: bool,
):
    rows = []
    values = list(values)
    strategy_labels = {
        "pi": "PI-FIplug",
        "arr_fill": "ARR-fill-FIplug",
        "arr_full": "ARR-full-FIplug",
    }

    pbar = tqdm(total=len(values) * 3, desc=f"{name} full-arrival FIplug sweep")
    for idx, value in enumerate(values):
        mm = make_params(value)

        for sidx, strategy in enumerate(["pi", "arr_fill", "arr_full"]):
            seed_point = int(seed + 10007 * idx + 1009 * sidx)
            res = simulate_full_arrival_fiplug_strategy(
                mm, strategy=strategy, sims=sims, Nt=Nt, J=J, seed=seed_point, grid=grid
            )
            rows.append({
                "sweep": name,
                "x": float(value),
                "strategy": strategy_labels[strategy],
                "mean": res["mean"],
                "sd": res["sd"],
                "se": res["se"],
                "psi": res["psi"],
                "qconst": res["qconst"],
                "gamma": res["gamma"],
                "sims": int(sims),
                "Nt": int(Nt),
                "seed": int(seed_point),
                "grid": grid,
                "J": int(J),
            })
            pbar.set_postfix({"x": f"{value:.3g}", "strategy": strategy})
            pbar.update(1)
    pbar.close()

    ensure_dir(outdir)
    csv_path = os.path.join(outdir, f"{name}_full_arrival_fiplug_sweep.csv")
    save_csv_rows(csv_path, rows)

    if not no_plots:
        series = {}
        for label in strategy_labels.values():
            series[label] = [r["mean"] for r in rows if r["strategy"] == label]
        plot_lines(np.asarray(values), series, name, "Mean objective",
                   f"Full-arrival FIplug comparison: {name}-sweep",
                   os.path.join(outdir, f"{name}_full_arrival_fiplug_sweep.png"))

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="both", choices=["q", "gamma", "both"])
    parser.add_argument("--q_values", type=str, default="0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--gamma_values", type=str, default="0,1,2,3")
    parser.add_argument("--q_for_gamma", type=float, default=0.6)
    parser.add_argument("--gamma_for_q", type=float, default=1.0)

    parser.add_argument("--sims", type=int, default=100000)
    parser.add_argument("--Nt", type=int, default=1000)
    parser.add_argument("--J", type=int, default=7)
    parser.add_argument("--grid", type=str, default="equidistant", choices=["equidistant", "equal_probability"])
    parser.add_argument("--q_bar", type=int, default=50)
    parser.add_argument("--fixed_psi", type=float, default=None)
    parser.add_argument("--seed", type=int, default=91357)
    parser.add_argument("--outdir", type=str, default="outputs_full_arrival/45c_fiplug_sweeps")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    all_rows = []

    if args.mode in {"q", "both"}:
        q_values = parse_float_list(args.q_values)

        def make_q_params(q):
            return make_clean_params(
                q=float(q),
                gamma=float(args.gamma_for_q),
                q_bar=args.q_bar,
                fixed_psi=args.fixed_psi,
                Nt_for_recalib=args.Nt,
            )

        all_rows.extend(run_sweep(
            name="q",
            values=q_values,
            make_params=make_q_params,
            sims=args.sims,
            Nt=args.Nt,
            J=args.J,
            seed=args.seed,
            grid=args.grid,
            outdir=args.outdir,
            no_plots=args.no_plots,
        ))

    if args.mode in {"gamma", "both"}:
        gamma_values = parse_float_list(args.gamma_values)

        def make_gamma_params(gamma):
            return make_clean_params(
                q=float(args.q_for_gamma),
                gamma=float(gamma),
                q_bar=args.q_bar,
                fixed_psi=args.fixed_psi,
                Nt_for_recalib=args.Nt,
            )

        all_rows.extend(run_sweep(
            name="gamma",
            values=gamma_values,
            make_params=make_gamma_params,
            sims=args.sims,
            Nt=args.Nt,
            J=args.J,
            seed=args.seed + 999999,
            grid=args.grid,
            outdir=args.outdir,
            no_plots=args.no_plots,
        ))

    save_csv_rows(os.path.join(args.outdir, "full_arrival_fiplug_all_sweeps.csv"), all_rows)
    with open(os.path.join(args.outdir, "full_arrival_fiplug_sweeps_summary.json"), "w") as f:
        json.dump({"args": vars(args), "rows": all_rows}, f, indent=2)

    print("Saved outputs to", args.outdir)


if __name__ == "__main__":
    main()
