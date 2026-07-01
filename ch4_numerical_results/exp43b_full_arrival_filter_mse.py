import argparse
import json
import math
import os
from typing import Dict

import numpy as np
from scipy.linalg import expm
from tqdm import tqdm

from clean_fiplug_core import (
    ensure_dir,
    make_clean_params,
    save_csv_matrix,
    plot_heatmap,
    coefficients_from_mm,
    kalman_riccati_array,
    create_ctmc_discretization,
    quote_from_fi_coeff,
    kalman_update,
)


def simulate_filter_mse_full_arrival(
    mm_aux,
    sims: int,
    Nt: int,
    J: int,
    seed: int,
    grid: str = "equidistant",
    quote_policy: str = "zero_fiplug",
) -> Dict[str, object]:
    if quote_policy not in {"zero_fiplug", "arr_fill_fiplug", "arr_full_fiplug", "pi_fiplug", "constant"}:
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

    pi_fill = np.tile(pi0, (sims, 1))
    pi_full = np.tile(pi0, (sims, 1))

    mse_pi_path = np.zeros(sims, dtype=float)
    mse_fill_path = np.zeros(sims, dtype=float)
    mse_full_path = np.zeros(sims, dtype=float)

    for it in range(Nt):
        U_prev = U.copy()

        Uhat_fill = pi_fill @ theta
        Uhat_full = pi_full @ theta

        if quote_policy == "zero_fiplug":
            U_for_quote = np.zeros(sims)
            rho_a, rho_b = quote_from_fi_coeff(Q, U_for_quote, A, b0, b1, k, it)
        elif quote_policy == "arr_fill_fiplug":
            rho_a, rho_b = quote_from_fi_coeff(Q, Uhat_fill, A, b0, b1, k, it)
        elif quote_policy == "arr_full_fiplug":
            rho_a, rho_b = quote_from_fi_coeff(Q, Uhat_full, A, b0, b1, k, it)
        elif quote_policy == "pi_fiplug":
            rho_a, rho_b = quote_from_fi_coeff(Q, Uhat_PI, A, b0, b1, k, it)
        else:
            rho_a = np.full(sims, 1.0 / k)
            rho_b = np.full(sims, 1.0 / k)

        # Full market-order arrivals M. These do not depend on quotes.
        fad_true = gamma * sigma * qconst * U
        lam_M_a = varphi + psi * np.exp(-fad_true)
        lam_M_b = varphi + psi * np.exp(fad_true)

        dMa = rng.poisson(np.maximum(lam_M_a * dt, 0.0)).astype(float)
        dMb = rng.poisson(np.maximum(lam_M_b * dt, 0.0)).astype(float)

        # Quote-based thinning from M to executed fills N.
        allow_a = (Q > Q_min).astype(float)
        allow_b = (Q < Q_max).astype(float)

        p_fill_a = np.clip(np.exp(-k * rho_a) * allow_a, 0.0, 1.0)
        p_fill_b = np.clip(np.exp(-k * rho_b) * allow_b, 0.0, 1.0)

        dNa = rng.binomial(dMa.astype(int), p_fill_a).astype(float)
        dNb = rng.binomial(dMb.astype(int), p_fill_b).astype(float)

        dNa = np.minimum(dNa, np.maximum(Q - Q_min, 0.0))
        dNb = np.minimum(dNb, np.maximum(Q_max - Q, 0.0))
        Q += dNb - dNa

        # Hidden fad and price observation.
        dB = rng.normal(0.0, math.sqrt(dt), sims)
        dZ = rng.normal(0.0, math.sqrt(dt), sims)

        U = U_prev - eta * U_prev * dt + dB
        dS = mu * dt + sigma * (qconst * (U - U_prev) + pconst * dZ)
        dY = (dS - mu * dt) / sigma

        # Price filter.
        Uhat_PI = kalman_update(Uhat_PI, dY, P[it], dt, eta, qconst)

        fad_j = gamma * sigma * qconst * theta[None, :]

        # ARR-full update from full arrivals M.
        pi_pred_full = pi_full @ trans

        lam_M_a_j = varphi + psi * np.exp(-fad_j)
        lam_M_b_j = varphi + psi * np.exp(fad_j)

        lam_M_a_j = np.maximum(lam_M_a_j, 1e-12)
        lam_M_b_j = np.maximum(lam_M_b_j, 1e-12)

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

        # ARR-fill update from executed fills N.
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

        Uhat_fill_post = pi_fill @ theta
        Uhat_full_post = pi_full @ theta

        mse_pi_path += (Uhat_PI - U) ** 2
        mse_fill_path += (Uhat_fill_post - U) ** 2
        mse_full_path += (Uhat_full_post - U) ** 2

    mse_pi_path /= Nt
    mse_fill_path /= Nt
    mse_full_path /= Nt

    fill_minus_pi = mse_fill_path - mse_pi_path
    full_minus_pi = mse_full_path - mse_pi_path
    full_minus_fill = mse_full_path - mse_fill_path

    return {
        "MSE_PI": float(np.mean(mse_pi_path)),
        "MSE_ARR_fill": float(np.mean(mse_fill_path)),
        "MSE_ARR_full": float(np.mean(mse_full_path)),
        "MSE_ARR_fill_minus_PI": float(np.mean(fill_minus_pi)),
        "MSE_ARR_full_minus_PI": float(np.mean(full_minus_pi)),
        "MSE_ARR_full_minus_fill": float(np.mean(full_minus_fill)),
        "sd_fill_minus_PI_path": float(np.sqrt(np.var(fill_minus_pi))),
        "sd_full_minus_PI_path": float(np.sqrt(np.var(full_minus_pi))),
        "sd_full_minus_fill_path": float(np.sqrt(np.var(full_minus_fill))),
        "psi": float(psi),
        "qconst": float(qconst),
        "gamma": float(gamma),
        "quote_policy": quote_policy,
        "grid": grid,
        "J": int(J),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--q_points", type=int, default=11)
    parser.add_argument("--gamma_points", type=int, default=11)
    parser.add_argument("--gamma_max", type=float, default=10.0)

    parser.add_argument("--sims", type=int, default=10000)
    parser.add_argument("--Nt", type=int, default=1000)
    parser.add_argument("--J", type=int, default=7)

    parser.add_argument("--grid", type=str, default="equidistant",
                        choices=["equidistant", "equal_probability"])

    parser.add_argument("--quote_policy", type=str, default="zero_fiplug",
                        choices=["zero_fiplug", "arr_fill_fiplug", "arr_full_fiplug", "pi_fiplug", "constant"])

    parser.add_argument("--q_bar", type=int, default=50)
    parser.add_argument("--fixed_psi", type=float, default=None)
    parser.add_argument("--seed", type=int, default=36457656)
    parser.add_argument("--outdir", type=str, default="outputs_full_arrival/43b_filter_mse")
    parser.add_argument("--no_plots", action="store_true")

    args = parser.parse_args()

    ensure_dir(args.outdir)

    q_values = np.linspace(0.0, 1.0, args.q_points)
    gamma_values = np.linspace(0.0, args.gamma_max, args.gamma_points)

    mse_pi = np.zeros((len(gamma_values), len(q_values)))
    mse_fill = np.zeros_like(mse_pi)
    mse_full = np.zeros_like(mse_pi)

    diff_fill_pi = np.zeros_like(mse_pi)
    diff_full_pi = np.zeros_like(mse_pi)
    diff_full_fill = np.zeros_like(mse_pi)

    psi_mat = np.zeros_like(mse_pi)
    rows = []

    pbar = tqdm(total=len(q_values) * len(gamma_values), desc="full-arrival filter MSE grid")

    for i, gamma in enumerate(gamma_values):
        for j, q in enumerate(q_values):
            mm = make_clean_params(
                q=float(q),
                gamma=float(gamma),
                q_bar=args.q_bar,
                fixed_psi=args.fixed_psi,
                Nt_for_recalib=args.Nt,
            )

            seed_point = int(args.seed + 1009 * i + 10007 * j)

            res = simulate_filter_mse_full_arrival(
                mm,
                sims=args.sims,
                Nt=args.Nt,
                J=args.J,
                seed=seed_point,
                grid=args.grid,
                quote_policy=args.quote_policy,
            )

            mse_pi[i, j] = res["MSE_PI"]
            mse_fill[i, j] = res["MSE_ARR_fill"]
            mse_full[i, j] = res["MSE_ARR_full"]

            diff_fill_pi[i, j] = res["MSE_ARR_fill_minus_PI"]
            diff_full_pi[i, j] = res["MSE_ARR_full_minus_PI"]
            diff_full_fill[i, j] = res["MSE_ARR_full_minus_fill"]

            psi_mat[i, j] = res["psi"]

            rows.append({
                **res,
                "q": float(q),
                "gamma": float(gamma),
                "sims": args.sims,
                "Nt": args.Nt,
                "seed": seed_point,
            })

            pbar.set_postfix({
                "q": f"{q:.2f}",
                "gamma": f"{gamma:.2f}",
                "full-fill": f"{diff_full_fill[i, j]:.4g}",
            })
            pbar.update(1)

    pbar.close()

    save_csv_matrix(os.path.join(args.outdir, "mse_pi.csv"), q_values, gamma_values, mse_pi)
    save_csv_matrix(os.path.join(args.outdir, "mse_arr_fill.csv"), q_values, gamma_values, mse_fill)
    save_csv_matrix(os.path.join(args.outdir, "mse_arr_full.csv"), q_values, gamma_values, mse_full)

    save_csv_matrix(os.path.join(args.outdir, "mse_arr_fill_minus_pi.csv"), q_values, gamma_values, diff_fill_pi)
    save_csv_matrix(os.path.join(args.outdir, "mse_arr_full_minus_pi.csv"), q_values, gamma_values, diff_full_pi)
    save_csv_matrix(os.path.join(args.outdir, "mse_arr_full_minus_fill.csv"), q_values, gamma_values, diff_full_fill)

    save_csv_matrix(os.path.join(args.outdir, "psi_values.csv"), q_values, gamma_values, psi_mat)

    if not args.no_plots:
        label = "recalibrated psi" if args.fixed_psi is None else f"fixed psi={args.fixed_psi:g}"
        title_suffix = f"{label}, policy={args.quote_policy}, J={args.J}"

        plot_heatmap(q_values, gamma_values, diff_full_fill,
                     os.path.join(args.outdir, "mse_arr_full_minus_fill_heatmap.png"),
                     rf"Filter MSE: $MSE^{{ARR-full}}-MSE^{{ARR-fill}}$ ({title_suffix})",
                     r"$MSE^{ARR-full}-MSE^{ARR-fill}$")

        plot_heatmap(q_values, gamma_values, diff_full_pi,
                     os.path.join(args.outdir, "mse_arr_full_minus_pi_heatmap.png"),
                     rf"Filter MSE: $MSE^{{ARR-full}}-MSE^{{PI}}$ ({title_suffix})",
                     r"$MSE^{ARR-full}-MSE^{PI}$")

        plot_heatmap(q_values, gamma_values, diff_fill_pi,
                     os.path.join(args.outdir, "mse_arr_fill_minus_pi_heatmap.png"),
                     rf"Filter MSE: $MSE^{{ARR-fill}}-MSE^{{PI}}$ ({title_suffix})",
                     r"$MSE^{ARR-fill}-MSE^{PI}$")

    summary = vars(args)
    summary.update({
        "q_values": q_values.tolist(),
        "gamma_values": gamma_values.tolist(),
        "MSE_PI": mse_pi.tolist(),
        "MSE_ARR_fill": mse_fill.tolist(),
        "MSE_ARR_full": mse_full.tolist(),
        "MSE_ARR_fill_minus_PI": diff_fill_pi.tolist(),
        "MSE_ARR_full_minus_PI": diff_full_pi.tolist(),
        "MSE_ARR_full_minus_fill": diff_full_fill.tolist(),
        "psi": psi_mat.tolist(),
        "point_results": rows,
    })

    with open(os.path.join(args.outdir, "full_arrival_filter_mse_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved outputs to", args.outdir)


if __name__ == "__main__":
    main()
