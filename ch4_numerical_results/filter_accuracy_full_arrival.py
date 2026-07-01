import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from clean_fiplug_core import (
    benjamini_hochberg_mask,
    ensure_dir,
    make_clean_params,
    plot_heatmap,
    save_csv_matrix,
    simulate_filter_mse,
    two_sided_pvalue_from_z,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q_points", type=int, default=11)
    parser.add_argument("--gamma_points", type=int, default=11)
    parser.add_argument("--gamma_max", type=float, default=10.0)
    parser.add_argument("--sims", type=int, default=10000)
    parser.add_argument("--Nt", type=int, default=1000)
    parser.add_argument("--J", type=int, default=7)
    parser.add_argument("--grid", type=str, default="equidistant", choices=["equidistant", "equal_probability"])
    parser.add_argument("--quote_policy", type=str, default="zero_fiplug", choices=["zero_fiplug", "arr_fiplug", "pi_fiplug", "constant"])
    parser.add_argument("--q_bar", type=int, default=50)
    parser.add_argument("--fixed_psi", type=float, default=None)
    parser.add_argument("--seed", type=int, default=36457656)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--outdir", type=str, default="outputs_clean/43_filter_mse")
    parser.add_argument("--no_plots", action="store_true", help="Skip PNG generation; CSV/JSON are still saved.")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    q_values = np.linspace(0.0, 1.0, args.q_points)
    gamma_values = np.linspace(0.0, args.gamma_max, args.gamma_points)

    mse_pi = np.zeros((len(gamma_values), len(q_values)))
    mse_arr = np.zeros_like(mse_pi)
    diff = np.zeros_like(mse_pi)
    sd_diff = np.zeros_like(mse_pi)
    psi_mat = np.zeros_like(mse_pi)
    rows = []

    pbar = tqdm(total=len(q_values) * len(gamma_values), desc="filter MSE grid")
    for i, gamma in enumerate(gamma_values):
        for j, q in enumerate(q_values):
            mm = make_clean_params(q=float(q), gamma=float(gamma), q_bar=args.q_bar, fixed_psi=args.fixed_psi, Nt_for_recalib=args.Nt)
            seed_point = int(args.seed + 1009 * i + 10007 * j)
            res = simulate_filter_mse(
                mm, sims=args.sims, Nt=args.Nt, J=args.J, seed=seed_point,
                grid=args.grid, quote_policy=args.quote_policy,
            )
            mse_pi[i, j] = res["MSE_PI"]
            mse_arr[i, j] = res["MSE_ARR"]
            diff[i, j] = res["MSE_ARR_minus_PI"]
            sd_diff[i, j] = res["sd_diff_path"]
            psi_mat[i, j] = res["psi"]
            rows.append({**res, "q": float(q), "gamma": float(gamma), "sims": args.sims, "Nt": args.Nt, "seed": seed_point})
            pbar.set_postfix({"q": f"{q:.2f}", "gamma": f"{gamma:.2f}", "dMSE": f"{diff[i,j]:.4g}"})
            pbar.update(1)
    pbar.close()

    se = sd_diff / np.sqrt(float(args.sims))
    z = np.zeros_like(diff)
    pvals = np.ones_like(diff)
    mask = se > 1e-14
    z[mask] = diff[mask] / se[mask]
    pvals[mask] = np.vectorize(two_sided_pvalue_from_z)(z[mask])
    pvals[(~mask) & (np.abs(diff) > 1e-14)] = 0.0
    reject = benjamini_hochberg_mask(pvals.ravel(), alpha=args.alpha).reshape(diff.shape)
    diff_sig = np.where(reject, diff, 0.0)

    save_csv_matrix(os.path.join(args.outdir, "mse_pi.csv"), q_values, gamma_values, mse_pi)
    save_csv_matrix(os.path.join(args.outdir, "mse_arr.csv"), q_values, gamma_values, mse_arr)
    save_csv_matrix(os.path.join(args.outdir, "mse_arr_minus_pi.csv"), q_values, gamma_values, diff)
    save_csv_matrix(os.path.join(args.outdir, "mse_arr_minus_pi_sig.csv"), q_values, gamma_values, diff_sig)
    save_csv_matrix(os.path.join(args.outdir, "psi_values.csv"), q_values, gamma_values, psi_mat)
    save_csv_matrix(os.path.join(args.outdir, "p_values.csv"), q_values, gamma_values, pvals)
    save_csv_matrix(os.path.join(args.outdir, "bh_reject.csv"), q_values, gamma_values, reject.astype(int))

    label = "recalibrated psi" if args.fixed_psi is None else f"fixed psi={args.fixed_psi:g}"
    title_suffix = f"{label}, policy={args.quote_policy}, J={args.J}"
    if not args.no_plots:
        plot_heatmap(q_values, gamma_values, diff, os.path.join(args.outdir, "mse_arr_minus_pi_heatmap.png"),
                     rf"Filter MSE: $MSE^{{ARR}}-MSE^{{PI}}$ ({title_suffix})", r"$MSE^{ARR}-MSE^{PI}$")
        plot_heatmap(q_values, gamma_values, diff_sig, os.path.join(args.outdir, "mse_arr_minus_pi_sig_heatmap.png"),
                     rf"Significance-filtered filter MSE ({title_suffix})", r"displayed $MSE^{ARR}-MSE^{PI}$")
        plot_heatmap(q_values, gamma_values, psi_mat, os.path.join(args.outdir, "psi_heatmap.png"),
                     rf"$\psi$ values ({label})", r"$\psi$", cmap="viridis", symmetric=False)

    summary = vars(args)
    summary.update({
        "q_values": q_values.tolist(),
        "gamma_values": gamma_values.tolist(),
        "MSE_PI": mse_pi.tolist(),
        "MSE_ARR": mse_arr.tolist(),
        "MSE_ARR_minus_PI": diff.tolist(),
        "MSE_ARR_minus_PI_sig": diff_sig.tolist(),
        "p_values": pvals.tolist(),
        "bh_reject": reject.astype(int).tolist(),
        "psi": psi_mat.tolist(),
        "point_results": rows,
    })
    with open(os.path.join(args.outdir, "filter_mse_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved outputs to", args.outdir)


if __name__ == "__main__":
    main()
