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
    simulate_common_fiplug_strategy,
    two_sided_pvalue_from_z,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q_points", type=int, default=11)
    parser.add_argument("--gamma_points", type=int, default=11)
    parser.add_argument("--gamma_max", type=float, default=10.0)
    parser.add_argument("--sims", type=int, default=50000,
                        help="Use 50000 for diagnostic; increase to 100000 for final confirmation if needed.")
    parser.add_argument("--Nt", type=int, default=1000)
    parser.add_argument("--J", type=int, default=7)
    parser.add_argument("--grid", type=str, default="equidistant", choices=["equidistant", "equal_probability"])
    parser.add_argument("--q_bar", type=int, default=50)
    parser.add_argument("--fixed_psi", type=float, default=None)
    parser.add_argument("--seed", type=int, default=36457656)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--outdir", type=str, default="outputs_clean/47_high_gamma_fiplug_heatmap")
    parser.add_argument("--no_plots", action="store_true", help="Skip PNG generation; CSV/JSON are still saved.")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    q_values = np.linspace(0.0, 1.0, args.q_points)
    gamma_values = np.linspace(0.0, args.gamma_max, args.gamma_points)

    pi_mean = np.zeros((len(gamma_values), len(q_values)))
    arr_mean = np.zeros_like(pi_mean)
    pi_sd = np.zeros_like(pi_mean)
    arr_sd = np.zeros_like(pi_mean)
    diff = np.zeros_like(pi_mean)
    se = np.zeros_like(pi_mean)
    psi_mat = np.zeros_like(pi_mean)
    rows = []

    pbar = tqdm(total=len(q_values) * len(gamma_values), desc="high-gamma FI plug-in heatmap")
    for i, gamma in enumerate(gamma_values):
        for j, q in enumerate(q_values):
            mm = make_clean_params(q=float(q), gamma=float(gamma), q_bar=args.q_bar,
                                   fixed_psi=args.fixed_psi, Nt_for_recalib=args.Nt)
            seed_base = int(args.seed + 1009 * i + 10007 * j)
            pi = simulate_common_fiplug_strategy(mm, "pi", sims=args.sims, Nt=args.Nt,
                                                 J=args.J, seed=seed_base + 1, grid=args.grid)
            arr = simulate_common_fiplug_strategy(mm, "arr", sims=args.sims, Nt=args.Nt,
                                                  J=args.J, seed=seed_base + 2, grid=args.grid)

            pi_mean[i, j] = pi.mean
            arr_mean[i, j] = arr.mean
            pi_sd[i, j] = pi.sd
            arr_sd[i, j] = arr.sd
            diff[i, j] = arr.mean - pi.mean
            se[i, j] = np.sqrt((arr.sd ** 2 + pi.sd ** 2) / float(args.sims))
            psi_mat[i, j] = float(mm.psi)
            rows.append({
                "q": float(q),
                "gamma": float(gamma),
                "psi": float(mm.psi),
                "PI_FIplug_mean": pi.mean,
                "PI_FIplug_sd": pi.sd,
                "ARR_FIplug_mean": arr.mean,
                "ARR_FIplug_sd": arr.sd,
                "ARR_minus_PI_FIplug": diff[i, j],
                "se_independent": se[i, j],
                "sims": args.sims,
                "Nt": args.Nt,
                "J": args.J,
                "grid": args.grid,
                "seed_pi": seed_base + 1,
                "seed_arr": seed_base + 2,
            })
            pbar.set_postfix({"q": f"{q:.2f}", "gamma": f"{gamma:.2f}", "dJ": f"{diff[i,j]:.4g}"})
            pbar.update(1)
    pbar.close()

    z = np.zeros_like(diff)
    pvals = np.ones_like(diff)
    mask = se > 1e-14
    z[mask] = diff[mask] / se[mask]
    pvals[mask] = np.vectorize(two_sided_pvalue_from_z)(z[mask])
    pvals[(~mask) & (np.abs(diff) > 1e-14)] = 0.0
    reject = benjamini_hochberg_mask(pvals.ravel(), alpha=args.alpha).reshape(diff.shape)
    diff_sig = np.where(reject, diff, 0.0)

    save_csv_matrix(os.path.join(args.outdir, "pi_fiplug_mean.csv"), q_values, gamma_values, pi_mean)
    save_csv_matrix(os.path.join(args.outdir, "arr_fiplug_mean.csv"), q_values, gamma_values, arr_mean)
    save_csv_matrix(os.path.join(args.outdir, "arr_minus_pi_fiplug.csv"), q_values, gamma_values, diff)
    save_csv_matrix(os.path.join(args.outdir, "arr_minus_pi_fiplug_sig.csv"), q_values, gamma_values, diff_sig)
    save_csv_matrix(os.path.join(args.outdir, "p_values.csv"), q_values, gamma_values, pvals)
    save_csv_matrix(os.path.join(args.outdir, "bh_reject.csv"), q_values, gamma_values, reject.astype(int))
    save_csv_matrix(os.path.join(args.outdir, "psi_values.csv"), q_values, gamma_values, psi_mat)

    label = "recalibrated psi" if args.fixed_psi is None else f"fixed psi={args.fixed_psi:g}"
    suffix = f"{label}, J={args.J}, grid={args.grid}, gamma<= {args.gamma_max:g}"
    if not args.no_plots:
        plot_heatmap(q_values, gamma_values, diff, os.path.join(args.outdir, "arr_minus_pi_fiplug_heatmap.png"),
                     rf"High-$\gamma$ FI plug-in: $\bar J^{{ARR}}-\bar J^{{PI}}$ ({suffix})", r"$\bar J^{ARR}-\bar J^{PI}$")
        plot_heatmap(q_values, gamma_values, diff_sig, os.path.join(args.outdir, "arr_minus_pi_fiplug_sig_heatmap.png"),
                     rf"Significance-filtered high-$\gamma$ FI plug-in ({suffix})", r"displayed $\bar J^{ARR}-\bar J^{PI}$")
        plot_heatmap(q_values, gamma_values, psi_mat, os.path.join(args.outdir, "psi_heatmap.png"),
                     rf"$\psi$ values ({label})", r"$\psi$", cmap="viridis", symmetric=False)

    summary = vars(args)
    summary.update({
        "purpose": "Diagnostic high-gamma trading heatmap. Positive ARR_minus_PI_FIplug means ARR filter gives higher performance under the common FI quote map.",
        "q_values": q_values.tolist(),
        "gamma_values": gamma_values.tolist(),
        "PI_FIplug_mean": pi_mean.tolist(),
        "ARR_FIplug_mean": arr_mean.tolist(),
        "ARR_minus_PI_FIplug": diff.tolist(),
        "ARR_minus_PI_FIplug_sig": diff_sig.tolist(),
        "p_values": pvals.tolist(),
        "bh_reject": reject.astype(int).tolist(),
        "psi": psi_mat.tolist(),
        "point_results": rows,
    })
    with open(os.path.join(args.outdir, "high_gamma_fiplug_heatmap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved outputs to", args.outdir)


if __name__ == "__main__":
    main()
