import json
import math
import argparse
from statistics import NormalDist

import numpy as np
import matplotlib.pyplot as plt


def two_sided_pvalue_from_z(z):
    normal = NormalDist()
    return 2.0 * (1.0 - normal.cdf(abs(float(z))))


def benjamini_hochberg_mask(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    thresh = alpha * np.arange(1, n + 1) / n
    passed = sorted_p <= thresh

    reject = np.zeros(n, dtype=bool)
    if np.any(passed):
        k = np.max(np.where(passed)[0])
        reject[order[:k+1]] = True
    return reject


def plot_heatmap(q_values, gamma_values, mat, path, title, cbar_label):
    plt.figure(figsize=(8.0, 5.8))

    vmax = float(np.nanmax(np.abs(mat)))
    if vmax <= 1e-14:
        vmax = 1.0
    vmin = -vmax

    extent = [min(q_values), max(q_values), min(gamma_values), max(gamma_values)]
    im = plt.imshow(
        mat,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    plt.xlabel(r"$q$")
    plt.ylabel(r"$\gamma$")
    plt.title(title)

    cbar = plt.colorbar(im)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_matrix_csv(path, q_values, gamma_values, mat):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gamma\\q"] + list(q_values))
        for i, gamma in enumerate(gamma_values):
            writer.writerow([gamma] + list(mat[i, :]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=str,
        default="outputs_full_arrival/43b_filter_mse_gamma10_J7/full_arrival_filter_mse_summary.json",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs_full_arrival/43b_filter_mse_gamma10_J7_sig",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    import os
    os.makedirs(args.outdir, exist_ok=True)

    with open(args.input_json) as f:
        d = json.load(f)

    rows = d["point_results"]

    q_values = sorted({float(r["q"]) for r in rows})
    gamma_values = sorted({float(r["gamma"]) for r in rows})

    q_to_j = {q: j for j, q in enumerate(q_values)}
    g_to_i = {g: i for i, g in enumerate(gamma_values)}

    delta = np.zeros((len(gamma_values), len(q_values)))
    se = np.zeros_like(delta)
    pvals = np.ones_like(delta)

    sims = None

    for r in rows:
        q = float(r["q"])
        g = float(r["gamma"])
        i = g_to_i[g]
        j = q_to_j[q]

        diff = float(r["MSE_ARR_full_minus_PI"])
        sd_path = float(r["sd_full_minus_PI_path"])
        sims = int(r["sims"])

        delta[i, j] = diff
        se_ij = sd_path / math.sqrt(sims)
        se[i, j] = se_ij

        if se_ij > 1e-14:
            z = diff / se_ij
            pvals[i, j] = two_sided_pvalue_from_z(z)
        else:
            pvals[i, j] = 0.0 if abs(diff) > 1e-14 else 1.0

    reject = benjamini_hochberg_mask(pvals.ravel(), alpha=args.alpha).reshape(delta.shape)
    delta_sig = np.where(reject, delta, 0.0)

    save_matrix_csv(f"{args.outdir}/delta_arrfull_minus_pi_raw.csv", q_values, gamma_values, delta)
    save_matrix_csv(f"{args.outdir}/delta_arrfull_minus_pi_sig.csv", q_values, gamma_values, delta_sig)
    save_matrix_csv(f"{args.outdir}/p_values.csv", q_values, gamma_values, pvals)
    save_matrix_csv(f"{args.outdir}/bh_reject.csv", q_values, gamma_values, reject.astype(int))

    plot_heatmap(
        q_values,
        gamma_values,
        delta,
        f"{args.outdir}/mse_arr_full_minus_pi_heatmap_raw.png",
        r"Filter MSE: $MSE^{ARR-full}-MSE^{PI}$",
        r"$MSE^{ARR-full}-MSE^{PI}$",
    )

    plot_heatmap(
        q_values,
        gamma_values,
        delta_sig,
        f"{args.outdir}/mse_arr_full_minus_pi_heatmap_sig.png",
        r"Filter MSE: $MSE^{ARR-full}-MSE^{PI}$ (significance-filtered)",
        r"displayed $MSE^{ARR-full}-MSE^{PI}$",
    )

    total = delta.size
    sig_cells = int(reject.sum())
    blue_sig = int(np.sum((delta_sig < 0)))
    red_sig = int(np.sum((delta_sig > 0)))

    summary = {
        "input_json": args.input_json,
        "alpha": args.alpha,
        "sims": sims,
        "total_cells": total,
        "significant_cells_after_BH": sig_cells,
        "significant_blue_cells": blue_sig,
        "significant_red_cells": red_sig,
    }

    with open(f"{args.outdir}/significance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved to:", args.outdir)
    print(summary)


if __name__ == "__main__":
    main()
