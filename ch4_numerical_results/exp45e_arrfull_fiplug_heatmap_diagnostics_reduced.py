import argparse
import csv
import json
import math
import os
import inspect
from statistics import NormalDist

import numpy as np
import matplotlib.pyplot as plt

from clean_fiplug_core import ensure_dir, make_clean_params
from exp45c_full_arrival_fiplug_sweeps import simulate_full_arrival_fiplug_strategy


def parse_float_list(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def two_sided_pvalue_from_z(z):
    normal = NormalDist()
    return 2.0 * (1.0 - normal.cdf(abs(float(z))))


def benjamini_hochberg_mask(pvals, alpha):
    pvals = np.asarray(pvals, dtype=float)
    n = pvals.size
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    threshold = alpha * np.arange(1, n + 1) / n
    passed = sorted_p <= threshold

    reject = np.zeros(n, dtype=bool)
    if np.any(passed):
        k = np.max(np.where(passed)[0])
        reject[order[: k + 1]] = True
    return reject


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_matrix_csv(path, q_values, gamma_values, mat):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gamma\\q"] + list(q_values))
        for i, gamma in enumerate(gamma_values):
            writer.writerow([gamma] + list(mat[i, :]))


def plot_heatmap(q_values, gamma_values, mat, path, title, cbar_label):
    plt.figure(figsize=(7.2, 5.2))

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


def make_params_with_overrides(q, gamma, eta, sigma, q_bar, fixed_psi, Nt):
    """
    Build clean model parameters and override eta/sigma.

    This wrapper is deliberately defensive:
    - if make_clean_params supports eta/sigma keywords, pass them directly;
    - otherwise create the baseline object and overwrite attributes.

    Important: if sigma affects the internal psi recalibration in your local
    clean_fiplug_core.py, the best case is that make_clean_params accepts sigma.
    The script prints the resulting psi values to csv, so we can inspect them.
    """
    sig = inspect.signature(make_clean_params)
    kwargs = {
        "q": float(q),
        "gamma": float(gamma),
        "q_bar": q_bar,
        "fixed_psi": fixed_psi,
        "Nt_for_recalib": Nt,
    }

    if "eta" in sig.parameters:
        kwargs["eta"] = float(eta)
    if "sigma" in sig.parameters:
        kwargs["sigma"] = float(sigma)

    mm = make_clean_params(**kwargs)

    if "eta" not in sig.parameters:
        if hasattr(mm, "eta"):
            mm.eta = float(eta)
        else:
            raise AttributeError("Parameter object has no eta attribute.")

    if "sigma" not in sig.parameters:
        if hasattr(mm, "sigma"):
            mm.sigma = float(sigma)
        else:
            raise AttributeError("Parameter object has no sigma attribute.")

    return mm


def run_one_diagnostic(
    label,
    eta,
    sigma,
    q_values,
    gamma_values,
    sims,
    Nt,
    J,
    grid,
    q_bar,
    fixed_psi,
    seed,
    alpha,
    outdir,
):
    diag_dir = os.path.join(outdir, label)
    ensure_dir(diag_dir)

    n_g = len(gamma_values)
    n_q = len(q_values)

    mean_pi = np.zeros((n_g, n_q))
    mean_arr = np.zeros((n_g, n_q))
    delta = np.zeros((n_g, n_q))

    sd_pi = np.zeros((n_g, n_q))
    sd_arr = np.zeros((n_g, n_q))
    se_delta = np.zeros((n_g, n_q))
    pvals = np.ones((n_g, n_q))
    psi_mat = np.zeros((n_g, n_q))

    rows = []

    total = n_g * n_q
    counter = 0

    for i, gamma in enumerate(gamma_values):
        for j, q in enumerate(q_values):
            counter += 1
            print(
                f"[{label}] [{counter}/{total}] q={q:g}, gamma={gamma:g}, "
                f"eta={eta:g}, sigma={sigma:g}",
                flush=True,
            )

            mm = make_params_with_overrides(
                q=q,
                gamma=gamma,
                eta=eta,
                sigma=sigma,
                q_bar=q_bar,
                fixed_psi=fixed_psi,
                Nt=Nt,
            )

            seed_pi = int(seed + 100000 * i + 1000 * j + 11)
            seed_arr = int(seed + 100000 * i + 1000 * j + 29)

            res_pi = simulate_full_arrival_fiplug_strategy(
                mm,
                strategy="pi",
                sims=sims,
                Nt=Nt,
                J=J,
                seed=seed_pi,
                grid=grid,
            )

            res_arr = simulate_full_arrival_fiplug_strategy(
                mm,
                strategy="arr_full",
                sims=sims,
                Nt=Nt,
                J=J,
                seed=seed_arr,
                grid=grid,
            )

            mean_pi[i, j] = float(res_pi["mean"])
            mean_arr[i, j] = float(res_arr["mean"])
            sd_pi[i, j] = float(res_pi["sd"])
            sd_arr[i, j] = float(res_arr["sd"])

            delta[i, j] = mean_arr[i, j] - mean_pi[i, j]

            se = math.sqrt(sd_pi[i, j] ** 2 / sims + sd_arr[i, j] ** 2 / sims)
            se_delta[i, j] = se

            if se > 1e-14:
                z = delta[i, j] / se
                pvals[i, j] = two_sided_pvalue_from_z(z)
            else:
                pvals[i, j] = 0.0 if abs(delta[i, j]) > 1e-14 else 1.0

            psi_mat[i, j] = float(res_pi.get("psi", getattr(mm, "psi", float("nan"))))

            rows.append({
                "diagnostic": label,
                "eta": eta,
                "sigma": sigma,
                "q": q,
                "gamma": gamma,
                "PI_mean": mean_pi[i, j],
                "ARR_full_mean": mean_arr[i, j],
                "Delta_J": delta[i, j],
                "PI_sd": sd_pi[i, j],
                "ARR_full_sd": sd_arr[i, j],
                "SE_Delta": se_delta[i, j],
                "p_value": pvals[i, j],
                "psi": psi_mat[i, j],
                "sims": sims,
                "Nt": Nt,
                "J": J,
                "grid": grid,
                "seed_PI": seed_pi,
                "seed_ARR_full": seed_arr,
            })

    reject = benjamini_hochberg_mask(pvals.ravel(), alpha).reshape(delta.shape)
    delta_sig = np.where(reject, delta, 0.0)

    save_matrix_csv(os.path.join(diag_dir, "pi_mean.csv"), q_values, gamma_values, mean_pi)
    save_matrix_csv(os.path.join(diag_dir, "arr_full_mean.csv"), q_values, gamma_values, mean_arr)
    save_matrix_csv(os.path.join(diag_dir, "delta_arrfull_minus_pi.csv"), q_values, gamma_values, delta)
    save_matrix_csv(os.path.join(diag_dir, "delta_arrfull_minus_pi_sig.csv"), q_values, gamma_values, delta_sig)
    save_matrix_csv(os.path.join(diag_dir, "p_values.csv"), q_values, gamma_values, pvals)
    save_matrix_csv(os.path.join(diag_dir, "bh_reject.csv"), q_values, gamma_values, reject.astype(int))
    save_matrix_csv(os.path.join(diag_dir, "psi_values.csv"), q_values, gamma_values, psi_mat)

    write_csv(
        os.path.join(diag_dir, "arrfull_fiplug_heatmap_rows.csv"),
        rows,
        [
            "diagnostic", "eta", "sigma", "q", "gamma",
            "PI_mean", "ARR_full_mean", "Delta_J",
            "PI_sd", "ARR_full_sd", "SE_Delta",
            "p_value", "psi", "sims", "Nt", "J", "grid",
            "seed_PI", "seed_ARR_full",
        ],
    )

    summary = {
        "label": label,
        "eta": eta,
        "sigma": sigma,
        "q_values": q_values,
        "gamma_values": gamma_values,
        "sims": sims,
        "Nt": Nt,
        "J": J,
        "grid": grid,
        "alpha": alpha,
        "total_cells": int(delta.size),
        "significant_cells_after_BH": int(reject.sum()),
        "significant_red_cells": int(np.sum(delta_sig > 0)),
        "significant_blue_cells": int(np.sum(delta_sig < 0)),
        "mean_delta": float(np.mean(delta)),
        "max_delta": float(np.max(delta)),
        "min_delta": float(np.min(delta)),
    }

    with open(os.path.join(diag_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "summary": summary,
                "point_results": rows,
            },
            f,
            indent=2,
        )

    plot_heatmap(
        q_values,
        gamma_values,
        delta,
        os.path.join(diag_dir, "delta_arrfull_minus_pi_heatmap.png"),
        rf"{label}: ARR-full-FIplug minus PI-FIplug",
        r"$J^{ARR-full-FIplug}-J^{PI-FIplug}$",
    )

    plot_heatmap(
        q_values,
        gamma_values,
        delta_sig,
        os.path.join(diag_dir, "delta_arrfull_minus_pi_sig_heatmap.png"),
        rf"{label}: ARR-full-FIplug minus PI-FIplug, significance-filtered",
        r"displayed $J^{ARR-full-FIplug}-J^{PI-FIplug}$",
    )

    print(f"\nSaved {label} outputs to {diag_dir}")
    print(summary)

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--q_values", type=str, default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--gamma_values", type=str, default="0,1,2,3,4,5,6,7,8,9,10")

    parser.add_argument("--sims", type=int, default=100000)
    parser.add_argument("--Nt", type=int, default=1000)
    parser.add_argument("--J", type=int, default=7)
    parser.add_argument("--grid", type=str, default="equidistant", choices=["equidistant", "equal_probability"])

    parser.add_argument("--q_bar", type=int, default=50)
    parser.add_argument("--fixed_psi", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7063101)
    parser.add_argument("--alpha", type=float, default=0.05)

    parser.add_argument("--outdir", type=str, default="outputs_full_arrival/45e_appendix_diagnostics")

    parser.add_argument(
        "--diagnostics",
        type=str,
        default="eta1_sigma1,eta10_sigma2,eta1_sigma2",
        help="Comma-separated subset of eta1_sigma1, eta10_sigma2, eta1_sigma2.",
    )

    args = parser.parse_args()

    ensure_dir(args.outdir)

    q_values = parse_float_list(args.q_values)
    gamma_values = parse_float_list(args.gamma_values)

    diagnostic_map = {
        "eta1_sigma1": {"eta": 1.0, "sigma": 1.0},
        "eta10_sigma2": {"eta": 10.0, "sigma": 2.0},
        "eta1_sigma2": {"eta": 1.0, "sigma": 2.0},
    }

    selected = [x.strip() for x in args.diagnostics.split(",") if x.strip()]

    all_summaries = []
    for idx, label in enumerate(selected):
        if label not in diagnostic_map:
            raise ValueError(f"Unknown diagnostic label: {label}")

        pars = diagnostic_map[label]
        summary = run_one_diagnostic(
            label=label,
            eta=pars["eta"],
            sigma=pars["sigma"],
            q_values=q_values,
            gamma_values=gamma_values,
            sims=args.sims,
            Nt=args.Nt,
            J=args.J,
            grid=args.grid,
            q_bar=args.q_bar,
            fixed_psi=args.fixed_psi,
            seed=args.seed + idx * 10000000,
            alpha=args.alpha,
            outdir=args.outdir,
        )
        all_summaries.append(summary)

    with open(os.path.join(args.outdir, "all_diagnostics_summary.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\nAll diagnostics completed.")
    print("Saved combined summary to:", os.path.join(args.outdir, "all_diagnostics_summary.json"))


if __name__ == "__main__":
    main()
