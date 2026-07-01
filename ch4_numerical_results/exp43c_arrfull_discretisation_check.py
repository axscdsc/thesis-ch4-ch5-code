import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from clean_fiplug_core import ensure_dir, make_clean_params
from exp43b_full_arrival_filter_mse import simulate_filter_mse_full_arrival


def parse_points(text):
    points = []
    for item in text.split(","):
        q_str, gamma_str = item.split(":")
        points.append((float(q_str), float(gamma_str)))
    return points


def parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def conclusion_from_delta(delta):
    if delta > 0:
        return "PI more accurate"
    if delta < 0:
        return "ARR-full more accurate"
    return "Tie"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=str, default="0.5:2,0.6:1,0.6:3,0.9:1")
    parser.add_argument("--J_values", type=str, default="3,5,7,15")
    parser.add_argument("--grids", type=str, default="equidistant,equal_probability")
    parser.add_argument("--sims", type=int, default=10000)
    parser.add_argument("--Nt", type=int, default=1000)
    parser.add_argument("--q_bar", type=int, default=50)
    parser.add_argument("--fixed_psi", type=float, default=None)
    parser.add_argument("--quote_policy", type=str, default="zero_fiplug")
    parser.add_argument("--seed", type=int, default=4759103)
    parser.add_argument("--outdir", type=str, default="outputs_full_arrival/43c_arrfull_discretisation_check")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    points = parse_points(args.points)
    J_values = parse_int_list(args.J_values)
    grids = parse_str_list(args.grids)

    detailed_rows = []
    total = len(points) * len(grids) * len(J_values)
    counter = 0

    for point_idx, (q, gamma) in enumerate(points):
        for grid_idx, grid in enumerate(grids):
            for j_idx, J in enumerate(J_values):
                counter += 1
                seed_point = int(args.seed + 100000 * point_idx + 10000 * grid_idx + 1000 * j_idx)

                print(
                    f"[{counter}/{total}] q={q:g}, gamma={gamma:g}, grid={grid}, J={J}, seed={seed_point}",
                    flush=True,
                )

                mm = make_clean_params(
                    q=float(q),
                    gamma=float(gamma),
                    q_bar=args.q_bar,
                    fixed_psi=args.fixed_psi,
                    Nt_for_recalib=args.Nt,
                )

                res = simulate_filter_mse_full_arrival(
                    mm,
                    sims=args.sims,
                    Nt=args.Nt,
                    J=J,
                    seed=seed_point,
                    grid=grid,
                    quote_policy=args.quote_policy,
                )

                delta = float(res["MSE_ARR_full_minus_PI"])

                detailed_rows.append({
                    "q": q,
                    "gamma": gamma,
                    "grid": grid,
                    "J": J,
                    "MSE_PI": float(res["MSE_PI"]),
                    "MSE_ARR_full": float(res["MSE_ARR_full"]),
                    "Delta_MSE": delta,
                    "Conclusion": conclusion_from_delta(delta),
                    "psi": float(res["psi"]),
                    "sims": args.sims,
                    "Nt": args.Nt,
                    "seed": seed_point,
                })

    detailed_path = f"{args.outdir}/arrfull_discretisation_detailed.csv"
    write_csv(
        detailed_path,
        detailed_rows,
        ["q", "gamma", "grid", "J", "MSE_PI", "MSE_ARR_full", "Delta_MSE", "Conclusion", "psi", "sims", "Nt", "seed"],
    )

    grouped = defaultdict(list)
    for row in detailed_rows:
        grouped[(row["q"], row["gamma"], row["grid"])].append(row)

    summary_rows = []
    for (q, gamma, grid), rows in grouped.items():
        arr_vals = np.array([r["MSE_ARR_full"] for r in rows], dtype=float)
        delta_vals = np.array([r["Delta_MSE"] for r in rows], dtype=float)

        arr_range = float(arr_vals.max() - arr_vals.min())
        delta_range = float(delta_vals.max() - delta_vals.min())
        delta_mean = float(delta_vals.mean())

        signs = np.sign(delta_vals)
        if np.all(signs > 0):
            conclusion = "PI more accurate"
        elif np.all(signs < 0):
            conclusion = "ARR-full more accurate"
        else:
            conclusion = "Mixed across J"

        summary_rows.append({
            "q": q,
            "gamma": gamma,
            "grid": grid,
            "J_values": ",".join(str(J) for J in J_values),
            "Range_ARR_full_MSE": arr_range,
            "Range_Delta_MSE": delta_range,
            "Mean_Delta_MSE": delta_mean,
            "Conclusion": conclusion,
        })

    summary_rows = sorted(summary_rows, key=lambda r: (r["q"], r["gamma"], r["grid"]))

    summary_path = f"{args.outdir}/arrfull_discretisation_summary.csv"
    write_csv(
        summary_path,
        summary_rows,
        ["q", "gamma", "grid", "J_values", "Range_ARR_full_MSE", "Range_Delta_MSE", "Mean_Delta_MSE", "Conclusion"],
    )

    with open(f"{args.outdir}/arrfull_discretisation_summary.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "points": points,
                "J_values": J_values,
                "grids": grids,
                "detailed_rows": detailed_rows,
                "summary_rows": summary_rows,
            },
            f,
            indent=2,
        )

    print("\nSaved:")
    print("  ", detailed_path)
    print("  ", summary_path)


if __name__ == "__main__":
    main()
