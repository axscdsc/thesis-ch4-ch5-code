from __future__ import annotations

import argparse
import csv
import inspect
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from exp56_train_compare_arrfull_nn_point_diagnostic import (
    load_nn_model,
    recover_nn_quotes_numpy,
)


def call_load_nn_model(model_dir: Path, device: torch.device):
    sig = inspect.signature(load_nn_model)
    params = list(sig.parameters)

    try:
        if len(params) >= 2:
            out = load_nn_model(model_dir, device)
        else:
            out = load_nn_model(model_dir)
    except TypeError:
        out = load_nn_model(str(model_dir), device)

    if not isinstance(out, tuple):
        raise TypeError(f"load_nn_model returned {type(out)}, expected tuple.")

    if len(out) < 5:
        raise ValueError(f"load_nn_model returned tuple of length {len(out)}, expected at least 5.")

    model, p_nn, theta_t, L_t, quote_grid = out[:5]
    extra = out[5:] if len(out) > 5 else ()
    return model, p_nn, theta_t, L_t, quote_grid, extra


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.std(x) < 1e-14 or np.std(y) < 1e-14:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--t_value", type=float, default=0.5)
    parser.add_argument("--Q_value", type=int, default=0)
    parser.add_argument("--batch_paths", type=int, default=512)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    print("=== Quote-vs-belief diagnostic ===", flush=True)
    print("model_dir:", model_dir, flush=True)
    print("outdir:", outdir, flush=True)
    print("device:", device, flush=True)

    model, p_nn, theta_t, L_t, quote_grid, extra = call_load_nn_model(model_dir, device)

    theta = theta_t.detach().cpu().numpy().astype(float)
    J = len(theta)

    print("theta:", theta, flush=True)
    print("J:", J, flush=True)
    if torch.is_tensor(quote_grid):
        quote_grid_np = quote_grid.detach().cpu().numpy()
    else:
        quote_grid_np = np.asarray(quote_grid, dtype=float)

    print("quote_grid_min_max:", float(np.min(quote_grid_np)), float(np.max(quote_grid_np)), flush=True)

    rng = np.random.default_rng(args.seed)

    # Mix centre-heavy and corner-heavy simplex samples.
    n1 = args.n_samples // 2
    n2 = args.n_samples - n1
    pi_centre = rng.dirichlet(np.ones(J) * 2.0, size=n1)
    pi_corner = rng.dirichlet(np.ones(J) * 0.35, size=n2)
    pi = np.vstack([pi_centre, pi_corner])

    # Add deterministic vertices and centre if J=3.
    extra_pi = [np.ones(J) / J]
    for j in range(J):
        e = np.zeros(J)
        e[j] = 1.0
        extra_pi.append(e)
    pi = np.vstack([pi, np.asarray(extra_pi)])

    Q = np.full(pi.shape[0], args.Q_value, dtype=int)
    m_pi = pi @ theta

    rho_a, rho_b = recover_nn_quotes_numpy(
        model=model,
        p_nn=p_nn,
        theta_t=theta_t,
        L_t=L_t,
        quote_grid=quote_grid,
        t_value=float(args.t_value),
        Q_np=Q,
        pi_np=pi,
        device=device,
        batch_paths=int(args.batch_paths),
    )

    rho_a = np.asarray(rho_a, dtype=float)
    rho_b = np.asarray(rho_b, dtype=float)

    summary = {
        "model_dir": str(model_dir),
        "outdir": str(outdir),
        "device": str(device),
        "J": int(J),
        "theta": theta.tolist(),
        "t_value": float(args.t_value),
        "Q_value": int(args.Q_value),
        "n_samples": int(pi.shape[0]),
        "m_pi_mean": float(np.mean(m_pi)),
        "m_pi_sd": float(np.std(m_pi)),
        "rho_a_mean": float(np.mean(rho_a)),
        "rho_a_sd": float(np.std(rho_a)),
        "rho_a_min": float(np.min(rho_a)),
        "rho_a_max": float(np.max(rho_a)),
        "rho_b_mean": float(np.mean(rho_b)),
        "rho_b_sd": float(np.std(rho_b)),
        "rho_b_min": float(np.min(rho_b)),
        "rho_b_max": float(np.max(rho_b)),
        "corr_m_pi_rho_a": safe_corr(m_pi, rho_a),
        "corr_m_pi_rho_b": safe_corr(m_pi, rho_b),
    }

    for j in range(J):
        summary[f"corr_pi{j}_rho_a"] = safe_corr(pi[:, j], rho_a)
        summary[f"corr_pi{j}_rho_b"] = safe_corr(pi[:, j], rho_b)

    print(json.dumps(summary, indent=2), flush=True)

    with open(outdir / "quote_vs_belief_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(outdir / "quote_vs_belief_samples.csv", "w", newline="") as f:
        fieldnames = ["m_pi", "rho_a", "rho_b"] + [f"pi_{j}" for j in range(J)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(pi.shape[0]):
            row = {
                "m_pi": float(m_pi[i]),
                "rho_a": float(rho_a[i]),
                "rho_b": float(rho_b[i]),
            }
            for j in range(J):
                row[f"pi_{j}"] = float(pi[i, j])
            writer.writerow(row)

    # Plot ask quote vs posterior mean.
    plt.figure(figsize=(6.2, 4.4))
    plt.scatter(m_pi, rho_a, s=8, alpha=0.35)
    plt.xlabel(r"$m(\pi)=\sum_j \pi^j \theta_j$")
    plt.ylabel(r"Recovered ask quote $\rho^a$")
    plt.title("NN ask quote vs posterior mean")
    plt.tight_layout()
    plt.savefig(outdir / "rho_a_vs_mpi.png", dpi=200)
    plt.close()

    # Plot bid quote vs posterior mean.
    plt.figure(figsize=(6.2, 4.4))
    plt.scatter(m_pi, rho_b, s=8, alpha=0.35)
    plt.xlabel(r"$m(\pi)=\sum_j \pi^j \theta_j$")
    plt.ylabel(r"Recovered bid quote $\rho^b$")
    plt.title("NN bid quote vs posterior mean")
    plt.tight_layout()
    plt.savefig(outdir / "rho_b_vs_mpi.png", dpi=200)
    plt.close()

    # If J=3, also plot along a simple line pi=(s,1-s,0).
    if J == 3:
        s_grid = np.linspace(0.0, 1.0, 201)
        pi_line = np.column_stack([s_grid, 1.0 - s_grid, np.zeros_like(s_grid)])
        Q_line = np.full(pi_line.shape[0], args.Q_value, dtype=int)
        m_line = pi_line @ theta

        ra_line, rb_line = recover_nn_quotes_numpy(
            model=model,
            p_nn=p_nn,
            theta_t=theta_t,
            L_t=L_t,
            quote_grid=quote_grid,
            t_value=float(args.t_value),
            Q_np=Q_line,
            pi_np=pi_line,
            device=device,
            batch_paths=int(args.batch_paths),
        )

        plt.figure(figsize=(6.2, 4.4))
        plt.plot(m_line, ra_line, label=r"$\rho^a$")
        plt.plot(m_line, rb_line, label=r"$\rho^b$")
        plt.xlabel(r"$m(\pi)$ along $\pi=(s,1-s,0)$")
        plt.ylabel("Recovered quote")
        plt.title("NN quotes along a simplex edge")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "quotes_along_simplex_edge.png", dpi=200)
        plt.close()

    print("Saved outputs to:", outdir, flush=True)


if __name__ == "__main__":
    main()
