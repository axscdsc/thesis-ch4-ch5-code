import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from exp55_train_compare_arrfull_nn_point_v2_ctmconly import (
    load_nn_model,
    recover_nn_quotes_numpy,
)


def plot_quote_vs_inventory(model_dir, outdir, device, t_value=0.5, q_min=-20, q_max=20):
    model_dir = Path(model_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model, p_nn, theta_t, L_t, quote_grid, cfg = load_nn_model(model_dir, device)

    J = int(cfg["J"])
    Q_vals = np.arange(q_min, q_max + 1, dtype=float)

    # Neutral belief: no posterior tilt toward any fad state.
    pi = np.ones((len(Q_vals), J), dtype=float) / J

    rho_a, rho_b = recover_nn_quotes_numpy(
        model=model,
        p_nn=p_nn,
        theta_t=theta_t,
        L_t=L_t,
        quote_grid=quote_grid,
        t_value=t_value,
        Q_np=Q_vals,
        pi_np=pi,
        device=device,
        batch_paths=4096,
    )

    rho_a = np.asarray(rho_a, dtype=float)
    rho_b = np.asarray(rho_b, dtype=float)

    # Save CSV for reproducibility.
    csv_path = outdir / "nn_quote_vs_inventory.csv"
    with csv_path.open("w") as f:
        f.write("Q,rho_a,rho_b\n")
        for Q, ra, rb in zip(Q_vals, rho_a, rho_b):
            f.write(f"{Q:.0f},{ra:.10f},{rb:.10f}\n")

    # Plot ask and bid displacements against inventory.
    plt.figure(figsize=(7, 4.5))
    plt.plot(Q_vals, rho_a, marker="o", markersize=3, label=r"Ask displacement $\delta^a$")
    plt.plot(Q_vals, rho_b, marker="s", markersize=3, label=r"Bid displacement $\delta^b$")
    plt.axvline(0, linewidth=1, linestyle="--")
    plt.xlabel(r"Inventory $Q$")
    plt.ylabel("Recovered quote displacement")
    plt.title(rf"NN learned quotes vs inventory, $t={t_value}$, neutral belief")
    plt.legend()
    plt.tight_layout()

    fig_path = outdir / "nn_quote_vs_inventory.png"
    plt.savefig(fig_path, dpi=220)
    plt.close()

    summary = {
        "model_dir": str(model_dir),
        "t_value": t_value,
        "J": J,
        "Q_min": q_min,
        "Q_max": q_max,
        "rho_a_min": float(np.min(rho_a)),
        "rho_a_max": float(np.max(rho_a)),
        "rho_b_min": float(np.min(rho_b)),
        "rho_b_max": float(np.max(rho_b)),
        "csv": str(csv_path),
        "figure": str(fig_path),
    }

    (outdir / "nn_quote_vs_inventory_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def plot_quote_vs_belief(model_dir, outdir, device, t_value=0.5, Q_value=0.0, n_points=101):
    model_dir = Path(model_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model, p_nn, theta_t, L_t, quote_grid, cfg = load_nn_model(model_dir, device)

    J = int(cfg["J"])
    if J != 3:
        raise ValueError("This simple belief path currently assumes J=3.")

    theta = theta_t.detach().cpu().numpy().astype(float)

    # Simple belief path from low-fad state to high-fad state:
    # pi(w) = (1-w, 0, w). This makes posterior mean move monotonically.
    w_vals = np.linspace(0.0, 1.0, n_points)
    pi = np.zeros((n_points, J), dtype=float)
    pi[:, 0] = 1.0 - w_vals
    pi[:, 1] = 0.0
    pi[:, 2] = w_vals

    m_pi = pi @ theta
    Q_vals = np.full(n_points, float(Q_value), dtype=float)

    rho_a, rho_b = recover_nn_quotes_numpy(
        model=model,
        p_nn=p_nn,
        theta_t=theta_t,
        L_t=L_t,
        quote_grid=quote_grid,
        t_value=t_value,
        Q_np=Q_vals,
        pi_np=pi,
        device=device,
        batch_paths=4096,
    )

    rho_a = np.asarray(rho_a, dtype=float)
    rho_b = np.asarray(rho_b, dtype=float)

    csv_path = outdir / "nn_quote_vs_belief.csv"
    with csv_path.open("w") as f:
        f.write("w,m_pi,pi_0,pi_1,pi_2,rho_a,rho_b\n")
        for w, m, p0, p1, p2, ra, rb in zip(w_vals, m_pi, pi[:,0], pi[:,1], pi[:,2], rho_a, rho_b):
            f.write(f"{w:.10f},{m:.10f},{p0:.10f},{p1:.10f},{p2:.10f},{ra:.10f},{rb:.10f}\n")

    plt.figure(figsize=(7, 4.5))
    plt.plot(m_pi, rho_a, marker="o", markersize=3, label=r"Ask displacement $\delta^a$")
    plt.plot(m_pi, rho_b, marker="s", markersize=3, label=r"Bid displacement $\delta^b$")
    plt.axvline(0, linewidth=1, linestyle="--")
    plt.xlabel(r"Posterior mean $m(\pi)$")
    plt.ylabel("Recovered quote displacement")
    plt.title(rf"NN learned quotes vs belief, $t={t_value}$, $Q={Q_value}$")
    plt.legend()
    plt.tight_layout()

    fig_path = outdir / "nn_quote_vs_belief.png"
    plt.savefig(fig_path, dpi=220)
    plt.close()

    summary = {
        "model_dir": str(model_dir),
        "t_value": t_value,
        "Q_value": Q_value,
        "J": J,
        "theta": theta.tolist(),
        "m_pi_min": float(np.min(m_pi)),
        "m_pi_max": float(np.max(m_pi)),
        "rho_a_min": float(np.min(rho_a)),
        "rho_a_max": float(np.max(rho_a)),
        "rho_b_min": float(np.min(rho_b)),
        "rho_b_max": float(np.max(rho_b)),
        "corr_m_pi_rho_a": float(np.corrcoef(m_pi, rho_a)[0, 1]) if np.std(rho_a) > 0 else None,
        "corr_m_pi_rho_b": float(np.corrcoef(m_pi, rho_b)[0, 1]) if np.std(rho_b) > 0 else None,
        "csv": str(csv_path),
        "figure": str(fig_path),
    }

    (outdir / "nn_quote_vs_belief_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--t_value", type=float, default=0.5)
    ap.add_argument("--q_min", type=int, default=-20)
    ap.add_argument("--q_max", type=int, default=20)
    ap.add_argument("--plot", type=str, default="inventory", choices=["inventory", "belief"])
    ap.add_argument("--Q_value", type=float, default=0.0)
    ap.add_argument("--n_points", type=int, default=101)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    if args.plot == "inventory":
        plot_quote_vs_inventory(
            model_dir=args.model_dir,
            outdir=args.outdir,
            device=device,
            t_value=args.t_value,
            q_min=args.q_min,
            q_max=args.q_max,
        )
    elif args.plot == "belief":
        plot_quote_vs_belief(
            model_dir=args.model_dir,
            outdir=args.outdir,
            device=device,
            t_value=args.t_value,
            Q_value=args.Q_value,
            n_points=args.n_points,
        )


if __name__ == "__main__":
    main()
