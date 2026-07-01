import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

from clean_fiplug_core import (
    coefficients_from_mm,
    quote_from_fi_coeff,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=0.6)
    ap.add_argument("--gamma", type=float, default=6.0)
    ap.add_argument("--psi", type=float, default=11.015088214118308)
    ap.add_argument("--t_value", type=float, default=0.5)
    ap.add_argument("--Q_value", type=float, default=0.0)
    ap.add_argument("--Nt", type=int, default=1000)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=10.0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--phi_inventory", type=float, default=0.1)
    ap.add_argument("--varphi", type=float, default=15.0)
    ap.add_argument("--k", type=float, default=1.0)
    ap.add_argument("--u_min", type=float, default=-0.670820415019989)
    ap.add_argument("--u_max", type=float, default=0.670820415019989)
    ap.add_argument("--n_points", type=int, default=101)
    ap.add_argument("--outdir", type=str, default="outputs_ch5_arrfull/policy_shapes/q060_g600")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Match the attribute names expected by coefficients_from_mm.
    mm_aux = SimpleNamespace(
        T=args.T,
        eta=args.eta,
        mu=args.mu,
        sigma=args.sigma,
        qconst=args.q,
        gamma=args.gamma,
        alpha=args.alpha,
        phi=args.phi_inventory,
        varphi=args.varphi,
        psi=args.psi,
        k=args.k,
    )

    A, b0, b1 = coefficients_from_mm(mm_aux, args.Nt)

    t_idx = int(round(args.t_value / args.T * args.Nt))
    t_idx = max(0, min(args.Nt, t_idx))

    U_vals = np.linspace(args.u_min, args.u_max, args.n_points)
    Q_vals = np.full(args.n_points, float(args.Q_value))

    rho_a, rho_b = quote_from_fi_coeff(
        Q=Q_vals,
        U_est=U_vals,
        A=A,
        b0=b0,
        b1=b1,
        k=args.k,
        t_idx=t_idx,
    )

    rho_a = np.asarray(rho_a, dtype=float)
    rho_b = np.asarray(rho_b, dtype=float)

    csv_path = outdir / "fi_quote_vs_u_true.csv"
    with csv_path.open("w") as f:
        f.write("U,rho_a,rho_b\n")
        for u, ra, rb in zip(U_vals, rho_a, rho_b):
            f.write(f"{u:.10f},{ra:.10f},{rb:.10f}\n")

    plt.figure(figsize=(7, 4.5))
    plt.plot(U_vals, rho_a, marker="o", markersize=3, label=r"FI ask displacement $\delta^a$")
    plt.plot(U_vals, rho_b, marker="s", markersize=3, label=r"FI bid displacement $\delta^b$")
    plt.axvline(0, linewidth=1, linestyle="--")
    plt.xlabel(r"Hidden fad $U$")
    plt.ylabel("Quote displacement")
    plt.title(rf"FI quote displacements vs hidden fad, $t={args.t_value}$, $Q={args.Q_value}$")
    plt.legend()
    plt.tight_layout()

    fig_path = outdir / "fi_quote_vs_u_true.png"
    plt.savefig(fig_path, dpi=220)
    plt.close()

    summary = {
        "q": args.q,
        "gamma": args.gamma,
        "psi": args.psi,
        "t_value": args.t_value,
        "t_idx": t_idx,
        "Q_value": args.Q_value,
        "Nt": args.Nt,
        "T": args.T,
        "eta": args.eta,
        "sigma": args.sigma,
        "alpha": args.alpha,
        "phi": args.phi_inventory,
        "varphi": args.varphi,
        "k": args.k,
        "u_min": float(np.min(U_vals)),
        "u_max": float(np.max(U_vals)),
        "A_t": float(A[t_idx]),
        "b0_t": float(b0[t_idx]),
        "b1_t": float(b1[t_idx]),
        "rho_a_min": float(np.min(rho_a)),
        "rho_a_max": float(np.max(rho_a)),
        "rho_b_min": float(np.min(rho_b)),
        "rho_b_max": float(np.max(rho_b)),
        "corr_U_rho_a": float(np.corrcoef(U_vals, rho_a)[0, 1]) if np.std(rho_a) > 0 else None,
        "corr_U_rho_b": float(np.corrcoef(U_vals, rho_b)[0, 1]) if np.std(rho_b) > 0 else None,
        "csv": str(csv_path),
        "figure": str(fig_path),
    }

    (outdir / "fi_quote_vs_u_true_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
