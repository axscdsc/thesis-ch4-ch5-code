import argparse
import json
import math
from pathlib import Path

import numpy as np


def mean_sd(x):
    return float(np.mean(x)), float(np.std(x, ddof=1))


def simulate_constant_quote_aligned(
    sims: int,
    Nt: int,
    seed: int,
    outdir: str,
    q: float = 0.6,
    gamma: float = 6.0,
    psi: float = 11.015088214118308,
    T: float = 1.0,
    eta: float = 10.0,
    sigma: float = 1.0,
    mu: float = 0.0,
    alpha: float = 0.001,
    phi_inventory: float = 0.1,
    varphi: float = 15.0,
    k: float = 1.0,
    q_bar: int = 50,
    S0: float = 0.0,
    U0: float = 0.0,
    Q0: float = 0.0,
    X0: float = 0.0,
):

    rng = np.random.default_rng(seed)

    dt = T / Nt
    Q_min, Q_max = -int(q_bar), int(q_bar)
    pconst = math.sqrt(max(1.0 - q ** 2, 0.0))

    delta_const = 1.0 / k

    S = np.full(sims, S0, dtype=float)
    U = np.full(sims, U0, dtype=float)
    Q = np.full(sims, Q0, dtype=float)
    X = np.full(sims, X0, dtype=float)
    int_Q2 = np.zeros(sims, dtype=float)

    mean_rho_a, mean_rho_b, mean_Q = [], [], []
    mean_Ma, mean_Mb, mean_Na, mean_Nb = [], [], [], []

    for it in range(Nt):
        S_prev = S.copy()
        U_prev = U.copy()

        rho_a = np.full(sims, delta_const, dtype=float)
        rho_b = np.full(sims, delta_const, dtype=float)

        allow_a = (Q > Q_min).astype(float)
        allow_b = (Q < Q_max).astype(float)

        fad_true = gamma * sigma * q * U
        ell_a_true = varphi + psi * np.exp(-fad_true)
        ell_b_true = varphi + psi * np.exp(+fad_true)

        dMa = rng.poisson(np.maximum(ell_a_true * dt, 0.0)).astype(int)
        dMb = rng.poisson(np.maximum(ell_b_true * dt, 0.0)).astype(int)

        pfill_a = np.clip(np.exp(-k * rho_a) * allow_a, 0.0, 1.0)
        pfill_b = np.clip(np.exp(-k * rho_b) * allow_b, 0.0, 1.0)

        dNa = rng.binomial(dMa, pfill_a).astype(float)
        dNb = rng.binomial(dMb, pfill_b).astype(float)

        dNa = np.minimum(dNa, np.maximum(Q - Q_min, 0.0))
        dNb = np.minimum(dNb, np.maximum(Q_max - Q, 0.0))

        X += dNa * (S_prev + rho_a) - dNb * (S_prev - rho_b)
        Q += dNb - dNa
        int_Q2 += Q ** 2 * dt

        dB = rng.normal(0.0, math.sqrt(dt), sims)
        dZ = rng.normal(0.0, math.sqrt(dt), sims)

        U = U_prev - eta * U_prev * dt + dB
        S = S_prev + mu * dt + sigma * (q * (U - U_prev) + pconst * dZ)

        if it in {0, Nt // 2, Nt - 1}:
            mean_rho_a.append(float(np.mean(rho_a)))
            mean_rho_b.append(float(np.mean(rho_b)))
            mean_Q.append(float(np.mean(Q)))
            mean_Ma.append(float(np.mean(dMa)))
            mean_Mb.append(float(np.mean(dMb)))
            mean_Na.append(float(np.mean(dNa)))
            mean_Nb.append(float(np.mean(dNb)))

    perf = X + Q * S - alpha * Q ** 2 - phi_inventory * int_Q2
    mean, sd = mean_sd(perf)
    se = sd / math.sqrt(float(sims))

    result = {
        "strategy": "constant_quote_aligned",
        "delta_a": delta_const,
        "delta_b": delta_const,
        "q": q,
        "gamma": gamma,
        "psi": psi,
        "sims": sims,
        "Nt": Nt,
        "seed": seed,
        "T": T,
        "eta": eta,
        "sigma": sigma,
        "mu": mu,
        "alpha": alpha,
        "phi_inventory": phi_inventory,
        "varphi": varphi,
        "k": k,
        "q_bar": q_bar,
        "mean": mean,
        "sd": sd,
        "se": se,
        "mean_final_Q": float(np.mean(Q)),
        "sd_final_Q": float(np.std(Q, ddof=1)),
        "mean_quote_a_start_mid_end": mean_rho_a,
        "mean_quote_b_start_mid_end": mean_rho_b,
        "mean_Q_start_mid_end": mean_Q,
        "mean_full_arrivals_a_start_mid_end": mean_Ma,
        "mean_full_arrivals_b_start_mid_end": mean_Mb,
        "mean_fills_a_start_mid_end": mean_Na,
        "mean_fills_b_start_mid_end": mean_Nb,
    }

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "constant_quote_aligned_result.json").write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print("Saved to:", out / "constant_quote_aligned_result.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=20000)
    ap.add_argument("--Nt", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=36457656 + 991)
    ap.add_argument("--outdir", type=str, default="outputs_ch5_arrfull/constant_quote_aligned/q060_g600")
    args = ap.parse_args()

    simulate_constant_quote_aligned(
        sims=args.sims,
        Nt=args.Nt,
        seed=args.seed,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
