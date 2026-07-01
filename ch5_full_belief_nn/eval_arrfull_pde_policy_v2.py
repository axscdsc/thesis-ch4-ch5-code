import argparse
import csv
import json
import math
import os

import numpy as np

try:
    from scipy.linalg import expm
except Exception:
    expm = None

from clean_fiplug_core import make_clean_params


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def nearest_pi_index(pi, pi_grid):
    pi = np.asarray(pi, dtype=float)
    pi = np.maximum(pi, 1e-14)
    pi = pi / pi.sum()
    d2 = np.sum((pi_grid - pi[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))


def q_to_index(Q, q_grid):
    Q = int(Q)
    if Q <= int(q_grid[0]):
        return 0
    if Q >= int(q_grid[-1]):
        return len(q_grid) - 1
    return int(Q - int(q_grid[0]))


def time_to_index(t, T, Nt_policy):
    n = int((t / T) * Nt_policy)
    if n < 0:
        return 0
    if n >= Nt_policy:
        return Nt_policy - 1
    return n


def stationary_distribution_from_generator(G):
    # solve pi G = 0, sum pi = 1
    J = G.shape[0]
    A = G.T.copy()
    A[-1, :] = 1.0
    b = np.zeros(J)
    b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        pi = np.ones(J) / J
    pi = np.maximum(pi, 1e-14)
    return pi / pi.sum()


def transition_matrix(G, dt):
    if expm is not None:
        P = expm(G * dt)
    else:
        P = np.eye(G.shape[0]) + G * dt
    P = np.maximum(P, 0.0)
    P = P / P.sum(axis=1, keepdims=True)
    return P


def load_pde_policy(pde_dir):
    delta_a = np.load(os.path.join(pde_dir, "delta_a.npy"))
    delta_b = np.load(os.path.join(pde_dir, "delta_b.npy"))
    q_grid = np.load(os.path.join(pde_dir, "q_grid.npy"))
    pi_grid = np.load(os.path.join(pde_dir, "pi_grid.npy"))
    theta = np.load(os.path.join(pde_dir, "theta.npy"))
    G = np.load(os.path.join(pde_dir, "generator.npy"))
    lam_a = np.load(os.path.join(pde_dir, "lambda_a.npy"))
    lam_b = np.load(os.path.join(pde_dir, "lambda_b.npy"))

    with open(os.path.join(pde_dir, "metadata.json")) as f:
        meta = json.load(f)

    return delta_a, delta_b, q_grid, pi_grid, theta, G, lam_a, lam_b, meta


def simulate_pde_policy_v2(
    pde_dir,
    q,
    gamma,
    sims,
    Nt,
    seed,
    outdir,
):
    rng = np.random.default_rng(seed)

    delta_a, delta_b, q_grid, pi_grid, theta, G, lam_a_vec, lam_b_vec, meta = load_pde_policy(pde_dir)

    mm = make_clean_params(q=float(q), gamma=float(gamma), q_bar=50, fixed_psi=None, Nt_for_recalib=Nt)

    T = float(getattr(mm, "T", 1.0))
    dt = T / Nt

    k = float(getattr(mm, "k", 1.0))
    alpha = float(getattr(mm, "alpha", 0.001))
    phi_inv = float(getattr(mm, "phi", getattr(mm, "phi_inventory", 0.1)))

    sigma = float(getattr(mm, "sigma", 1.0))
    eta = float(getattr(mm, "eta", 10.0))
    mu = float(getattr(mm, "mu", 0.0))
    qconst = float(getattr(mm, "qconst", getattr(mm, "q", q)))
    pconst = math.sqrt(max(1.0 - qconst ** 2, 0.0))

    S0 = float(getattr(mm, "S_0", 0.0))
    U0 = float(getattr(mm, "U_0", 0.0))
    Q0 = int(getattr(mm, "Q_0", 0.0))
    X0 = float(getattr(mm, "X_0", 0.0))

    varphi = float(getattr(mm, "varphi", 15.0))
    psi = float(getattr(mm, "psi"))

    policy_Nt = delta_a.shape[0]
    Qmin = int(q_grid[0])
    Qmax = int(q_grid[-1])

    P = transition_matrix(G, dt)
    pi0 = stationary_distribution_from_generator(G)

    values = np.zeros(sims, dtype=float)

    mean_rho_a = []
    mean_rho_b = []
    mean_Q = []
    mean_Ma = []
    mean_Mb = []
    mean_Na = []
    mean_Nb = []

    for r in range(sims):
        S = float(S0)
        U = float(U0)
        Q = int(Q0)
        X = float(X0)
        int_Q2 = 0.0
        pi = pi0.copy()

        rho_a_track = []
        rho_b_track = []
        Q_track = []
        Ma_track = []
        Mb_track = []
        Na_track = []
        Nb_track = []

        for it in range(Nt):
            S_prev = S
            U_prev = U
            t_value = it * dt

            nt_pol = time_to_index(t_value, T, policy_Nt)
            iq = q_to_index(Q, q_grid)
            ip = nearest_pi_index(pi, pi_grid)

            rho_a = float(delta_a[nt_pol, iq, ip])
            rho_b = float(delta_b[nt_pol, iq, ip])
            rho_a = max(rho_a, 1e-8)
            rho_b = max(rho_b, 1e-8)

            # True full-arrival intensities from continuous U, matching NN simulator.
            fad_true = gamma * sigma * qconst * U_prev
            ell_a_true = varphi + psi * math.exp(-fad_true)
            ell_b_true = varphi + psi * math.exp(+fad_true)

            dMa = rng.poisson(max(ell_a_true * dt, 0.0))
            dMb = rng.poisson(max(ell_b_true * dt, 0.0))

            allow_a = Q > Qmin
            allow_b = Q < Qmax

            pfill_a = min(max(math.exp(-k * rho_a), 0.0), 1.0) if allow_a else 0.0
            pfill_b = min(max(math.exp(-k * rho_b), 0.0), 1.0) if allow_b else 0.0

            dNa = rng.binomial(dMa, pfill_a) if dMa > 0 else 0
            dNb = rng.binomial(dMb, pfill_b) if dMb > 0 else 0

            dNa = min(dNa, max(Q - Qmin, 0))
            dNb = min(dNb, max(Qmax - Q, 0))

            X += dNa * (S_prev + rho_a) - dNb * (S_prev - rho_b)
            Q += dNb - dNa
            int_Q2 += (Q ** 2) * dt

            # Continuous OU and price dynamics, matching NN simulator order.
            dB = rng.normal(0.0, math.sqrt(dt))
            dZ = rng.normal(0.0, math.sqrt(dt))
            U = U_prev - eta * U_prev * dt + dB
            S = S_prev + mu * dt + sigma * (qconst * (U - U_prev) + pconst * dZ)

            # Belief prediction and full-arrival likelihood update.
            pi_pred = pi @ P

            fad_j = gamma * sigma * qconst * theta
            ell_a_j = np.maximum(varphi + psi * np.exp(-fad_j), 1e-12)
            ell_b_j = np.maximum(varphi + psi * np.exp(+fad_j), 1e-12)

            log_like = (
                dMa * np.log(ell_a_j * dt) - ell_a_j * dt
                + dMb * np.log(ell_b_j * dt) - ell_b_j * dt
            )
            log_like -= np.max(log_like)

            weights = pi_pred * np.exp(log_like)
            denom = float(weights.sum())
            if denom <= 1e-14:
                pi = pi0.copy()
            else:
                pi = weights / denom

            if it in {0, Nt // 2, Nt - 1}:
                rho_a_track.append(rho_a)
                rho_b_track.append(rho_b)
                Q_track.append(Q)
                Ma_track.append(dMa)
                Mb_track.append(dMb)
                Na_track.append(dNa)
                Nb_track.append(dNb)

        values[r] = X + Q * S - alpha * (Q ** 2) - phi_inv * int_Q2

        if r == 0:
            mean_rho_a = rho_a_track
            mean_rho_b = rho_b_track
            mean_Q = Q_track
            mean_Ma = Ma_track
            mean_Mb = Mb_track
            mean_Na = Na_track
            mean_Nb = Nb_track

    ensure_dir(outdir)

    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if sims > 1 else 0.0
    se = float(sd / math.sqrt(sims)) if sims > 0 else float("nan")

    summary = {
        "strategy": "ARR-full-PDE-J3",
        "evaluation": "continuous_U_full_arrival_v2",
        "pde_dir": pde_dir,
        "q": q,
        "gamma": gamma,
        "sims": sims,
        "Nt": Nt,
        "seed": seed,
        "mean": mean,
        "sd": sd,
        "se": se,
        "Qmin": Qmin,
        "Qmax": Qmax,
        "policy_Nt": int(policy_Nt),
        "n_pi": int(len(pi_grid)),
        "psi": psi,
        "varphi": varphi,
        "eta": eta,
        "sigma": sigma,
        "diagnostics_first_path": {
            "rho_a_start_mid_end": mean_rho_a,
            "rho_b_start_mid_end": mean_rho_b,
            "Q_start_mid_end": mean_Q,
            "Ma_start_mid_end": mean_Ma,
            "Mb_start_mid_end": mean_Mb,
            "Na_start_mid_end": mean_Na,
            "Nb_start_mid_end": mean_Nb,
        },
    }

    with open(os.path.join(outdir, "pde_policy_eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(outdir, "pde_policy_values.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "value"])
        for i, v in enumerate(values):
            writer.writerow([i, float(v)])

    print(json.dumps(summary, indent=2))
    print("Saved PDE policy evaluation to:", outdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde_dir", type=str, required=True)
    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--sims", type=int, default=1000)
    parser.add_argument("--Nt", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()

    simulate_pde_policy_v2(
        pde_dir=args.pde_dir,
        q=args.q,
        gamma=args.gamma,
        sims=args.sims,
        Nt=args.Nt,
        seed=args.seed,
        outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
