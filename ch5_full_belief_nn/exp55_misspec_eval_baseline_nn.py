import argparse
import csv
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.linalg import expm
from tqdm import tqdm

from clean_fiplug_core import make_clean_params, BASE_PARAMS_ENV as CLEAN_BASE_PARAMS_ENV

from arr_hjb_nn_solver_arrfull_v2_ctmconly import (
    ARRModelParams,
    ValueNet,
    make_ou_grid,
    sample_interior_batch,
    hjb_residual,
    terminal_loss,
    save_loss_plot,
    set_seed,
)

def fmt_float(x: float) -> str:
    return f"{float(x):.2f}".replace(".", "p")


def point_name(q: float, gamma: float) -> str:
    return f"q_{fmt_float(q)}_gamma_{fmt_float(gamma)}"


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested --device cuda, but CUDA is not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _try_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _match(a: float, b: float, tol: float = 1e-8) -> bool:
    return abs(float(a) - float(b)) <= tol


def load_common_fiplug_point(outputs_root: Path, q: float, gamma: float) -> Dict[str, float]:
    out = {
        "PI_FIplug_mean": float("nan"),
        "PI_FIplug_sd": float("nan"),
        "ARR_full_FIplug_mean": float("nan"),
        "ARR_full_FIplug_sd": float("nan"),
        "ARR_full_minus_PI_FIplug": float("nan"),
        "source": "not_found",
    }
    if not outputs_root.exists():
        return out

    def update_from_row(row, source):
        pi_mean = row.get("PI_FIplug_mean", row.get("PI_mean", np.nan))
        pi_sd = row.get("PI_FIplug_sd", row.get("PI_sd", np.nan))
        arr_mean = row.get("ARR_full_FIplug_mean", row.get("ARR_full_mean", np.nan))
        arr_sd = row.get("ARR_full_FIplug_sd", row.get("ARR_full_sd", np.nan))
        delta = row.get("ARR_full_minus_PI_FIplug", row.get("Delta_J", np.nan))
        if not np.isfinite(float(delta)) and np.isfinite(float(pi_mean)) and np.isfinite(float(arr_mean)):
            delta = float(arr_mean) - float(pi_mean)
        return {
            "PI_FIplug_mean": float(pi_mean),
            "PI_FIplug_sd": float(pi_sd),
            "ARR_full_FIplug_mean": float(arr_mean),
            "ARR_full_FIplug_sd": float(arr_sd),
            "ARR_full_minus_PI_FIplug": float(delta),
            "source": str(source),
        }

    for path in outputs_root.rglob("*.json"):
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        rows = None
        if isinstance(obj, list):
            rows = obj
        elif isinstance(obj, dict):
            if "q" in obj and "gamma" in obj:
                rows = [obj]
            else:
                for key in ["point_results", "results", "rows", "records"]:
                    if isinstance(obj.get(key), list):
                        rows = obj[key]
                        break
        if not rows:
            continue
        for row in rows:
            if isinstance(row, dict) and "q" in row and "gamma" in row and _match(row["q"], q) and _match(row["gamma"], gamma):
                if any(k in row for k in ["PI_mean", "PI_FIplug_mean"]):
                    return update_from_row(row, path)

    for path in outputs_root.rglob("*.csv"):
        try:
            import pandas as pd
            df = pd.read_csv(path)
        except Exception:
            continue
        if not {"q", "gamma"}.issubset(set(df.columns)):
            continue
        if not any(c in df.columns for c in ["PI_mean", "PI_FIplug_mean"]):
            continue
        mask = (np.isclose(df["q"].astype(float), q)) & (np.isclose(df["gamma"].astype(float), gamma))
        if mask.any():
            return update_from_row(df.loc[mask].iloc[0].to_dict(), path)
    return out


def build_mm_aux(q: float, gamma: float, seed: int = 12345, sims_for_calibration: int = 8):
    return make_clean_params(q=float(q), gamma=float(gamma), q_bar=50, fixed_psi=None, Nt_for_recalib=1000)

def train_arr_nn_model(args, psi: float, model_dir: Path, device: torch.device) -> Dict[str, float]:
    model_dir.mkdir(parents=True, exist_ok=True)

    p = ARRModelParams(
        T=args.T,
        mu=args.mu,
        sigma=args.sigma,
        eta=args.eta,
        alpha=args.alpha,
        phi_inventory=args.phi_inventory,
        varphi_uninformed=args.varphi,
        psi_informed=psi,
        k=args.k,
        q_loading=args.q,
        gamma=args.gamma,
        q_min=args.q_min,
        q_max=args.q_max,
    )

    theta, L = make_ou_grid(args.J, args.eta, device)
    quote_grid = torch.linspace(args.quote_min, args.quote_max, args.quote_points, device=device)
    model = ValueNet(input_dim=2 + args.J, width=args.width, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    config = vars(args).copy()
    config.update({
        "psi": float(psi),
        "varphi": float(args.varphi),
        "actual_device": str(device),
        "model_type": "ARR_full_NN_full_arrival_filter_HJB",
        "comparison_framework": "PI-FIplug_vs_ARR-full-FIplug_vs_ARR-full-NN",
    })
    if device.type == "cuda":
        config["gpu_name"] = torch.cuda.get_device_name(0)
        config["cuda_version"] = torch.version.cuda

    (model_dir / "config.json").write_text(json.dumps(config, indent=2))

    loss_history = []
    hjb_history = []
    terminal_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()

        t, Q, pi = sample_interior_batch(
            args.batch_size, args.J, device, p,
            args.focus_width, args.focus_prob
        )
        R, _, _, _ = hjb_residual(model, t, Q, pi, theta, L, p, quote_grid)
        loss_hjb = (R ** 2).mean()

        # terminal condition is hard-coded, but keep this as a diagnostic and guard.
        loss_term = terminal_loss(model, args.batch_size, args.J, device, p)
        loss = loss_hjb + args.lambda_term * loss_term

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        opt.step()

        loss_history.append(float(loss.detach().cpu()))
        hjb_history.append(float(loss_hjb.detach().cpu()))
        terminal_history.append(float(loss_term.detach().cpu()))

        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"epoch {epoch:6d} | loss={loss_history[-1]:.6e} | "
                f"HJB={hjb_history[-1]:.6e} | terminal={terminal_history[-1]:.6e}"
            )

    torch.save(model.state_dict(), model_dir / "value_net.pt")

    hist = {
        "loss": loss_history,
        "hjb_loss": hjb_history,
        "terminal_loss": terminal_history,
    }
    (model_dir / "loss_history.json").write_text(json.dumps(hist))
    save_loss_plot(loss_history, model_dir)

    metrics = {
        "final_loss": loss_history[-1],
        "final_hjb_loss": hjb_history[-1],
        "final_terminal_loss": terminal_history[-1],
        "epochs": int(args.epochs),
    }
    (model_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def stationary_distribution_from_generator(L_np: np.ndarray) -> np.ndarray:
    J = L_np.shape[0]
    A = L_np.T.copy()
    b = np.zeros(J)
    A[-1, :] = 1.0
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)
    pi = np.maximum(pi, 0.0)
    pi = pi / pi.sum()
    return pi.astype(np.float64)


def load_nn_model(model_dir: Path, device: torch.device):
    cfg = json.loads((model_dir / "config.json").read_text())
    J = int(cfg["J"])
    p_nn = ARRModelParams(
        T=float(cfg["T"]),
        mu=float(cfg["mu"]),
        sigma=float(cfg["sigma"]),
        eta=float(cfg["eta"]),
        alpha=float(cfg["alpha"]),
        phi_inventory=float(cfg["phi_inventory"]),
        varphi_uninformed=float(cfg["varphi"]),
        psi_informed=float(cfg["psi"]),
        k=float(cfg["k"]),
        q_loading=float(cfg["q"]),
        gamma=float(cfg["gamma"]),
        q_min=int(cfg["q_min"]),
        q_max=int(cfg["q_max"]),
    )
    theta_t, L_t = make_ou_grid(J, p_nn.eta, device)
    model = ValueNet(input_dim=2 + J, width=int(cfg["width"]), depth=int(cfg["depth"])).to(device)
    model.load_state_dict(torch.load(model_dir / "value_net.pt", map_location=device))
    model.eval()
    quote_grid = torch.linspace(
        float(cfg.get("quote_min", 0.01)),
        float(cfg.get("quote_max", 3.0)),
        int(cfg.get("quote_points", 41)),
        device=device,
    )
    return model, p_nn, theta_t, L_t, quote_grid, cfg


def recover_nn_quotes_numpy(model, p_nn, theta_t, L_t, quote_grid, t_value, Q_np, pi_np, device, batch_paths):
    sims = Q_np.shape[0]
    da_out = np.empty(sims, dtype=np.float64)
    db_out = np.empty(sims, dtype=np.float64)

    for start in range(0, sims, batch_paths):
        end = min(start + batch_paths, sims)
        b = end - start

        t = torch.full((b,), float(t_value), dtype=torch.float32, device=device)
        Q = torch.tensor(Q_np[start:end], dtype=torch.float32, device=device)
        pi = torch.tensor(pi_np[start:end], dtype=torch.float32, device=device)

        with torch.enable_grad():
            _, idx, da_flat, db_flat = hjb_residual(model, t, Q, pi, theta_t, L_t, p_nn, quote_grid)

        da_out[start:end] = da_flat[idx].detach().cpu().numpy()
        db_out[start:end] = db_flat[idx].detach().cpu().numpy()

    return da_out, db_out


def mean_sd(x):
    return float(np.mean(x)), float(np.std(x, ddof=1))


def simulate_arr_nn(mm_aux, model_dir: Path, sims: int, Nt: int, seed: int,
                    device: torch.device, batch_paths: int, verbose: bool = True):
    rng = np.random.default_rng(seed)
    model, p_nn, theta_t, L_t, quote_grid, cfg = load_nn_model(model_dir, device)

    T = float(getattr(mm_aux, "T", CLEAN_BASE_PARAMS_ENV.get("T", 1.0)))
    dt = T / Nt
    k = float(getattr(mm_aux, "k", CLEAN_BASE_PARAMS_ENV.get("k", 1.0)))
    phi_inv = float(getattr(mm_aux, "phi", CLEAN_BASE_PARAMS_ENV.get("phi", 0.1)))
    alpha = float(getattr(mm_aux, "alpha", CLEAN_BASE_PARAMS_ENV.get("alpha", 0.001)))
    q_bar = int(getattr(mm_aux, "q_bar", 50))
    Q_min, Q_max = -q_bar, q_bar

    sigma = float(getattr(mm_aux, "sigma", 1.0))
    eta = float(getattr(mm_aux, "eta", 10.0))
    qconst = float(getattr(mm_aux, "qconst"))
    pconst = math.sqrt(max(1.0 - qconst ** 2, 0.0))
    mu = float(getattr(mm_aux, "mu", 0.0))
    S0 = float(getattr(mm_aux, "S_0", 0.0))
    U0 = float(getattr(mm_aux, "U_0", 0.0))
    Q0 = float(getattr(mm_aux, "Q_0", 0.0))
    X0 = float(getattr(mm_aux, "X_0", 0.0))
    varphi = float(getattr(mm_aux, "varphi", 15.0))
    psi = float(getattr(mm_aux, "psi"))
    gamma = float(getattr(mm_aux, "gamma"))

    checks = [("q_min", p_nn.q_min, Q_min), ("q_max", p_nn.q_max, Q_max), ("q", p_nn.q_loading, qconst),
              ("varphi", p_nn.varphi, varphi),
              ("T", p_nn.T, T), ("eta", p_nn.eta, eta), ("sigma", p_nn.sigma, sigma),
              ("mu", p_nn.mu, mu), ("alpha", p_nn.alpha, alpha), ("phi_inventory", p_nn.phi_inventory, phi_inv), ("k", p_nn.k, k)]
    for name, nn_val, sim_val in checks:
        if abs(float(nn_val) - float(sim_val)) > 1e-7:
            raise ValueError(f"NN parameter {name}={nn_val} does not match simulation {sim_val}.")

    theta_np = theta_t.detach().cpu().numpy()
    L_np = L_t.detach().cpu().numpy()
    trans = expm(L_np * dt)
    trans = np.maximum(trans, 0.0)
    trans = trans / trans.sum(axis=1, keepdims=True)
    pi0 = stationary_distribution_from_generator(L_np)

    S = np.full(sims, S0, dtype=float)
    U = np.full(sims, U0, dtype=float)
    Q = np.full(sims, Q0, dtype=float)
    X = np.full(sims, X0, dtype=float)
    int_Q2 = np.zeros(sims, dtype=float)
    pi = np.tile(pi0, (sims, 1))

    mean_rho_a, mean_rho_b, mean_Q = [], [], []
    mean_Ma, mean_Mb, mean_Na, mean_Nb = [], [], [], []
    iterator = tqdm(range(Nt), desc="ARR-full-NN simulation") if verbose else range(Nt)

    for it in iterator:
        S_prev = S.copy()
        U_prev = U.copy()
        t_value = it * dt

        rho_a, rho_b = recover_nn_quotes_numpy(model, p_nn, theta_t, L_t, quote_grid, t_value, Q, pi, device, batch_paths)
        rho_a = np.maximum(rho_a, 1e-8)
        rho_b = np.maximum(rho_b, 1e-8)

        allow_a = (Q > Q_min).astype(float)
        allow_b = (Q < Q_max).astype(float)

        fad_true = gamma * sigma * qconst * U
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
        S = S_prev + mu * dt + sigma * (qconst * (U - U_prev) + pconst * dZ)

        pi_pred = pi @ trans
        fad_j = gamma * sigma * qconst * theta_np[None, :]
        ell_a_j = np.maximum(varphi + psi * np.exp(-fad_j), 1e-12)
        ell_b_j = np.maximum(varphi + psi * np.exp(+fad_j), 1e-12)
        log_like = dMa[:, None] * np.log(ell_a_j * dt) - ell_a_j * dt + dMb[:, None] * np.log(ell_b_j * dt) - ell_b_j * dt
        log_like -= np.max(log_like, axis=1, keepdims=True)
        weights = pi_pred * np.exp(log_like)
        denom = weights.sum(axis=1, keepdims=True)
        bad = denom[:, 0] <= 1e-14
        pi = weights / np.maximum(denom, 1e-14)
        if np.any(bad):
            pi[bad, :] = pi0

        if it in {0, Nt // 2, Nt - 1}:
            mean_rho_a.append(float(np.mean(rho_a))); mean_rho_b.append(float(np.mean(rho_b))); mean_Q.append(float(np.mean(Q)))
            mean_Ma.append(float(np.mean(dMa))); mean_Mb.append(float(np.mean(dMb)))
            mean_Na.append(float(np.mean(dNa))); mean_Nb.append(float(np.mean(dNb)))

    perf = X + Q * S - alpha * Q ** 2 - phi_inv * int_Q2
    m, sd = mean_sd(perf)
    diag = {"mean_final_Q": float(np.mean(Q)), "sd_final_Q": float(np.std(Q, ddof=1)),
            "mean_quote_a_start_mid_end": mean_rho_a, "mean_quote_b_start_mid_end": mean_rho_b,
            "mean_Q_start_mid_end": mean_Q,
            "mean_full_arrivals_a_start_mid_end": mean_Ma, "mean_full_arrivals_b_start_mid_end": mean_Mb,
            "mean_fills_a_start_mid_end": mean_Na, "mean_fills_b_start_mid_end": mean_Nb}
    return m, sd, diag




def simulate_constant_quote_same_framework(mm_aux, sims: int, Nt: int, seed: int,
                                           delta_const=None,
                                           verbose: bool = True):
    """Simulate constant quote under the same market/performance criterion as ARR-full-NN."""
    rng = np.random.default_rng(seed)

    T = float(getattr(mm_aux, "T", CLEAN_BASE_PARAMS_ENV.get("T", 1.0)))
    dt = T / Nt
    k = float(getattr(mm_aux, "k", CLEAN_BASE_PARAMS_ENV.get("k", 1.0)))
    phi_inv = float(getattr(mm_aux, "phi", CLEAN_BASE_PARAMS_ENV.get("phi", 0.1)))
    alpha = float(getattr(mm_aux, "alpha", CLEAN_BASE_PARAMS_ENV.get("alpha", 0.001)))
    q_bar = int(getattr(mm_aux, "q_bar", 50))
    Q_min, Q_max = -q_bar, q_bar

    sigma = float(getattr(mm_aux, "sigma", 1.0))
    eta = float(getattr(mm_aux, "eta", 10.0))
    qconst = float(getattr(mm_aux, "qconst"))
    pconst = math.sqrt(max(1.0 - qconst ** 2, 0.0))
    mu = float(getattr(mm_aux, "mu", 0.0))
    S0 = float(getattr(mm_aux, "S_0", 0.0))
    U0 = float(getattr(mm_aux, "U_0", 0.0))
    Q0 = float(getattr(mm_aux, "Q_0", 0.0))
    X0 = float(getattr(mm_aux, "X_0", 0.0))
    varphi = float(getattr(mm_aux, "varphi", 15.0))
    psi = float(getattr(mm_aux, "psi"))
    gamma = float(getattr(mm_aux, "gamma"))

    if delta_const is None:
        delta_const = 1.0 / k

    S = np.full(sims, S0, dtype=float)
    U = np.full(sims, U0, dtype=float)
    Q = np.full(sims, Q0, dtype=float)
    X = np.full(sims, X0, dtype=float)
    int_Q2 = np.zeros(sims, dtype=float)

    mean_rho_a, mean_rho_b, mean_Q = [], [], []
    mean_Ma, mean_Mb, mean_Na, mean_Nb = [], [], [], []
    iterator = tqdm(range(Nt), desc="Constant quote simulation") if verbose else range(Nt)

    for it in iterator:
        S_prev = S.copy()
        U_prev = U.copy()

        rho_a = np.full(sims, float(delta_const), dtype=float)
        rho_b = np.full(sims, float(delta_const), dtype=float)

        allow_a = (Q > Q_min).astype(float)
        allow_b = (Q < Q_max).astype(float)

        fad_true = gamma * sigma * qconst * U
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
        S = S_prev + mu * dt + sigma * (qconst * (U - U_prev) + pconst * dZ)

        if it in {0, Nt // 2, Nt - 1}:
            mean_rho_a.append(float(np.mean(rho_a)))
            mean_rho_b.append(float(np.mean(rho_b)))
            mean_Q.append(float(np.mean(Q)))
            mean_Ma.append(float(np.mean(dMa)))
            mean_Mb.append(float(np.mean(dMb)))
            mean_Na.append(float(np.mean(dNa)))
            mean_Nb.append(float(np.mean(dNb)))

    perf = X + Q * S - alpha * Q ** 2 - phi_inv * int_Q2
    m, sd = mean_sd(perf)
    diag = {
        "delta_const": float(delta_const),
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
    return m, sd, diag

def apply_preset(args):
    if args.preset == "smoke":
        args.epochs = 2
        args.batch_size = 32
        args.sims_nn = 2
        args.Nt_arr = 2
        args.quote_points = 11
        args.width = 48
        args.depth = 2
    elif args.preset == "quick":
        args.epochs = 4000
        args.batch_size = 512
        args.sims_nn = 10000
        args.Nt_arr = 1000
        args.quote_points = 41
        args.width = 128
        args.depth = 3
    elif args.preset == "final":
        args.epochs = 12000
        args.batch_size = 768
        args.sims_nn = 20000
        args.Nt_arr = 1000
        args.quote_points = 51
        args.width = 160
        args.depth = 4


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--q", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)

    parser.add_argument("--model_q", type=float, default=None)
    parser.add_argument("--model_gamma", type=float, default=None)
    parser.add_argument("--model_root", type=str, default=None)

    parser.add_argument("--out_root", type=str, default="outputs_clean/arr_nn_common_fiplug_v6")
    parser.add_argument("--chapter4_outputs", type=str, default="outputs_clean")
    parser.add_argument("--seed", type=int, default=36457656)

    parser.add_argument("--preset", type=str, default="quick", choices=["smoke", "quick", "final"])
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--matmul_precision", type=str, default="high", choices=["highest", "high", "medium"])

    parser.add_argument("--J", type=int, default=7)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--phi_inventory", type=float, default=0.1)
    parser.add_argument("--varphi", type=float, default=15.0)
    parser.add_argument("--k", type=float, default=1.0)

    parser.add_argument("--q_min", type=int, default=-50)
    parser.add_argument("--q_max", type=int, default=50)
    parser.add_argument("--focus_width", type=int, default=15)
    parser.add_argument("--focus_prob", type=float, default=0.85)

    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--lambda_term", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)

    parser.add_argument("--quote_min", type=float, default=0.01)
    parser.add_argument("--quote_max", type=float, default=3.0)
    parser.add_argument("--quote_points", type=int, default=41)

    parser.add_argument("--sims_nn", type=int, default=10000)
    parser.add_argument("--Nt_arr", type=int, default=1000)
    parser.add_argument("--batch_paths", type=int, default=512)
    parser.add_argument("--print_every", type=int, default=200)

    return parser


def main():
    args = build_parser().parse_args()
    apply_preset(args)

    set_seed(args.seed)

    device = choose_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision(args.matmul_precision)
        print("Using device: cuda")
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Using device: cpu")

    out_root = Path(args.out_root)
    pdir = out_root / "points" / point_name(args.q, args.gamma)
    pdir.mkdir(parents=True, exist_ok=True)

    model_q = float(args.q if args.model_q is None else args.model_q)
    model_gamma = float(args.gamma if args.model_gamma is None else args.model_gamma)
    model_root = Path(args.out_root if args.model_root is None else args.model_root)
    model_dir = model_root / "points" / point_name(model_q, model_gamma) / "model"

    print(f"Testing environment: q={args.q}, gamma={args.gamma}")
    print(f"Loaded NN model:       q={model_q}, gamma={model_gamma}")
    print(f"Model directory:      {model_dir}")

    # Get calibrated psi from the testing environment. This matches the paper-style calibration.
    mm_aux = build_mm_aux(args.q, args.gamma, seed=args.seed)
    psi = float(getattr(mm_aux, "psi"))
    print(f"Calibrated psi from original environment: {psi:.12g}")

    # In misspecification mode, we only evaluate an existing trained model.
    if not (model_dir / "value_net.pt").exists():
        raise FileNotFoundError(f"Could not find trained model at {model_dir / 'value_net.pt'}")

    print(f"Loading existing model in {model_dir}")
    training_metrics = json.loads((model_dir / "training_metrics.json").read_text()) if (model_dir / "training_metrics.json").exists() else {}

    # Simulate ARR-full-NN under the same market/performance criterion.
    arr_nn_mean, arr_nn_sd, diag = simulate_arr_nn(
        mm_aux=mm_aux,
        model_dir=model_dir,
        sims=args.sims_nn,
        Nt=args.Nt_arr,
        seed=args.seed + 991,
        device=device,
        batch_paths=args.batch_paths,
        verbose=True,
    )

    # Simulate a constant-quote lower benchmark under the same market/performance criterion.
    const_mean, const_sd, const_diag = simulate_constant_quote_same_framework(
        mm_aux=mm_aux,
        sims=args.sims_nn,
        Nt=args.Nt_arr,
        seed=args.seed + 991,
        delta_const=None,
        verbose=True,
    )

    # Load PI-FIplug / ARR-full-FIplug from completed Chapter 4 outputs.
    common = load_common_fiplug_point(Path(args.chapter4_outputs), args.q, args.gamma)

    result = {
        "q": float(args.q),
        "gamma": float(args.gamma),
        "model_q": float(model_q),
        "model_gamma": float(model_gamma),
        "misspecification_eval": True,
        "psi": psi,
        "varphi": float(args.varphi),
        "eta": float(args.eta),
        "sigma": float(args.sigma),
        "T": float(args.T),
        "J": int(args.J),
        "sims_nn": int(args.sims_nn),
        "Nt_arr": int(args.Nt_arr),
        "PI_FIplug_mean": common["PI_FIplug_mean"],
        "PI_FIplug_sd": common["PI_FIplug_sd"],
        "ARR_full_FIplug_mean": common["ARR_full_FIplug_mean"],
        "ARR_full_FIplug_sd": common["ARR_full_FIplug_sd"],
        "ARR_full_minus_PI_FIplug": common["ARR_full_minus_PI_FIplug"],
        "common_fiplug_source": common["source"],
        "Constant_quote_mean": const_mean,
        "Constant_quote_sd": const_sd,
        "Constant_quote_SE": const_sd / math.sqrt(float(args.sims_nn)),
        "Constant_quote_delta": 1.0 / float(getattr(mm_aux, "k", CLEAN_BASE_PARAMS_ENV.get("k", 1.0))),
        "constant_quote_diagnostics": const_diag,
        "ARR_full_NN_mean": arr_nn_mean,
        "ARR_full_NN_sd": arr_nn_sd,
        "ARR_full_NN_minus_constant_quote": arr_nn_mean - const_mean,
        "ARR_full_NN_minus_PI_FIplug": arr_nn_mean - common["PI_FIplug_mean"] if np.isfinite(common["PI_FIplug_mean"]) else float("nan"),
        "ARR_full_NN_minus_ARR_full_FIplug": arr_nn_mean - common["ARR_full_FIplug_mean"] if np.isfinite(common["ARR_full_FIplug_mean"]) else float("nan"),
        "training_metrics": training_metrics,
        "arr_nn_diagnostics": diag,
        "model_dir": str(model_dir),
    }

    (pdir / "arr_nn_common_fiplug_result.json").write_text(json.dumps(result, indent=2))

    with open(pdir / "arr_nn_common_fiplug_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "mean", "sd"])
        writer.writerow(["Constant quote", result["Constant_quote_mean"], result["Constant_quote_sd"]])
        writer.writerow(["PI-FIplug", result["PI_FIplug_mean"], result["PI_FIplug_sd"]])
        writer.writerow(["ARR-full-FIplug", result["ARR_full_FIplug_mean"], result["ARR_full_FIplug_sd"]])
        writer.writerow(["ARR-full-NN", result["ARR_full_NN_mean"], result["ARR_full_NN_sd"]])

    # Small plot for this point.
    labels = ["PI-FIplug", "ARR-full-FIplug", "ARR-full-NN"]
    means = [result["PI_FIplug_mean"], result["ARR_full_FIplug_mean"], result["ARR_full_NN_mean"]]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, means)
    plt.ylabel("average performance")
    plt.title(f"Common comparison, q={args.q}, gamma={args.gamma}")
    plt.tight_layout()
    plt.savefig(pdir / "point_comparison_bar.png", dpi=250, bbox_inches="tight")
    plt.close()

    print("\nResult:")
    print(json.dumps(result, indent=2))
    print("\nSaved to:", pdir)


if __name__ == "__main__":
    main()
