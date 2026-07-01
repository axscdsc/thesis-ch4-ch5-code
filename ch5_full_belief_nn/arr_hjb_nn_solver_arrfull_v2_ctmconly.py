import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ARRModelParams:
    def __init__(
        self,
        T=1.0,
        mu=0.0,
        sigma=1.0,
        eta=10.0,
        alpha=0.001,
        phi_inventory=0.1,
        varphi_uninformed=15.0,
        psi_informed=0.1,
        k=1.0,
        q_loading=0.6,
        gamma=1.0,
        q_min=-5,
        q_max=5,
    ):
        self.T = float(T)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.phi_inventory = float(phi_inventory)
        self.varphi = float(varphi_uninformed)
        self.psi = float(psi_informed)
        self.k = float(k)
        self.q_loading = float(q_loading)
        self.gamma = float(gamma)
        self.q_min = int(q_min)
        self.q_max = int(q_max)


def make_ou_grid(J: int, eta: float, device: torch.device):
    if J == 1:
        theta = torch.tensor([0.0], dtype=torch.float32, device=device)
        L = torch.zeros((1, 1), dtype=torch.float32, device=device)
        return theta, L

    std_u = math.sqrt(1.0 / (2.0 * eta))
    theta_np = np.linspace(-3.0 * std_u, 3.0 * std_u, J).astype(np.float32)
    theta = torch.tensor(theta_np, dtype=torch.float32, device=device)
    L = make_ou_generator(theta, eta)
    return theta, L


def make_ou_generator(theta: torch.Tensor, eta: float):
    J = theta.numel()
    device = theta.device
    L = torch.zeros((J, J), dtype=torch.float32, device=device)

    if J == 1:
        return L

    h = float(theta[1] - theta[0])
    for i in range(J):
        x = float(theta[i])
        drift = -eta * x
        rate_up = 0.5 / (h * h) + max(drift, 0.0) / h
        rate_dn = 0.5 / (h * h) + max(-drift, 0.0) / h

        if i == 0:
            L[i, i + 1] = rate_up + rate_dn
        elif i == J - 1:
            L[i, i - 1] = rate_up + rate_dn
        else:
            L[i, i + 1] = rate_up
            L[i, i - 1] = rate_dn

    L[torch.arange(J, device=device), torch.arange(J, device=device)] = -L.sum(dim=1)
    return L


def sample_beliefs(batch_size: int, J: int, device: torch.device, alpha: float = 1.0):
    dist = torch.distributions.Dirichlet(torch.full((J,), alpha, device=device))
    return dist.sample((batch_size,))


def normalise_Q(Q: torch.Tensor, q_min: int, q_max: int):
    mid = 0.5 * (q_min + q_max)
    half = 0.5 * (q_max - q_min)
    return (Q - mid) / half


def state_intensity_coefficients(theta: torch.Tensor, p: ARRModelParams):
    c = p.gamma * p.sigma * p.q_loading * theta
    ell_a = p.varphi + p.psi * torch.exp(-c)
    ell_b = p.varphi + p.psi * torch.exp(+c)
    return ell_a, ell_b


def conditional_mean(pi: torch.Tensor, theta: torch.Tensor):
    return (pi * theta.view(1, -1)).sum(dim=1)


class ValueNet(nn.Module):
    def __init__(self, input_dim: int, width: int = 128, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(input_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, Q: torch.Tensor, pi: torch.Tensor, p: ARRModelParams):
        if t.dim() == 1:
            t = t[:, None]
        if Q.dim() == 1:
            Q = Q[:, None]
        t_scaled = t / p.T
        Q_scaled = normalise_Q(Q, p.q_min, p.q_max)
        x = torch.cat([t_scaled, Q_scaled, pi], dim=1)
        raw = self.net(x).squeeze(-1)
        base_terminal = -p.alpha * Q.squeeze(-1) ** 2
        return base_terminal + (p.T - t.squeeze(-1)) * raw

def hjb_residual(
    model: ValueNet,
    t: torch.Tensor,
    Q: torch.Tensor,
    pi: torch.Tensor,
    theta: torch.Tensor,
    L: torch.Tensor,
    p: ARRModelParams,
    quote_grid: torch.Tensor,
):
    """
    HJB residual for the ARR-full control problem.

    The filter observes full market-order arrivals M^a,M^b. Their intensities
    ell^{a,b,j} do not depend on the quote controls. Quotes only enter through
    fill probabilities exp(-k delta) after an arrival has occurred.
    """
    B = t.shape[0]
    J = pi.shape[1]

    t_req = t.clone().detach().requires_grad_(True)
    pi_req = pi.clone().detach().requires_grad_(True)
    Q_const = Q.clone().detach()

    V = model(t_req, Q_const, pi_req, p)

    dV_dt = torch.autograd.grad(V.sum(), t_req, create_graph=True, retain_graph=True)[0]
    dV_dpi = torch.autograd.grad(V.sum(), pi_req, create_graph=True, retain_graph=True)[0]

    m_pi = conditional_mean(pi_req, theta)
    running = -p.phi_inventory * Q_const**2 + Q_const * (
        p.mu - p.eta * p.sigma * p.q_loading * m_pi
    )

    ell_a, ell_b = state_intensity_coefficients(theta, p)
    ell_a_b = ell_a.view(1, J)
    ell_b_b = ell_b.view(1, J)

    # Full-arrival predicted intensities, independent of quote controls.
    hat_a = (pi_req * ell_a_b).sum(dim=1)
    hat_b = (pi_req * ell_b_b).sum(dim=1)

    eps = 1e-10
    Gamma_a = (pi_req * ell_a_b) / (hat_a[:, None] + eps)
    Gamma_b = (pi_req * ell_b_b) / (hat_b[:, None] + eps)
    Gamma_a = torch.where(hat_a[:, None] > eps, Gamma_a, pi_req)
    Gamma_b = torch.where(hat_b[:, None] > eps, Gamma_b, pi_req)

    # CTMC prediction term only.
    # The arrival compensator effect is represented through the Bayes jump
    # terms below. Including pi*(hat_lambda-lambda) here as well would
    # double-count the same first-order effect.
    pred = pi_req @ L
    belief_drift_term = (pred * dV_dpi).sum(dim=1)

    da_vals, db_vals = torch.meshgrid(quote_grid, quote_grid, indexing="ij")
    da_flat = da_vals.reshape(-1)
    db_flat = db_vals.reshape(-1)
    M = da_flat.numel()

    ind_a = (Q_const > p.q_min).float()
    ind_b = (Q_const < p.q_max).float()

    da = da_flat.view(M, 1)
    db = db_flat.view(M, 1)

    pfill_a = torch.clamp(torch.exp(-p.k * da) * ind_a.view(1, B), 0.0, 1.0)
    pfill_b = torch.clamp(torch.exp(-p.k * db) * ind_b.view(1, B), 0.0, 1.0)

    t_rep = t_req.unsqueeze(0).expand(M, B).reshape(-1)
    Q_rep = Q_const.unsqueeze(0).expand(M, B).reshape(-1)
    Q_a = (Q_const - 1.0).unsqueeze(0).expand(M, B).reshape(-1)
    Q_b = (Q_const + 1.0).unsqueeze(0).expand(M, B).reshape(-1)

    Gamma_a_rep = Gamma_a.unsqueeze(0).expand(M, B, J).reshape(M * B, J)
    Gamma_b_rep = Gamma_b.unsqueeze(0).expand(M, B, J).reshape(M * B, J)

    V_a_fill = model(t_rep, Q_a, Gamma_a_rep, p).reshape(M, B)
    V_b_fill = model(t_rep, Q_b, Gamma_b_rep, p).reshape(M, B)
    V_a_nofill = model(t_rep, Q_rep, Gamma_a_rep, p).reshape(M, B)
    V_b_nofill = model(t_rep, Q_rep, Gamma_b_rep, p).reshape(M, B)

    V_current = V.unsqueeze(0)

    ask_arrival = hat_a.view(1, B) * (
        pfill_a * (da + V_a_fill - V_current)
        + (1.0 - pfill_a) * (V_a_nofill - V_current)
    )
    bid_arrival = hat_b.view(1, B) * (
        pfill_b * (db + V_b_fill - V_current)
        + (1.0 - pfill_b) * (V_b_nofill - V_current)
    )

    H = (
        dV_dt.unsqueeze(0)
        + belief_drift_term.unsqueeze(0)
        + running.unsqueeze(0)
        + ask_arrival
        + bid_arrival
    )
    R, argmax_idx = H.max(dim=0)

    return R, argmax_idx, da_flat, db_flat

def terminal_loss(model: ValueNet, batch_size: int, J: int, device: torch.device, p: ARRModelParams):
    pi = sample_beliefs(batch_size, J, device)
    Q_int = torch.randint(p.q_min, p.q_max + 1, (batch_size,), device=device).float()
    t = torch.full((batch_size,), p.T, device=device)
    V_T = model(t, Q_int, pi, p)
    target = -p.alpha * Q_int**2
    return ((V_T - target) ** 2).mean()


def sample_inventory_mixture(
    batch_size: int,
    device: torch.device,
    p: ARRModelParams,
    focus_width: int = 15,
    focus_prob: float = 0.8,
):

    focus_width = int(min(focus_width, max(abs(p.q_min), abs(p.q_max))))
    use_focus = torch.rand(batch_size, device=device) < float(focus_prob)

    Q_full = torch.randint(p.q_min, p.q_max + 1, (batch_size,), device=device).float()
    Q_focus = torch.randint(-focus_width, focus_width + 1, (batch_size,), device=device).float()

    return torch.where(use_focus, Q_focus, Q_full)


def sample_interior_batch(
    batch_size: int,
    J: int,
    device: torch.device,
    p: ARRModelParams,
    focus_width: int = 15,
    focus_prob: float = 0.8,
):
    t = torch.rand(batch_size, device=device) * p.T
    Q = sample_inventory_mixture(batch_size, device, p, focus_width, focus_prob)
    pi = sample_beliefs(batch_size, J, device)
    return t, Q, pi

def compute_quotes_for_states(model, theta, L, p, quote_grid, device, states):
    rows = []
    J = theta.numel()
    for (t_val, Q_val, pi_np) in states:
        t = torch.tensor([float(t_val)], device=device)
        Q = torch.tensor([float(Q_val)], device=device)
        pi = torch.tensor(pi_np, dtype=torch.float32, device=device).view(1, J)
        R, idx, da_flat, db_flat = hjb_residual(model, t, Q, pi, theta, L, p, quote_grid)
        k = int(idx.item())
        rows.append({
            "t": float(t_val),
            "Q": float(Q_val),
            "residual": float(R.detach().cpu().item()),
            "delta_a_star": float(da_flat[k].detach().cpu().item()),
            "delta_b_star": float(db_flat[k].detach().cpu().item()),
        })
    return rows


def save_loss_plot(loss_history, outdir: Path):
    fig = plt.figure(figsize=(7, 4))
    plt.plot(loss_history)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.title("Neural HJB training loss")
    plt.tight_layout()
    fig.savefig(outdir / "training_loss.png", dpi=200)
    plt.close(fig)


def apply_preset(args):
    # The presets overwrite the defaults for a direct final run.
    if args.preset == "final_h100":
        args.epochs = 15000
        args.batch_size = 1024
        args.quote_points = 51
        args.lambda_term = 80.0
        args.lr = 3e-4
        args.width = 192
        args.depth = 4
        args.outdir = "outputs/arr_hjb_nn_colab_h100_final"
    elif args.preset == "final_a100":
        args.epochs = 12000
        args.batch_size = 768
        args.quote_points = 41
        args.lambda_term = 80.0
        args.lr = 4e-4
        args.width = 160
        args.depth = 4
        args.outdir = "outputs/arr_hjb_nn_colab_a100_final"
    elif args.preset == "qbar50_v2":
        args.epochs = 20000
        args.batch_size = 768
        args.quote_points = 51
        args.lambda_term = 10.0
        args.lr = 2e-4
        args.width = 256
        args.depth = 4
        args.q_min = -50
        args.q_max = 50
        args.focus_width = 15
        args.focus_prob = 0.85
        args.outdir = "outputs/arr_hjb_nn_colab_qbar50_v2"
    elif args.preset == "safe":
        args.epochs = 8000
        args.batch_size = 512
        args.quote_points = 41
        args.lambda_term = 50.0
        args.lr = 5e-4
        args.width = 128
        args.depth = 3
        args.outdir = "outputs/arr_hjb_nn_colab_safe"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default="safe", choices=["safe", "final_h100", "final_a100", "qbar50_v2"])
    parser.add_argument("--outdir", type=str, default="outputs/arr_hjb_nn_colab_final")
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--J", type=int, default=7)
    parser.add_argument("--q", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--psi", type=float, default=0.1)
    parser.add_argument("--varphi", type=float, default=15.0)

    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--phi_inventory", type=float, default=0.1)
    parser.add_argument("--k", type=float, default=1.0)

    parser.add_argument("--q_min", type=int, default=-5)
    parser.add_argument("--q_max", type=int, default=5)
    parser.add_argument("--focus_width", type=int, default=15,
                        help="Central inventory range [-focus_width,focus_width] used by mixture sampling.")
    parser.add_argument("--focus_prob", type=float, default=0.8,
                        help="Probability of sampling Q from the central inventory region.")

    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lambda_term", type=float, default=50.0)

    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)

    parser.add_argument("--quote_min", type=float, default=0.01)
    parser.add_argument("--quote_max", type=float, default=3.0)
    parser.add_argument("--quote_points", type=int, default=41)

    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--matmul_precision", type=str, default="high", choices=["highest", "high", "medium"])
    parser.add_argument("--no_preset", action="store_true", help="Use explicitly passed values instead of preset values.")

    args = parser.parse_args()
    if not args.no_preset:
        apply_preset(args)

    set_seed(args.seed)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("You requested --device cuda, but torch.cuda.is_available() is False.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision(args.matmul_precision)
        print("Using device: cuda")
        print("GPU name:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
    else:
        print("Using device: cpu")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    p = ARRModelParams(
        T=args.T,
        mu=args.mu,
        sigma=args.sigma,
        eta=args.eta,
        alpha=args.alpha,
        phi_inventory=args.phi_inventory,
        varphi_uninformed=args.varphi,
        psi_informed=args.psi,
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

    config_to_save = vars(args).copy()
    config_to_save["actual_device"] = str(device)
    if device.type == "cuda":
        config_to_save["gpu_name"] = torch.cuda.get_device_name(0)
        config_to_save["cuda_version"] = torch.version.cuda

    with open(outdir / "config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)

    loss_history = []
    residual_history = []
    terminal_history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()

        t, Q, pi = sample_interior_batch(args.batch_size, args.J, device, p, args.focus_width, args.focus_prob)
        R, _, _, _ = hjb_residual(model, t, Q, pi, theta, L, p, quote_grid)
        loss_hjb = (R ** 2).mean()
        loss_term = terminal_loss(model, args.batch_size, args.J, device, p)
        loss = loss_hjb + args.lambda_term * loss_term

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        opt.step()

        loss_history.append(float(loss.detach().cpu().item()))
        residual_history.append(float(loss_hjb.detach().cpu().item()))
        terminal_history.append(float(loss_term.detach().cpu().item()))

        if epoch % args.print_every == 0 or epoch == 1:
            print(
                f"epoch {epoch:6d} | "
                f"loss={loss_history[-1]:.6e} | "
                f"HJB={residual_history[-1]:.6e} | "
                f"term={terminal_history[-1]:.6e}"
            )

    torch.save(model.state_dict(), outdir / "value_net.pt")

    hist = {
        "loss": loss_history,
        "hjb_loss": residual_history,
        "terminal_loss": terminal_history,
    }
    with open(outdir / "loss_history.json", "w") as f:
        json.dump(hist, f)

    save_loss_plot(loss_history, outdir)

    model.eval()
    with torch.no_grad():
        pi_val = sample_beliefs(2048, args.J, device)
        Q_val = torch.randint(p.q_min, p.q_max + 1, (2048,), device=device).float()
        t_T = torch.full((2048,), p.T, device=device)
        V_T = model(t_T, Q_val, pi_val, p)
        target_T = -p.alpha * Q_val**2
        term_mse = ((V_T - target_T) ** 2).mean().item()
        term_mae = (V_T - target_T).abs().mean().item()

    pi_uniform = np.ones(args.J, dtype=np.float32) / args.J
    states = [
        (0.25 * p.T, -10, pi_uniform),
        (0.25 * p.T, -2, pi_uniform),
        (0.25 * p.T, 0, pi_uniform),
        (0.25 * p.T, 2, pi_uniform),
        (0.25 * p.T, 10, pi_uniform),
        (0.75 * p.T, -10, pi_uniform),
        (0.75 * p.T, -2, pi_uniform),
        (0.75 * p.T, 0, pi_uniform),
        (0.75 * p.T, 2, pi_uniform),
        (0.75 * p.T, 10, pi_uniform),
    ]
    quote_rows = compute_quotes_for_states(model, theta, L, p, quote_grid, device, states)

    with open(outdir / "quote_sanity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "Q", "residual", "delta_a_star", "delta_b_star"])
        writer.writeheader()
        writer.writerows(quote_rows)

    metrics = {
        "terminal_mse": term_mse,
        "terminal_mae": term_mae,
        "final_loss": loss_history[-1],
        "final_hjb_loss": residual_history[-1],
        "final_terminal_loss": terminal_history[-1],
        "quote_sanity": quote_rows,
    }
    with open(outdir / "validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nValidation metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nOutputs saved to:", outdir)


if __name__ == "__main__":
    main()
