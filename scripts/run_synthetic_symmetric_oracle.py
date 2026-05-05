from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ssrk import SSRKModel
from ssrk.knockoff_generators import generate_oracle_knockoffs_latent_factor
from ssrk.statistics import compute_fdp_power, knockoff_plus_filter


@dataclass
class SyntheticConfig:
    n_samples: int = 5000
    p_features: int = 100
    s_signals: int = 50
    q: float = 0.10
    signal_strength: float = 5.0
    noise_std: float = 0.2
    latent_dim: int = 5
    epochs: int = 180
    batch_size: int = 128
    lr: float = 3e-4
    lr_gate_factor: float = 0.3
    lambda_entropy: float = 0.001
    stage1_frac: float = 0.2
    mask_prob: float = 0.5
    encoder_dims: tuple[int, ...] = (256, 128)
    model_latent_dim: int = 32
    decoder_dims: tuple[int, ...] = (128, 256)
    temperature: float = 1.0
    use_batchnorm: bool = True


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass


def binary_entropy(pi: torch.Tensor) -> torch.Tensor:
    eps = 1e-7
    return (-pi * torch.log(pi + eps) - (1.0 - pi) * torch.log(1.0 - pi + eps)).sum()


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sum(F.mse_loss(pred, target, reduction="none") * mask) / (torch.sum(mask) + 1e-9)


def build_model(cfg: SyntheticConfig) -> SSRKModel:
    return SSRKModel(
        p_features=cfg.p_features,
        encoder_dims=list(cfg.encoder_dims),
        latent_dim=cfg.model_latent_dim,
        decoder_dims=list(cfg.decoder_dims),
        temperature=cfg.temperature,
        use_batchnorm=cfg.use_batchnorm,
    )


def train_two_order_symmetric(
    model: SSRKModel,
    X: np.ndarray,
    X_tilde: np.ndarray,
    cfg: SyntheticConfig,
    device: torch.device,
) -> float:
    model = model.to(device)
    gate_params = [model.gating_layer.gate_logits]
    network_params = [p for name, p in model.named_parameters() if "gate_logits" not in name]
    optimizer = torch.optim.Adam(
        [
            {"params": network_params, "lr": cfg.lr},
            {"params": gate_params, "lr": cfg.lr * cfg.lr_gate_factor},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(X_tilde, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    stage1_epochs = max(1, int(cfg.epochs * cfg.stage1_frac))
    entropy_max = cfg.p_features * math.log(2.0)
    start = time.time()

    model.train()
    for epoch in range(cfg.epochs):
        stage1 = epoch < stage1_epochs
        optimizer.param_groups[1]["lr"] = 0.0 if stage1 else cfg.lr * cfg.lr_gate_factor
        for X_batch, Xk_batch in loader:
            X_batch = X_batch.to(device)
            Xk_batch = Xk_batch.to(device)
            mask = torch.bernoulli(torch.full(X_batch.shape, cfg.mask_prob, device=device))

            optimizer.zero_grad()
            pred_first = model(X_batch, Xk_batch, mask)
            pred_second = model(Xk_batch, X_batch, mask)
            loss_rec = 0.5 * masked_mse(pred_first, X_batch, mask) + 0.5 * masked_mse(
                pred_second, Xk_batch, mask
            )
            h = binary_entropy(model.gating_layer.get_gate_probabilities())
            loss_reg = (entropy_max - h) if stage1 else h
            loss = loss_rec + cfg.lambda_entropy * loss_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

    return float(time.time() - start)


def run_replicate(name: str, p_features: int, s_signals: int, seed: int, args: argparse.Namespace) -> dict[str, Any]:
    cfg = SyntheticConfig(
        n_samples=args.n_samples,
        p_features=p_features,
        s_signals=s_signals,
        q=args.q,
        signal_strength=args.signal_strength,
        noise_std=args.noise_std,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_gate_factor=args.lr_gate_factor,
        lambda_entropy=args.lambda_entropy,
        stage1_frac=args.stage1_frac,
        mask_prob=args.mask_prob,
    )
    set_seed(seed)
    X, X_tilde, true_support = generate_oracle_knockoffs_latent_factor(
        n_samples=cfg.n_samples,
        p_features=cfg.p_features,
        s_signals=cfg.s_signals,
        latent_dim=cfg.latent_dim,
        signal_strength=cfg.signal_strength,
        noise_std=cfg.noise_std,
        seed=seed,
    )
    set_seed(seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg)
    train_time = train_two_order_symmetric(model, X, X_tilde, cfg, device)
    W = np.asarray(model.get_W_statistics(), dtype=float)
    threshold, selected = knockoff_plus_filter(W, cfg.q)
    fdp, power = compute_fdp_power(selected, true_support, cfg.p_features)

    signal_idx = np.array(sorted(true_support), dtype=int)
    null_idx = np.array([j for j in range(cfg.p_features) if j not in true_support], dtype=int)
    signal_w = W[signal_idx]
    null_w = W[null_idx]
    return {
        "condition": name,
        "seed": int(seed),
        "fdp": float(fdp),
        "power": float(power),
        "n_selected": int(len(selected)),
        "threshold": None if math.isinf(float(threshold)) else float(threshold),
        "w_signal_mean": float(signal_w.mean()),
        "w_signal_min": float(signal_w.min()),
        "w_signal_max": float(signal_w.max()),
        "w_null_mean": float(null_w.mean()),
        "w_null_max": float(null_w.max()),
        "num_w_pos": int(np.sum(W > 0)),
        "num_w_neg": int(np.sum(W < 0)),
        "train_time_seconds": train_time,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in ["fdp", "power", "n_selected"]:
        vals = np.asarray([float(r[key]) for r in rows], dtype=float)
        out[f"{key}_mean"] = float(vals.mean())
        out[f"{key}_std"] = float(vals.std(ddof=0))
        out[f"{key}_min"] = float(vals.min())
        out[f"{key}_max"] = float(vals.max())
    out["nonempty_rate"] = float(np.mean([int(r["n_selected"]) > 0 for r in rows]))
    out["train_time_seconds_mean"] = float(np.mean([float(r["train_time_seconds"]) for r in rows]))
    return out


def parse_condition(text: str) -> tuple[str, int, int]:
    parts = text.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("condition must be name:p:s, e.g. p100_s50_q10:100:50")
    return parts[0], int(parts[1]), int(parts[2])


def write_csv(path: Path, payload: dict[str, Any]) -> None:
    rows = [row for condition in payload["conditions"] for row in condition["rows"]]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SSRK synthetic oracle demo.")
    parser.add_argument("--out", default="results/synthetic_symmetric_oracle_demo.json")
    parser.add_argument("--csv-out", default="results/synthetic_symmetric_oracle_demo.csv")
    parser.add_argument("--condition", action="append", type=parse_condition)
    parser.add_argument("--seeds", default="0,100,200,300,400,500,600,700,800,900")
    parser.add_argument("--device", default="")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--q", type=float, default=0.10)
    parser.add_argument("--signal-strength", type=float, default=5.0)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--latent-dim", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-gate-factor", type=float, default=0.3)
    parser.add_argument("--lambda-entropy", type=float, default=0.001)
    parser.add_argument("--stage1-frac", type=float, default=0.2)
    parser.add_argument("--mask-prob", type=float, default=0.5)
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(max(1, args.num_threads))

    conditions = args.condition or [
        ("p100_s50_q10_lam001", 100, 50),
        ("p200_s60_q10_lam001", 200, 60),
        ("p300_s90_q10_lam001", 300, 90),
        ("p500_s150_q10_lam001", 500, 150),
    ]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    payload: dict[str, Any] = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": "scripts/run_synthetic_symmetric_oracle.py",
        "objective": "two-order first-slot masked reconstruction average",
        "repro_command": (
            "python scripts/run_synthetic_symmetric_oracle.py --device cpu "
            "--out results/synthetic_symmetric_oracle_demo.json "
            "--csv-out results/synthetic_symmetric_oracle_demo.csv"
        ),
        "seeds": seeds,
        "conditions": [],
    }

    out_path = Path(args.out)
    csv_path = Path(args.csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    for name, p_features, s_signals in conditions:
        print(f"RUN {name} p={p_features} s={s_signals}", flush=True)
        rows = [run_replicate(name, p_features, s_signals, seed, args) for seed in seeds]
        payload["conditions"].append(
            {
                "name": name,
                "config": {
                    "n_samples": args.n_samples,
                    "p_features": p_features,
                    "s_signals": s_signals,
                    "q": args.q,
                    "signal_strength": args.signal_strength,
                    "noise_std": args.noise_std,
                    "latent_dim": args.latent_dim,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "lr_gate_factor": args.lr_gate_factor,
                    "lambda_entropy": args.lambda_entropy,
                    "stage1_frac": args.stage1_frac,
                    "mask_prob": args.mask_prob,
                    "num_threads": args.num_threads,
                },
                "summary": summarize(rows),
                "rows": rows,
            }
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        write_csv(csv_path, payload)

    print(f"WROTE {out_path}", flush=True)
    print(f"WROTE {csv_path}", flush=True)


if __name__ == "__main__":
    main()
