#!/usr/bin/env python3
"""Reusable inverse-dynamics readout for frozen latent transitions."""

from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ACTION_NAMES = ["action_0", "action_1", "action_2", "action_3", "action_4"]
EPS = 1e-12


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, min_std: float = 1e-6) -> "Standardizer":
        x64 = x.astype(np.float64, copy=False)
        mean = x64.mean(axis=0)
        std = np.maximum(x64.std(axis=0, ddof=0), min_std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)

    def state(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
        }


class InverseDynamicsMLP(nn.Module):
    """Small normalized-action MLP for (z_t, z_{t+1}) -> a_t."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: tuple[int, int] = (256, 128)) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def make_transition_features(obs: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
    """Concatenate endpoints and latent delta, keeping the encoder frozen."""
    delta = next_obs - obs
    return np.concatenate([obs, next_obs, delta], axis=1).astype(np.float32)


def load_transition_npz(path: Path) -> dict[str, Any]:
    with np.load(path) as data:
        required = [
            "obs",
            "next_obs",
            "action",
            "episode_id",
            "train_idx",
            "val_idx",
            "test_idx",
            "train_episodes",
            "val_episodes",
            "test_episodes",
        ]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{path} is missing required arrays: {missing}")
        arrays = {key: data[key].copy() for key in required}

    obs = arrays["obs"]
    next_obs = arrays["next_obs"]
    action = arrays["action"]
    if obs.ndim != 2 or next_obs.shape != obs.shape:
        raise ValueError(f"Expected obs/next_obs as matching 2D arrays, got {obs.shape} and {next_obs.shape}")
    if action.ndim != 2 or action.shape[0] != obs.shape[0]:
        raise ValueError(f"Expected action as N x A array aligned with obs, got {action.shape}")
    if not np.isfinite(obs).all() or not np.isfinite(next_obs).all() or not np.isfinite(action).all():
        raise ValueError("Transition cache contains non-finite obs, next_obs, or action values.")

    features = make_transition_features(obs, next_obs)
    splits = {
        "train": arrays["train_idx"],
        "val": arrays["val_idx"],
        "test": arrays["test_idx"],
    }
    episodes = {
        "train": arrays["train_episodes"],
        "val": arrays["val_episodes"],
        "test": arrays["test_episodes"],
    }
    return {
        "features": features,
        "action": action.astype(np.float32),
        "episode_id": arrays["episode_id"],
        "splits": splits,
        "episodes": episodes,
        "latent_dim": int(obs.shape[1]),
        "action_dim": int(action.shape[1]),
    }


def split_arrays(dataset: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = dataset["splits"][split]
    return (
        dataset["features"][idx],
        dataset["action"][idx],
        dataset["episode_id"][idx],
    )


def prediction_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    train_action_variance: np.ndarray,
    episode_id: np.ndarray,
) -> dict[str, Any]:
    err2 = (pred - target) ** 2
    per_dim_mse = err2.mean(axis=0)
    per_dim_rmse = np.sqrt(per_dim_mse)
    per_dim_mae = np.abs(pred - target).mean(axis=0)
    normalized_per_dim = per_dim_mse / np.maximum(train_action_variance, EPS)
    split_variance = target.var(axis=0, ddof=0)
    r2_per_dim = 1.0 - per_dim_mse / np.maximum(split_variance, EPS)

    per_transition_mse = err2.mean(axis=1)
    trajectory_rows = []
    for ep in np.unique(episode_id):
        mask = episode_id == ep
        trajectory_rows.append(
            {
                "episode_id": int(ep),
                "transitions": int(mask.sum()),
                "mse": float(per_transition_mse[mask].mean()),
            }
        )

    per_dim = []
    names = ACTION_NAMES if target.shape[1] == len(ACTION_NAMES) else [f"action_{i}" for i in range(target.shape[1])]
    for i, name in enumerate(names):
        per_dim.append(
            {
                "name": name,
                "mse": float(per_dim_mse[i]),
                "rmse": float(per_dim_rmse[i]),
                "mae": float(per_dim_mae[i]),
                "train_variance_normalized_mse": float(normalized_per_dim[i]),
                "r2": float(r2_per_dim[i]),
                "target_split_variance": float(split_variance[i]),
                "target_train_variance": float(train_action_variance[i]),
            }
        )

    return {
        "mse": float(per_dim_mse.mean()),
        "rmse": float(math.sqrt(float(per_dim_mse.mean()))),
        "mae": float(np.abs(pred - target).mean()),
        "train_variance_normalized_mse": float(normalized_per_dim.mean()),
        "r2_mean": float(r2_per_dim.mean()),
        "per_dim": per_dim,
        "per_trajectory": {
            "mean_mse": float(np.mean([row["mse"] for row in trajectory_rows])),
            "min_mse": float(np.min([row["mse"] for row in trajectory_rows])),
            "max_mse": float(np.max([row["mse"] for row in trajectory_rows])),
            "episodes": trajectory_rows,
        },
    }


def evaluate_named_predictions(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    train_action_variance: np.ndarray,
    episode_id: np.ndarray,
) -> dict[str, Any]:
    return {
        name: prediction_metrics(pred, target, train_action_variance, episode_id)
        for name, pred in predictions.items()
    }


def predict_model(
    model: InverseDynamicsMLP,
    x: np.ndarray,
    x_standardizer: Standardizer,
    y_standardizer: Standardizer,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    xz = x_standardizer.transform(x)
    preds = []
    with torch.no_grad():
        for start in range(0, xz.shape[0], batch_size):
            xb = torch.from_numpy(xz[start : start + batch_size]).to(device)
            pred = model(xb).detach().cpu().numpy()
            preds.append(pred)
    pred_norm = np.concatenate(preds, axis=0)
    return y_standardizer.inverse_transform(pred_norm)


def train_readout(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    device: torch.device | None = None,
    seed: int = 2026,
    max_epochs: int = 500,
    patience: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    min_delta: float = 1e-5,
    log_path: Path | None = None,
) -> tuple[InverseDynamicsMLP, dict[str, Any], Standardizer, Standardizer]:
    set_seed(seed)
    device = device or default_device()
    x_standardizer = Standardizer.fit(x_train)
    y_standardizer = Standardizer.fit(y_train)
    xtr = x_standardizer.transform(x_train)
    xva = x_standardizer.transform(x_val)
    ytr = y_standardizer.transform(y_train)
    yva = y_standardizer.transform(y_val)

    model = InverseDynamicsMLP(input_dim=xtr.shape[1], output_dim=ytr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    xtr_t = torch.from_numpy(xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)
    xva_t = torch.from_numpy(xva).to(device)
    yva_t = torch.from_numpy(yva).to(device)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val = math.inf
    wait = 0
    history = []
    start_time = time.perf_counter()

    log_fh = log_path.open("w", encoding="utf-8") if log_path is not None else None
    try:
        if log_fh:
            log_fh.write(
                json.dumps(
                    {
                        "event": "start",
                        "device": str(device),
                        "seed": seed,
                        "max_epochs": max_epochs,
                        "patience": patience,
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                    }
                )
                + "\n"
            )

        n = xtr_t.shape[0]
        for epoch in range(1, max_epochs + 1):
            model.train()
            perm = torch.randperm(n, generator=generator)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size].to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(xtr_t[idx]), ytr_t[idx])
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                train_loss = float(loss_fn(model(xtr_t), ytr_t).item())
                val_loss = float(loss_fn(model(xva_t), yva_t).item())
            sync_device(device)
            row = {"epoch": epoch, "train_normalized_mse": train_loss, "val_normalized_mse": val_loss}
            history.append(row)
            if log_fh and (epoch == 1 or epoch % 10 == 0):
                log_fh.write(json.dumps({"event": "epoch", **row}) + "\n")
                log_fh.flush()

            if val_loss < best_val - min_delta:
                best_val = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
    finally:
        if log_fh:
            log_fh.close()

    elapsed = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    stopped_epoch = history[-1]["epoch"] if history else 0
    info = {
        "architecture": "MLP([z_t, z_{t+1}, z_{t+1}-z_t] -> 256 GELU -> 128 GELU -> action)",
        "loss": "MSE on train-standardized action targets",
        "optimizer": "AdamW",
        "seed": seed,
        "device": str(device),
        "max_epochs": max_epochs,
        "patience": patience,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "min_delta": min_delta,
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "best_val_train_variance_normalized_mse": float(best_val),
        "elapsed_seconds": float(elapsed),
        "history": history,
    }
    return model, info, x_standardizer, y_standardizer
