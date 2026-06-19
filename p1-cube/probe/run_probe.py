#!/usr/bin/env python3
"""Train and evaluate the frozen LeWM-cube inverse-dynamics probe."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = ROOT / "harness"
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

from idm_readout import (  # noqa: E402
    default_device,
    evaluate_named_predictions,
    load_transition_npz,
    predict_model,
    split_arrays,
    train_readout,
)


DATA_PATH = ROOT / "data" / "cube_single_play_probe_transitions.npz"
SUMMARY_PATH = ROOT / "data" / "cube_single_play_probe_transitions_summary.json"
OUT_DIR = ROOT / "probe" / "outputs"
GATE_MIN_RELATIVE_IMPROVEMENT = 0.10


def as_builtin(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "probe.log"
    summary_out_path = args.out_dir / "probe_summary.json"
    model_path = args.out_dir / "readout_best.pt"

    dataset = load_transition_npz(args.data)
    source_summary = load_summary(args.summary)

    x_train, y_train, ep_train = split_arrays(dataset, "train")
    x_val, y_val, ep_val = split_arrays(dataset, "val")
    x_test, y_test, ep_test = split_arrays(dataset, "test")

    if args.device == "auto":
        device = default_device()
    else:
        device = torch.device(args.device)

    model, train_info, x_std, y_std = train_readout(
        x_train,
        y_train,
        x_val,
        y_val,
        device=device,
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_path=log_path,
    )

    readout_predictions = {
        "train": predict_model(model, x_train, x_std, y_std, device),
        "val": predict_model(model, x_val, x_std, y_std, device),
        "test": predict_model(model, x_test, x_std, y_std, device),
    }
    train_mean_action = y_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    train_action_variance = np.maximum(y_train.var(axis=0, ddof=0), 1e-12).astype(np.float32)
    baseline_predictions = {
        "train": np.broadcast_to(train_mean_action, y_train.shape).astype(np.float32),
        "val": np.broadcast_to(train_mean_action, y_val.shape).astype(np.float32),
        "test": np.broadcast_to(train_mean_action, y_test.shape).astype(np.float32),
    }

    targets = {"train": y_train, "val": y_val, "test": y_test}
    episode_ids = {"train": ep_train, "val": ep_val, "test": ep_test}
    metrics = {
        split: evaluate_named_predictions(
            {
                "mean_action_baseline": baseline_predictions[split],
                "readout_mlp": readout_predictions[split],
            },
            targets[split],
            train_action_variance,
            episode_ids[split],
        )
        for split in ("train", "val", "test")
    }

    baseline_test = metrics["test"]["mean_action_baseline"]["train_variance_normalized_mse"]
    readout_test = metrics["test"]["readout_mlp"]["train_variance_normalized_mse"]
    relative_improvement = (baseline_test - readout_test) / max(baseline_test, 1e-12)
    per_dim_baseline = metrics["test"]["mean_action_baseline"]["per_dim"]
    per_dim_readout = metrics["test"]["readout_mlp"]["per_dim"]
    dims_improved = sum(
        readout_row["train_variance_normalized_mse"] < baseline_row["train_variance_normalized_mse"]
        for baseline_row, readout_row in zip(per_dim_baseline, per_dim_readout)
    )
    gate_passed = bool(relative_improvement >= GATE_MIN_RELATIVE_IMPROVEMENT and readout_test < baseline_test)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "x_standardizer": x_std.state(),
            "y_standardizer": y_std.state(),
            "train_info": train_info,
            "feature_layout": "[z_t, z_{t+1}, z_{t+1}-z_t]",
        },
        model_path,
    )

    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(args.data),
        "source_summary_path": str(args.summary),
        "output_dir": str(args.out_dir),
        "dataset": {
            "variant": source_summary.get("dataset_variant", "unknown"),
            "latent_dim": dataset["latent_dim"],
            "action_dim": dataset["action_dim"],
            "transitions": int(dataset["action"].shape[0]),
            "splits": {
                split: {
                    "transitions": int(len(dataset["splits"][split])),
                    "episodes": [int(x) for x in dataset["episodes"][split]],
                }
                for split in ("train", "val", "test")
            },
            "encoder": source_summary.get("encoder", {}),
        },
        "readout_choice": {
            "features": "[z_t, z_{t+1}, z_{t+1}-z_t]",
            "architecture": train_info["architecture"],
            "loss": train_info["loss"],
            "optimizer": f"{train_info['optimizer']}(lr={train_info['lr']}, weight_decay={train_info['weight_decay']})",
            "normalization": (
                "Inputs and action targets are standardized with train-split statistics. "
                "Reported normalized MSE divides each action-dimension MSE by train action variance."
            ),
            "reason": (
                "A shallow MLP is expressive enough for a smoke-test readout from frozen latents, "
                "while the train-mean baseline and per-dimension normalized errors keep the result interpretable."
            ),
        },
        "train_info": train_info,
        "baseline": {
            "name": "mean_action_baseline",
            "definition": "Predict the constant train-split mean action for every transition.",
            "train_mean_action": train_mean_action.astype(float).tolist(),
            "train_action_variance": train_action_variance.astype(float).tolist(),
        },
        "metrics": metrics,
        "gate": {
            "criterion": (
                "PASS if held-out test train-variance-normalized MSE improves by at least "
                f"{GATE_MIN_RELATIVE_IMPROVEMENT:.0%} over the train-mean-action baseline."
            ),
            "baseline_test_train_variance_normalized_mse": float(baseline_test),
            "readout_test_train_variance_normalized_mse": float(readout_test),
            "relative_improvement": float(relative_improvement),
            "test_action_dims_improved": int(dims_improved),
            "test_action_dims_total": int(dataset["action_dim"]),
            "passed": gate_passed,
        },
        "artifacts": {
            "summary_json": str(summary_out_path),
            "log": str(log_path),
            "model_checkpoint": str(model_path),
        },
    }

    summary_out_path.write_text(json.dumps(output, indent=2, default=as_builtin) + "\n", encoding="utf-8")

    print(f"wrote {summary_out_path}")
    print(f"wrote {log_path}")
    print(f"wrote {model_path}")
    print(
        "test normalized MSE: "
        f"baseline={baseline_test:.6f}, readout={readout_test:.6f}, "
        f"relative_improvement={relative_improvement:.1%}, dims_improved={dims_improved}/{dataset['action_dim']}"
    )
    if not gate_passed:
        print("STOP: readout did not clearly beat the mean-action baseline on held-out test trajectories.")
        return 2
    print("PASS: readout clearly beats the mean-action baseline on held-out test trajectories.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
