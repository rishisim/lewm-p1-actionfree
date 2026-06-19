#!/usr/bin/env python3
"""Train and evaluate the Arm A inverse-dynamics readout."""

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
    ACTION_NAMES,
    default_device,
    evaluate_named_predictions,
    load_transition_npz,
    predict_model,
    split_arrays,
    train_readout,
)


DEFAULT_DATA = ROOT / "gap_test" / "outputs" / "arm_a_conditioned" / "arm_a_encoder_transitions.npz"
DEFAULT_SUMMARY = ROOT / "gap_test" / "outputs" / "arm_a_conditioned" / "arm_a_encoder_transitions_summary.json"
DEFAULT_OUT_DIR = ROOT / "gap_test" / "outputs" / "arm_a_conditioned"
DEFAULT_PROBE_SUMMARY = ROOT / "probe" / "outputs" / "probe_summary.json"
ACTION_3_INDEX = 3
DISCREPANCY_THRESHOLD = 0.10


def as_builtin(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def mean_without_action_3(per_dim: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for i, row in enumerate(per_dim) if i != ACTION_3_INDEX]
    return float(np.mean(values))


def readout_headlines(metrics: dict[str, Any], probe_summary: dict[str, Any]) -> dict[str, Any]:
    test_readout = metrics["test"]["readout_mlp"]
    per_dim = test_readout["per_dim"]
    action_3 = per_dim[ACTION_3_INDEX] if len(per_dim) > ACTION_3_INDEX else None

    headline = {
        "test_r2_all_dims": float(test_readout["r2_mean"]),
        "test_r2_excluding_action_3": mean_without_action_3(per_dim, "r2"),
        "test_train_variance_normalized_mse_all_dims": float(test_readout["train_variance_normalized_mse"]),
        "test_train_variance_normalized_mse_excluding_action_3": mean_without_action_3(
            per_dim, "train_variance_normalized_mse"
        ),
        "action_3": action_3,
    }
    baseline_metrics = metrics["test"].get("mean_action_baseline", {})
    baseline_per_dim = baseline_metrics.get("per_dim", [])
    if baseline_metrics:
        headline["mean_action_baseline"] = {
            "test_train_variance_normalized_mse_all_dims": float(
                baseline_metrics["train_variance_normalized_mse"]
            ),
            "test_train_variance_normalized_mse_excluding_action_3": mean_without_action_3(
                baseline_per_dim, "train_variance_normalized_mse"
            )
            if baseline_per_dim
            else None,
        }

    probe_metrics = probe_summary.get("metrics", {}).get("test", {}).get("readout_mlp", {})
    if probe_metrics:
        probe_per_dim = probe_metrics.get("per_dim", [])
        probe_all = float(probe_metrics["r2_mean"])
        probe_no_action_3 = mean_without_action_3(probe_per_dim, "r2") if probe_per_dim else None
        delta_all = headline["test_r2_all_dims"] - probe_all
        delta_no_action_3 = (
            headline["test_r2_excluding_action_3"] - probe_no_action_3
            if probe_no_action_3 is not None
            else None
        )
        headline["probe_comparison"] = {
            "probe_test_r2_all_dims": probe_all,
            "probe_test_r2_excluding_action_3": probe_no_action_3,
            "delta_vs_probe_all_dims": float(delta_all),
            "delta_vs_probe_excluding_action_3": float(delta_no_action_3)
            if delta_no_action_3 is not None
            else None,
            "large_discrepancy_threshold_abs": DISCREPANCY_THRESHOLD,
            "large_discrepancy_vs_probe_all_dims": bool(abs(delta_all) >= DISCREPANCY_THRESHOLD),
            "large_discrepancy_vs_probe_excluding_action_3": bool(
                delta_no_action_3 is not None and abs(delta_no_action_3) >= DISCREPANCY_THRESHOLD
            ),
        }

    return headline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--probe-summary", type=Path, default=DEFAULT_PROBE_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--artifact-prefix", default="arm_a")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / f"{args.artifact_prefix}_readout.log"
    summary_out_path = args.out_dir / f"{args.artifact_prefix}_readout_summary.json"
    model_path = args.out_dir / f"{args.artifact_prefix}_readout_best.pt"

    dataset = load_transition_npz(args.data)
    source_summary = load_json(args.summary)
    probe_summary = load_json(args.probe_summary)

    x_train, y_train, ep_train = split_arrays(dataset, "train")
    x_val, y_val, ep_val = split_arrays(dataset, "val")
    x_test, y_test, ep_test = split_arrays(dataset, "test")

    device = default_device() if args.device == "auto" else torch.device(args.device)
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
    dims_improved = sum(
        readout_row["train_variance_normalized_mse"] < baseline_row["train_variance_normalized_mse"]
        for baseline_row, readout_row in zip(
            metrics["test"]["mean_action_baseline"]["per_dim"],
            metrics["test"]["readout_mlp"]["per_dim"],
        )
    )

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
        "data_path": args.data,
        "source_summary_path": args.summary,
        "output_dir": args.out_dir,
        "dataset": {
            "latent_dim": dataset["latent_dim"],
            "action_dim": dataset["action_dim"],
            "action_names": ACTION_NAMES[: dataset["action_dim"]],
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
        },
        "train_info": train_info,
        "baseline": {
            "name": "mean_action_baseline",
            "definition": "Predict the constant train-split mean action for every transition.",
            "train_mean_action": train_mean_action.astype(float).tolist(),
            "train_action_variance": train_action_variance.astype(float).tolist(),
        },
        "metrics": metrics,
        "headline": readout_headlines(metrics, probe_summary),
        "gate": {
            "baseline_test_train_variance_normalized_mse": float(baseline_test),
            "readout_test_train_variance_normalized_mse": float(readout_test),
            "relative_improvement": float(relative_improvement),
            "test_action_dims_improved": int(dims_improved),
            "test_action_dims_total": int(dataset["action_dim"]),
            "passed": bool(readout_test < baseline_test),
        },
        "artifacts": {
            "summary_json": summary_out_path,
            "log": log_path,
            "model_checkpoint": model_path,
        },
    }

    summary_out_path.write_text(json.dumps(output, indent=2, default=as_builtin) + "\n", encoding="utf-8")
    print(f"wrote {summary_out_path}")
    print(f"wrote {log_path}")
    print(f"wrote {model_path}")
    print(
        "test R2: "
        f"all={output['headline']['test_r2_all_dims']:.6f}, "
        f"excluding_action_3={output['headline']['test_r2_excluding_action_3']:.6f}"
    )
    print(
        "test normalized MSE: "
        f"baseline={baseline_test:.6f}, readout={readout_test:.6f}, "
        f"relative_improvement={relative_improvement:.1%}, dims_improved={dims_improved}/{dataset['action_dim']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
