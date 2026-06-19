#!/usr/bin/env python3
"""Summarize and plot action_3 values in a readout transition cache."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def as_builtin(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def summarize(values: np.ndarray, *, bins: int) -> dict[str, Any]:
    hist_counts, hist_edges = np.histogram(values, bins=bins)
    rounded = np.round(values.astype(np.float64), 3)
    unique, counts = np.unique(rounded, return_counts=True)
    order = np.argsort(counts)[::-1]
    top_values = [
        {"value_rounded_3dp": float(unique[i]), "count": int(counts[i])}
        for i in order[:20]
    ]
    q = np.quantile(values, [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    value_min = float(values.min())
    value_max = float(values.max())
    span = max(value_max - value_min, 1e-12)
    near_low = float(np.mean(values <= value_min + 0.05 * span))
    near_high = float(np.mean(values >= value_max - 0.05 * span))
    middle_mass = float(np.mean((values > value_min + 0.10 * span) & (values < value_max - 0.10 * span)))
    return {
        "count": int(values.shape[0]),
        "min": value_min,
        "max": value_max,
        "mean": float(values.mean()),
        "std": float(values.std()),
        "quantiles": {
            "0.00": float(q[0]),
            "0.01": float(q[1]),
            "0.05": float(q[2]),
            "0.25": float(q[3]),
            "0.50": float(q[4]),
            "0.75": float(q[5]),
            "0.95": float(q[6]),
            "0.99": float(q[7]),
            "1.00": float(q[8]),
        },
        "near_low_fraction": near_low,
        "near_high_fraction": near_high,
        "near_extremes_fraction": near_low + near_high,
        "middle_80pct_span_fraction": middle_mass,
        "histogram": {
            "counts": hist_counts.astype(int),
            "bin_edges": hist_edges.astype(float),
        },
        "top_rounded_values": top_values,
    }


def save_plot(values_by_split: dict[str, np.ndarray], out_path: Path, *, bins: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, values in values_by_split.items():
        ax.hist(values, bins=bins, alpha=0.45, label=f"{name} (n={len(values)})")
    ax.set_title("action_3 value histogram")
    ax.set_xlabel("action_3")
    ax.set_ylabel("transition count")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--artifact-prefix", default="arm_a_converged")
    parser.add_argument("--bins", type=int, default=41)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.data) as data:
        action = data["action"].astype(np.float32)
        if action.shape[1] <= 3:
            raise ValueError(f"Expected action_dim >= 4, found {action.shape[1]}")
        action_3 = action[:, 3]
        values_by_split = {
            "all": action_3,
            "train": action_3[data["train_idx"]],
            "val": action_3[data["val_idx"]],
            "test": action_3[data["test_idx"]],
        }

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data": args.data,
        "action_index": 3,
        "bins": args.bins,
        "splits": {
            name: summarize(values, bins=args.bins)
            for name, values in values_by_split.items()
        },
    }
    summary_path = args.out_dir / f"{args.artifact_prefix}_action3_histogram_summary.json"
    plot_path = args.out_dir / f"{args.artifact_prefix}_action3_histogram.png"
    summary["plot_png"] = plot_path
    summary_path.write_text(json.dumps(summary, indent=2, default=as_builtin) + "\n", encoding="utf-8")
    save_plot(values_by_split, plot_path, bins=args.bins)
    print(f"wrote {summary_path}")
    print(f"wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
