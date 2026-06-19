#!/usr/bin/env python3
"""Save Arm A training curves and a conservative plateau summary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "gap_test" / "outputs" / "arm_a_conditioned"
PRIMARY_METRIC_CANDIDATES = {
    "fit_pred_loss": [
        "fit/pred_loss_epoch",
        "train/pred_loss_epoch",
        "fit/pred_loss",
    ],
    "fit_sigreg_loss": [
        "fit/sigreg_loss_epoch",
        "train/sigreg_loss_epoch",
        "fit/sigreg_loss",
    ],
    "validate_pred_loss": [
        "validate/pred_loss_epoch",
        "validate/pred_loss",
    ],
    "validate_sigreg_loss": [
        "validate/sigreg_loss_epoch",
        "validate/sigreg_loss",
    ],
}


def as_builtin(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def epoch_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "epoch" not in df.columns:
        raise ValueError("metrics CSV has no epoch column")

    rows = []
    for epoch, group in df.groupby("epoch", dropna=True):
        row: dict[str, Any] = {"epoch": int(epoch)}
        for col in df.columns:
            if col in {"epoch", "step"}:
                continue
            values = group[col].dropna()
            if not values.empty:
                row[col] = float(values.iloc[-1])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)


def choose_metric_column(curves: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in curves.columns and curves[col].dropna().shape[0] >= 2:
            return col
    return None


def plateau_stats(
    curves: pd.DataFrame,
    col: str,
    *,
    window: int | None = None,
    mean_change_threshold: float = 0.03,
    floor_threshold: float = 0.05,
) -> dict[str, Any]:
    series = curves[["epoch", col]].dropna()
    y = series[col].to_numpy(dtype=np.float64)
    epochs = series["epoch"].to_numpy(dtype=np.float64)
    n = int(y.shape[0])
    if window is None:
        window = max(5, n // 10)
    window = min(window, max(1, n // 2))

    last = y[-window:]
    prev = y[-2 * window : -window] if n >= 2 * window else y[:window]
    x_last = epochs[-window:]
    last_mean = float(last.mean())
    prev_mean = float(prev.mean())
    relative_window_change = float(abs(last_mean - prev_mean) / max(abs(prev_mean), 1e-12))
    slope = 0.0
    if window >= 2:
        slope = float(np.polyfit(x_last - x_last[0], last, deg=1)[0])
    value_range = float(max(y.max() - y.min(), 1e-12))
    slope_fraction_of_range_per_epoch = float(abs(slope) / value_range)
    last_to_min_fraction = float((last_mean - y.min()) / max(abs(y.min()), 1e-12))

    mean_flattened = bool(n >= 2 * window and relative_window_change <= mean_change_threshold)
    plateaued = bool(
        n >= 20
        and mean_flattened
        and slope_fraction_of_range_per_epoch <= 0.01
    )
    at_floor = bool(last_to_min_fraction <= floor_threshold)
    converged_at_floor = bool(mean_flattened and at_floor)
    return {
        "column": col,
        "epochs_observed": n,
        "window_epochs": int(window),
        "first": float(y[0]),
        "last": float(y[-1]),
        "min": float(y.min()),
        "last_window_epochs": [int(x) for x in epochs[-window:]],
        "last_window_mean": last_mean,
        "previous_window_mean": prev_mean,
        "relative_window_change": relative_window_change,
        "last_window_slope_per_epoch": slope,
        "slope_fraction_of_range_per_epoch": slope_fraction_of_range_per_epoch,
        "last_mean_above_running_min_fraction": last_to_min_fraction,
        "mean_flattened": mean_flattened,
        "plateaued": plateaued,
        "at_running_floor": at_floor,
        "converged_at_floor": converged_at_floor,
        "criteria": (
            "mean_flattened requires two full windows and <=3% mean change between the last "
            "window and previous window; at_running_floor requires the last-window mean to be "
            "within 5% of the running minimum; converged_at_floor requires both. The legacy "
            "plateaued field also requires last-window slope <=1% of observed range per epoch."
        ),
        "legacy_plateaued_criteria": (
            "plateaued requires at least 20 epochs, <=3% mean change between the last "
            "window and previous window, and last-window slope <=1% of observed range per epoch; "
            "at_running_floor is reported independently."
        ),
    }


def first_joint_validation_plateau_epoch(
    curves: pd.DataFrame,
    selected: dict[str, str],
    *,
    window: int,
    mean_change_threshold: float,
    floor_threshold: float,
) -> dict[str, Any]:
    required = {
        "validate_pred_loss": selected.get("validate_pred_loss"),
        "validate_sigreg_loss": selected.get("validate_sigreg_loss"),
    }
    if not all(required.values()):
        return {"epoch": None, "epoch_1based": None, "reason": "missing validation metrics"}

    min_epoch = int(curves["epoch"].min())
    max_epoch = int(curves["epoch"].max())
    first_each: dict[str, int | None] = {name: None for name in required}

    for epoch in range(min_epoch, max_epoch + 1):
        prefix = curves[curves["epoch"] <= epoch]
        stats = {
            name: plateau_stats(
                prefix,
                col,
                window=window,
                mean_change_threshold=mean_change_threshold,
                floor_threshold=floor_threshold,
            )
            for name, col in required.items()
            if col
        }
        for name, row in stats.items():
            if first_each[name] is None and row.get("converged_at_floor", False):
                first_each[name] = epoch
        if all(row.get("converged_at_floor", False) for row in stats.values()):
            return {
                "epoch": int(epoch),
                "epoch_1based": int(epoch + 1),
                "window_epochs": int(window),
                "first_each_epoch": first_each,
                "first_each_epoch_1based": {
                    name: (value + 1 if value is not None else None)
                    for name, value in first_each.items()
                },
                "mean_change_threshold": mean_change_threshold,
                "floor_threshold": floor_threshold,
                "criteria": (
                    "First epoch where both validate/pred_loss and validate/sigreg_loss have "
                    "last-window mean within the mean-change threshold of the previous window "
                    "and within the floor threshold of the running minimum."
                ),
            }

    return {
        "epoch": None,
        "epoch_1based": None,
        "window_epochs": int(window),
        "first_each_epoch": first_each,
        "first_each_epoch_1based": {
            name: (value + 1 if value is not None else None)
            for name, value in first_each.items()
        },
        "mean_change_threshold": mean_change_threshold,
        "floor_threshold": floor_threshold,
        "reason": "joint validation convergence not reached",
    }


def save_plot(curves: pd.DataFrame, selected: dict[str, str], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(selected), 1, figsize=(8, 3.2 * max(1, len(selected))), squeeze=False)
    for ax, (name, col) in zip(axes[:, 0], selected.items()):
        series = curves[["epoch", col]].dropna()
        ax.plot(series["epoch"], series[col], marker="o", linewidth=1.5, markersize=2.5)
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--artifact-prefix", default="arm_a")
    parser.add_argument("--window-epochs", type=int, default=20)
    parser.add_argument("--mean-change-threshold", type=float, default=0.03)
    parser.add_argument("--floor-threshold", type=float, default=0.05)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metrics_csv)
    curves = epoch_summary(df)
    curves_path = args.out_dir / f"{args.artifact_prefix}_training_curves.csv"
    summary_path = args.out_dir / f"{args.artifact_prefix}_training_curves_summary.json"
    plot_path = args.out_dir / f"{args.artifact_prefix}_training_curves.png"
    curves.to_csv(curves_path, index=False)

    selected = {
        name: col
        for name, candidates in PRIMARY_METRIC_CANDIDATES.items()
        if (col := choose_metric_column(curves, candidates)) is not None
    }
    plateau = {
        name: plateau_stats(
            curves,
            col,
            window=args.window_epochs,
            mean_change_threshold=args.mean_change_threshold,
            floor_threshold=args.floor_threshold,
        )
        for name, col in selected.items()
    }
    validation_plateau = first_joint_validation_plateau_epoch(
        curves,
        selected,
        window=args.window_epochs,
        mean_change_threshold=args.mean_change_threshold,
        floor_threshold=args.floor_threshold,
    )
    if not args.no_plot:
        save_plot(curves, selected, plot_path)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_csv": args.metrics_csv,
        "curves_csv": curves_path,
        "plot_png": plot_path,
        "epochs_in_csv": int(curves["epoch"].nunique()),
        "selected_columns": selected,
        "plateau": plateau,
        "validation_plateau": validation_plateau,
        "both_plateaued": bool(
            plateau.get("fit_pred_loss", {}).get("plateaued", False)
            and plateau.get("fit_sigreg_loss", {}).get("plateaued", False)
        ),
        "both_validate_plateaued": bool(
            plateau.get("validate_pred_loss", {}).get("plateaued", False)
            and plateau.get("validate_sigreg_loss", {}).get("plateaued", False)
        ),
        "both_validate_converged_at_floor": bool(
            plateau.get("validate_pred_loss", {}).get("converged_at_floor", False)
            and plateau.get("validate_sigreg_loss", {}).get("converged_at_floor", False)
        ),
        "sigreg_at_floor": bool(plateau.get("fit_sigreg_loss", {}).get("at_running_floor", False)),
        "validate_pred_at_floor": bool(plateau.get("validate_pred_loss", {}).get("at_running_floor", False)),
        "validate_sigreg_at_floor": bool(plateau.get("validate_sigreg_loss", {}).get("at_running_floor", False)),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=as_builtin) + "\n", encoding="utf-8")
    print(f"wrote {curves_path}")
    if not args.no_plot:
        print(f"wrote {plot_path}")
    print(f"wrote {summary_path}")
    print(
        "plateau: "
        f"fit_pred_loss={summary['plateau'].get('fit_pred_loss', {}).get('plateaued')}, "
        f"fit_sigreg_loss={summary['plateau'].get('fit_sigreg_loss', {}).get('plateaued')}, "
        f"fit_sigreg_at_floor={summary['sigreg_at_floor']}, "
        f"validate_pred_loss={summary['plateau'].get('validate_pred_loss', {}).get('plateaued')}, "
        f"validate_sigreg_loss={summary['plateau'].get('validate_sigreg_loss', {}).get('plateaued')}, "
        f"validation_converged_at_floor={summary['both_validate_converged_at_floor']}, "
        f"validation_plateau_epoch_1based={summary['validation_plateau'].get('epoch_1based')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
