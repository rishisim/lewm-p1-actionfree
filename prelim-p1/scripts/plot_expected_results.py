#!/usr/bin/env python3
"""Generate a hypothesized-results figure for LeWM P1.

This figure is illustrative proposal material, not experimental data.
Edit ANCHORS below to adjust the target story without touching plot code.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "proposal_expected_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


ANCHORS = {
    "data_fraction": [5, 25, 50, 100],
    "scratch": [10, 34, 56, 78],
    "pretrain_ft": [26, 48, 67, 85],
    "pretrain_invdyn": [40, 59, 74, 89],
    "band": [8, 6, 4, 3],
}

SERIES = {
    "scratch": {
        "label": "LeWM from scratch",
        "color": "#999999",
        "marker": "o",
    },
    "pretrain_ft": {
        "label": "Action-free pretrain + fine-tune",
        "color": "#1f77b4",
        "marker": "s",
    },
    "pretrain_invdyn": {
        "label": "Action-free pretrain + inverse dynamics",
        "color": "#2ca02c",
        "marker": "^",
    },
}


def interpolate_curve(x_anchor: np.ndarray, y_anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dense monotone-preserving linear interpolation through anchor points."""
    x_dense = np.linspace(x_anchor.min(), x_anchor.max(), 300)
    y_dense = np.interp(x_dense, x_anchor, y_anchor)
    return x_dense, y_dense


def save_anchor_csv(df: pd.DataFrame) -> Path:
    csv_path = OUT_DIR / "expected_results_anchors.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def make_figure(df: pd.DataFrame) -> tuple[Path, Path]:
    x = df["data_fraction"].to_numpy(dtype=float)
    band = df["band"].to_numpy(dtype=float)
    band_x, band_y = interpolate_curve(x, band)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    fig.subplots_adjust(left=0.105, right=0.975, top=0.84, bottom=0.18)

    for key, meta in SERIES.items():
        y = df[key].to_numpy(dtype=float)
        x_dense, y_dense = interpolate_curve(x, y)
        ax.fill_between(
            band_x,
            np.clip(y_dense - band_y, 0, 100),
            np.clip(y_dense + band_y, 0, 100),
            color=meta["color"],
            alpha=0.16,
            linewidth=0,
        )
        ax.plot(x_dense, y_dense, color=meta["color"], linewidth=2.6)
        ax.plot(
            x,
            y,
            linestyle="none",
            marker=meta["marker"],
            markersize=6.5,
            markerfacecolor="white",
            markeredgecolor=meta["color"],
            markeredgewidth=1.8,
            label=meta["label"],
        )

    ax.set_title("Expected Data Efficiency from Action-Free Pretraining (P1)\nHypothesized")
    ax.set_xlabel("Action-labeled data fraction")
    ax.set_ylabel("Planning success")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 100)
    ax.set_xticks([5, 25, 50, 100], ["5%", "25%", "50%", "100%"])
    ax.set_yticks(np.arange(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.grid(True, linestyle="--", linewidth=0.8, color="#d9d9d9")
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)

    fig.text(
        0.5,
        0.035,
        "Hypothesized target outcome - illustrative, not experimental data",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#666666",
    )

    png_path = OUT_DIR / "expected_results_plot.png"
    pdf_path = OUT_DIR / "expected_results_plot.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    df = pd.DataFrame(ANCHORS)
    csv_path = save_anchor_csv(df)
    png_path, pdf_path = make_figure(df)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
