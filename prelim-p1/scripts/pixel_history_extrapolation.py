#!/usr/bin/env python3
"""Pixel-space history extrapolation diagnostic for the LeWM task suite.

This script intentionally avoids LeWM checkpoints, encoders, and normalized
latent features. It compares raw next-frame prediction baselines using only the
pixel history available to action-free pretraining.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[2]
for rel in ("stable-worldmodel-readonly", "stable-pretraining-readonly", "le-wm"):
    path = str(ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402


OUT_DIR = ROOT / "prelim-p1" / "outputs" / "history_extrapolation"
HISTORY_SIZE = 3
NUM_PREDS = 1
FRAMESKIP = 5
PIXEL_KEY = "pixels"


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    load_name: str
    keys_to_load: tuple[str, ...]
    keys_to_cache: tuple[str, ...]
    local_candidates: tuple[Path, ...]
    hf_repo: str | None = None
    hf_archive: str | None = None
    extract_subdir: Path | None = None


def dataset_specs(cache_root: Path) -> dict[str, DatasetSpec]:
    datasets_dir = cache_root / "datasets"
    return {
        "pusht": DatasetSpec(
            label="PushT",
            load_name="pusht_expert_train.lance",
            keys_to_load=("pixels", "action", "proprio", "state"),
            keys_to_cache=("action", "proprio", "state"),
            local_candidates=(
                datasets_dir / "pusht_expert_train.lance",
                datasets_dir / "pusht_expert_train.h5",
            ),
            hf_repo="quentinll/lewm-pusht",
            hf_archive="pusht_expert_train.h5.zst",
        ),
        "reacher": DatasetSpec(
            label="Reacher",
            load_name="reacher.h5",
            keys_to_load=("pixels", "action", "observation"),
            keys_to_cache=("action", "observation"),
            local_candidates=(datasets_dir / "reacher.h5", datasets_dir / "reacher.lance"),
            hf_repo="quentinll/lewm-reacher",
            hf_archive="reacher.tar.zst",
        ),
        "cube": DatasetSpec(
            label="Cube",
            load_name="ogbench/cube_single_expert.h5",
            keys_to_load=("pixels", "action", "observation"),
            keys_to_cache=("action", "observation"),
            local_candidates=(
                datasets_dir / "ogbench" / "cube_single_expert.h5",
                datasets_dir / "cube_single_expert.h5",
            ),
            hf_repo="quentinll/lewm-cube",
            hf_archive="cube_single_expert.tar.zst",
            extract_subdir=Path("ogbench"),
        ),
        "tworoom": DatasetSpec(
            label="TwoRoom",
            load_name="tworoom.h5",
            keys_to_load=("pixels", "action", "proprio"),
            keys_to_cache=("action", "proprio"),
            local_candidates=(datasets_dir / "tworoom.h5",),
            hf_repo="quentinll/lewm-tworooms",
            hf_archive="tworoom.tar.zst",
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare copy-last-frame vs pixel-space history extrapolation baselines."
    )
    parser.add_argument(
        "--datasets",
        default="pusht,reacher,cube,tworoom",
        help="Comma-separated dataset keys to run. Defaults to all four.",
    )
    parser.add_argument("--max-windows-per-dataset", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--history-size", type=int, default=HISTORY_SIZE)
    parser.add_argument("--num-preds", type=int, default=NUM_PREDS)
    parser.add_argument("--frameskip", type=int, default=FRAMESKIP)
    parser.add_argument("--cache-dir", type=Path, default=Path(swm.data.utils.get_cache_dir()))
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--fetch-missing",
        dest="fetch_missing",
        action="store_true",
        default=True,
        help="Download and extract missing HF dataset archives when available.",
    )
    parser.add_argument(
        "--no-fetch-missing",
        dest="fetch_missing",
        action="store_false",
        help="Do not download missing datasets; record them as missing.",
    )
    parser.add_argument(
        "--write-montages",
        action="store_true",
        help="Write small qualitative copy/extrapolate/target PNG montages.",
    )
    return parser.parse_args()


def existing_candidate(spec: DatasetSpec) -> Path | None:
    for candidate in spec.local_candidates:
        if candidate.exists():
            return candidate
    return None


def extract_zst_file(archive: Path, dest: Path) -> None:
    import zstandard as zstd

    dest.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("rb") as src, dest.open("wb") as out:
        reader = zstd.ZstdDecompressor().stream_reader(src)
        shutil.copyfileobj(reader, out)


def extract_tar_zst(archive: Path, dest_dir: Path) -> None:
    import zstandard as zstd

    dest_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with archive.open("rb") as src:
            reader = zstd.ZstdDecompressor().stream_reader(src)
            shutil.copyfileobj(reader, tmp)
    try:
        with tarfile.open(tmp_path, "r") as tar:
            tar.extractall(dest_dir, filter="data")
    finally:
        tmp_path.unlink(missing_ok=True)


def fetch_dataset(spec: DatasetSpec, cache_root: Path) -> tuple[bool, str]:
    if existing_candidate(spec) is not None:
        return True, "already-present"
    if spec.hf_repo is None or spec.hf_archive is None:
        return False, "no-fetch-source-configured"

    try:
        archive = Path(
            hf_hub_download(
                repo_id=spec.hf_repo,
                filename=spec.hf_archive,
                repo_type="dataset",
                cache_dir=str(cache_root / "hf"),
            )
        )
    except Exception as exc:  # noqa: BLE001 - error is recorded in output JSON.
        return False, f"download-failed: {type(exc).__name__}: {exc}"

    datasets_dir = cache_root / "datasets"
    try:
        if spec.hf_archive.endswith(".tar.zst"):
            extract_dir = datasets_dir / spec.extract_subdir if spec.extract_subdir else datasets_dir
            extract_tar_zst(archive, extract_dir)
        elif spec.hf_archive.endswith(".h5.zst"):
            out_name = spec.hf_archive[: -len(".zst")]
            extract_zst_file(archive, datasets_dir / out_name)
        else:
            return False, f"unsupported-archive-format: {spec.hf_archive}"
    except (tarfile.TarError, OSError, RuntimeError) as exc:
        return False, f"extract-failed: {type(exc).__name__}: {exc}"

    found = existing_candidate(spec)
    if found is None:
        return False, f"extracted-but-expected-file-not-found: {spec.local_candidates}"
    return True, f"fetched-and-extracted: {found}"


def make_dataset(spec: DatasetSpec, args: argparse.Namespace):
    return swm.data.load_dataset(
        spec.load_name,
        cache_dir=str(args.cache_dir),
        transform=None,
        num_steps=args.history_size + args.num_preds,
        frameskip=args.frameskip,
        keys_to_load=list(spec.keys_to_load),
        keys_to_cache=list(spec.keys_to_cache),
    )


def normalize_pixels(pixels: Any) -> torch.Tensor:
    if not torch.is_tensor(pixels):
        pixels = torch.as_tensor(pixels)
    pixels = pixels.float()
    if pixels.ndim != 4:
        raise ValueError(f"Expected one window with 4D pixels, got shape {tuple(pixels.shape)}")
    if pixels.shape[-1] in (1, 3, 4) and pixels.shape[1] not in (1, 3, 4):
        pixels = pixels.permute(0, 3, 1, 2).contiguous()
    if pixels.shape[1] not in (1, 3, 4):
        raise ValueError(f"Expected channel-first or channel-last pixels, got shape {tuple(pixels.shape)}")
    if pixels.max().item() > 2.0:
        pixels = pixels / 255.0
    return pixels.clamp(0.0, 1.0)


def collate_windows(batch: list[dict[str, Any]]) -> torch.Tensor:
    return torch.stack([normalize_pixels(item[PIXEL_KEY]) for item in batch], dim=0)


def linear_extrapolate_three(history: torch.Tensor) -> torch.Tensor:
    # Least-squares line fit at x=0,1,2 evaluated at x=3:
    # y_hat(3) = -2/3*y0 + 1/3*y1 + 4/3*y2.
    latest = history[:, -3:]
    return ((-2.0 / 3.0) * latest[:, 0] + (1.0 / 3.0) * latest[:, 1] + (4.0 / 3.0) * latest[:, 2]).clamp(0.0, 1.0)


def update_sums(
    sums: dict[str, float],
    history: torch.Tensor,
    target: torch.Tensor,
) -> None:
    copy_pred = history[:, -1]
    velocity_pred = (history[:, -1] + (history[:, -1] - history[:, -2])).clamp(0.0, 1.0)
    linear_pred = linear_extrapolate_three(history)

    sums["copy_sse"] += (copy_pred - target).pow(2).sum().item()
    sums["velocity_sse"] += (velocity_pred - target).pow(2).sum().item()
    sums["linear_sse"] += (linear_pred - target).pow(2).sum().item()
    sums["pixels"] += float(target.numel())


def episode_count(dataset: Any, sampled_indices: np.ndarray) -> int | None:
    for attr in ("episode_idx", "ep_idx"):
        try:
            data = dataset.get_col_data(attr)
        except Exception:  # noqa: BLE001 - schema differs by dataset.
            continue
        try:
            return int(np.unique(np.asarray(data)[sampled_indices]).size)
        except Exception:  # noqa: BLE001 - some formats may not expose row-aligned columns.
            return None

    lengths = getattr(dataset, "lengths", None)
    offsets = getattr(dataset, "offsets", None)
    if lengths is None or offsets is None:
        return None
    starts = np.asarray(offsets)
    ends = starts + np.asarray(lengths)
    ep_ids = np.searchsorted(ends, sampled_indices, side="right")
    return int(np.unique(ep_ids).size)


def write_montage(
    output_dir: Path,
    dataset_key: str,
    windows: torch.Tensor,
    max_items: int = 4,
) -> str | None:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None

    if windows.numel() == 0:
        return None
    windows = windows[:max_items].cpu()
    history = windows[:, :3]
    target = windows[:, 3]
    copy_pred = history[:, -1]
    velocity_pred = (history[:, -1] + (history[:, -1] - history[:, -2])).clamp(0.0, 1.0)
    rows = []
    for i in range(windows.shape[0]):
        frames = [history[i, -1], copy_pred[i], velocity_pred[i], target[i]]
        imgs = []
        for frame in frames:
            arr = (frame[:3].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            imgs.append(Image.fromarray(arr))
        row = Image.new("RGB", (imgs[0].width * len(imgs), imgs[0].height + 22), "white")
        draw = ImageDraw.Draw(row)
        for j, img in enumerate(imgs):
            row.paste(img, (j * img.width, 22))
        for j, label in enumerate(("last", "copy", "extrap", "target")):
            draw.text((j * imgs[0].width + 4, 4), label, fill=(0, 0, 0))
        rows.append(row)

    montage = Image.new("RGB", (rows[0].width, rows[0].height * len(rows)), "white")
    for i, row in enumerate(rows):
        montage.paste(row, (0, i * row.height))
    path = output_dir / f"{dataset_key}_montage.png"
    montage.save(path)
    return str(path)


def run_dataset(
    key: str,
    spec: DatasetSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    availability = existing_candidate(spec)
    fetch_status = "already-present" if availability is not None else "missing"
    if availability is None and args.fetch_missing:
        fetched, fetch_status = fetch_dataset(spec, args.cache_dir)
        availability = existing_candidate(spec) if fetched else None

    if availability is None:
        return {
            "dataset": key,
            "label": spec.label,
            "status": "missing",
            "load_name": spec.load_name,
            "fetch_status": fetch_status,
        }

    started = time.perf_counter()
    try:
        dataset = make_dataset(spec, args)
    except Exception as exc:  # noqa: BLE001 - recorded, then other datasets continue.
        return {
            "dataset": key,
            "label": spec.label,
            "status": "load_failed",
            "load_name": spec.load_name,
            "local_path": str(availability),
            "fetch_status": fetch_status,
            "error": f"{type(exc).__name__}: {exc}",
        }

    total_windows = len(dataset)
    if total_windows <= 0:
        return {
            "dataset": key,
            "label": spec.label,
            "status": "empty",
            "load_name": spec.load_name,
            "local_path": str(availability),
            "fetch_status": fetch_status,
        }

    n = min(args.max_windows_per_dataset, total_windows)
    rng = np.random.default_rng(args.seed + sum(ord(ch) for ch in key))
    sampled = np.sort(rng.choice(total_windows, size=n, replace=False).astype(np.int64))
    subset = torch.utils.data.Subset(dataset, sampled.tolist())
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_windows,
    )

    sums = {"copy_sse": 0.0, "velocity_sse": 0.0, "linear_sse": 0.0, "pixels": 0.0}
    example_windows: torch.Tensor | None = None
    for windows in loader:
        if windows.shape[1] != args.history_size + args.num_preds:
            raise ValueError(
                f"{key}: expected {args.history_size + args.num_preds} frames, got {windows.shape[1]}"
            )
        history = windows[:, : args.history_size]
        target = windows[:, args.history_size]
        if history.shape[1] < 3:
            raise ValueError("The configured linear baselines require at least three history frames.")
        if target.shape != history[:, -1].shape:
            raise ValueError(f"{key}: target shape {target.shape} does not match prediction shape")
        update_sums(sums, history, target)
        if example_windows is None:
            example_windows = windows[:4].detach().cpu()

    copy_mse = sums["copy_sse"] / sums["pixels"]
    velocity_mse = sums["velocity_sse"] / sums["pixels"]
    linear_mse = sums["linear_sse"] / sums["pixels"]
    for metric_name, value in (
        ("copy_mse", copy_mse),
        ("velocity_mse", velocity_mse),
        ("linear_mse", linear_mse),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key}: invalid {metric_name}={value}")

    gap = copy_mse - velocity_mse
    relative = gap / copy_mse if copy_mse > 0 else float("nan")
    linear_gap = copy_mse - linear_mse
    linear_relative = linear_gap / copy_mse if copy_mse > 0 else float("nan")
    montage_path = write_montage(args.output_dir, key, example_windows) if args.write_montages and example_windows is not None else None

    return {
        "dataset": key,
        "label": spec.label,
        "status": "ok",
        "load_name": spec.load_name,
        "local_path": str(availability),
        "fetch_status": fetch_status,
        "total_valid_windows": int(total_windows),
        "sampled_windows": int(n),
        "sampled_episodes": episode_count(dataset, sampled),
        "history_size": args.history_size,
        "num_preds": args.num_preds,
        "frameskip": args.frameskip,
        "copy_mse": copy_mse,
        "velocity_mse": velocity_mse,
        "velocity_absolute_gap": gap,
        "velocity_relative_improvement": relative,
        "linear_mse": linear_mse,
        "linear_absolute_gap": linear_gap,
        "linear_relative_improvement": linear_relative,
        "elapsed_sec": time.perf_counter() - started,
        "montage_path": montage_path,
    }


def sorted_ok_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [row for row in results if row.get("status") == "ok"],
        key=lambda row: row["velocity_absolute_gap"],
        reverse=True,
    )


def write_outputs(results: list[dict[str, Any]], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at_unix": time.time(),
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "config": {
            "history_size": args.history_size,
            "num_preds": args.num_preds,
            "frameskip": args.frameskip,
            "max_windows_per_dataset": args.max_windows_per_dataset,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "cache_dir": str(args.cache_dir),
            "fetch_missing": args.fetch_missing,
        },
        "results": results,
        "ranking_by_velocity_absolute_gap": [
            row["dataset"] for row in sorted_ok_results(results)
        ],
        "complete_four_dataset_result": all(row.get("status") == "ok" for row in results)
        and len(results) == 4,
    }
    (args.output_dir / "history_extrapolation_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )

    table_fields = [
        "dataset",
        "label",
        "status",
        "sampled_windows",
        "sampled_episodes",
        "copy_mse",
        "velocity_mse",
        "velocity_absolute_gap",
        "velocity_relative_improvement",
        "linear_mse",
        "linear_absolute_gap",
        "linear_relative_improvement",
        "fetch_status",
    ]
    with (args.output_dir / "history_extrapolation_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=table_fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field, "") for field in table_fields})

    ranked = sorted_ok_results(results)
    lines = [
        "# Pixel History Extrapolation Diagnostic",
        "",
        f"- History/window setup: history_size={args.history_size}, num_preds={args.num_preds}, frameskip={args.frameskip}.",
        f"- Sampling: up to {args.max_windows_per_dataset:,} valid windows per dataset, seed={args.seed}.",
        "- Pixel-space only: no trained encoder, checkpoint, action input, or ImageNet normalization.",
        "",
        "## Ranking",
        "",
    ]
    if ranked:
        for i, row in enumerate(ranked, start=1):
            lines.append(
                f"{i}. {row['label']}: gap={row['velocity_absolute_gap']:.8f}, "
                f"relative={row['velocity_relative_improvement']:.2%}, "
                f"copy_mse={row['copy_mse']:.8f}, extrap_mse={row['velocity_mse']:.8f}"
            )
    else:
        lines.append("No datasets completed successfully.")
    lines.extend(["", "## Dataset Results", ""])
    for row in results:
        if row.get("status") == "ok":
            lines.append(
                f"- {row['label']}: sampled {row['sampled_windows']:,} windows "
                f"from {row.get('sampled_episodes') or 'unknown'} episodes; "
                f"linear sanity gap={row['linear_absolute_gap']:.8f}; "
                f"elapsed={row['elapsed_sec']:.1f}s."
            )
        else:
            lines.append(
                f"- {row['label']}: status={row.get('status')}; "
                f"fetch_status={row.get('fetch_status', '')}; error={row.get('error', '')}"
            )
    lines.extend(["", "## Recommendation", ""])
    if len(ranked) == 4:
        winner = ranked[0]
        lines.append(
            f"{winner['label']} has the largest copy-vs-extrapolation gap in this run. "
            "Use it as the action-free feasibility candidate if the ranking is stable under reruns."
        )
    else:
        lines.append(
            "This is not yet a complete four-dataset result; do not make the final dataset recommendation until all four datasets complete."
        )
    (args.output_dir / "history_extrapolation_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs = dataset_specs(args.cache_dir)
    requested = [item.strip().lower() for item in args.datasets.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(specs))
    if unknown:
        raise SystemExit(f"Unknown dataset key(s): {', '.join(unknown)}")

    results = []
    for key in requested:
        print(f"running {key}...", flush=True)
        try:
            result = run_dataset(key, specs[key], args)
        except Exception as exc:  # noqa: BLE001 - keep going and preserve failure.
            result = {
                "dataset": key,
                "label": specs[key].label,
                "status": "failed",
                "load_name": specs[key].load_name,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        if result.get("status") == "ok":
            print(
                f"  ok gap={result['velocity_absolute_gap']:.8f} "
                f"relative={result['velocity_relative_improvement']:.2%}",
                flush=True,
            )
        else:
            print(f"  {result.get('status')}: {result.get('error') or result.get('fetch_status')}", flush=True)

    write_outputs(results, args)
    print(f"wrote outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
