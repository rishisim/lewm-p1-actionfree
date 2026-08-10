#!/usr/bin/env python3
"""Build held-out readout transitions from a trained Cube LeWM encoder."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
LEWM_DIR = REPO / "le-wm"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(LEWM_DIR) not in sys.path:
    sys.path.insert(0, str(LEWM_DIR))

import hydra  # noqa: E402
from lewm_storage import p1_hdf5_dir, require_external_storage  # noqa: E402
from train import remap_legacy_vit_keys  # noqa: E402
from utils import get_img_preprocessor  # noqa: E402

require_external_storage()

DEFAULT_HDF5 = p1_hdf5_dir() / "visual_cube_single_play_readout_heldout.h5"
DEFAULT_SPLIT = ROOT / "gap_test" / "outputs" / "heldout_readout_split_seed2026.json"
DEFAULT_OUT = ROOT / "gap_test" / "outputs" / "arm_a_conditioned" / "arm_a_encoder_transitions.npz"


def as_builtin(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def load_model_config(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    cfg = OmegaConf.load(path)
    model_cfg = OmegaConf.select(cfg, "model") or cfg
    return OmegaConf.to_container(model_cfg, resolve=True)


def load_model(config_path: Path, weights_path: Path, device: torch.device) -> torch.nn.Module:
    model_cfg = load_model_config(config_path)
    model = hydra.utils.instantiate(model_cfg)
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        if not any(key.startswith("encoder.encoder.layer.") for key in state_dict):
            raise exc
        model.load_state_dict(remap_legacy_vit_keys(state_dict))
    model.to(device)
    model.eval()
    return model


def choose_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def load_split(path: Path) -> dict[str, np.ndarray]:
    split = json.loads(path.read_text(encoding="utf-8"))
    episodes = split["episodes"]
    return {
        name: np.asarray(episodes[name], dtype=np.int64)
        for name in ("train", "val", "test")
    }


def preprocess_pixels(raw: np.ndarray, transform: Any, device: torch.device) -> torch.Tensor:
    pixels = torch.from_numpy(raw).permute(0, 3, 1, 2).contiguous()
    pixels = transform({"pixels": pixels})["pixels"]
    return pixels.to(device, non_blocking=True)


@torch.inference_mode()
def encode_pixels(
    model: torch.nn.Module,
    h5: h5py.File,
    *,
    device: torch.device,
    batch_size: int,
    img_size: int,
) -> np.ndarray:
    pixels_ds = h5["pixels"]
    n_rows = int(pixels_ds.shape[0])
    transform = get_img_preprocessor(source="pixels", target="pixels", img_size=img_size)
    chunks: list[np.ndarray] = []

    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        pixels = preprocess_pixels(pixels_ds[start:end], transform, device)
        cls = model.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        latent = model.projector(cls).detach().float().cpu().numpy()
        chunks.append(latent.astype(np.float32, copy=False))
        if start == 0 or end == n_rows or (end // batch_size) % 20 == 0:
            print(f"encoded frames {end}/{n_rows}", flush=True)

    return np.concatenate(chunks, axis=0)


def build_transition_cache(
    hdf5_path: Path,
    split_path: Path,
    config_path: Path,
    weights_path: Path,
    out_path: Path,
    summary_path: Path,
    *,
    batch_size: int,
    img_size: int,
    device_name: str,
    encoder_label: str,
) -> None:
    device = choose_device(device_name)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model = load_model(config_path, weights_path, device)
    split = load_split(split_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "r") as h5:
        ep_len = h5["ep_len"][:].astype(np.int64)
        ep_offset = h5["ep_offset"][:].astype(np.int64)
        if len(ep_len) != 100:
            raise ValueError(f"Expected 100 held-out episodes, found {len(ep_len)}")
        if not np.all(ep_len == ep_len[0]):
            raise ValueError("Expected fixed-length held-out trajectories.")

        latents = encode_pixels(model, h5, device=device, batch_size=batch_size, img_size=img_size)
        action_dim = int(h5["action"].shape[1])
        transitions = int(np.sum(ep_len - 1))
        latent_dim = int(latents.shape[1])

        obs = np.empty((transitions, latent_dim), dtype=np.float32)
        next_obs = np.empty((transitions, latent_dim), dtype=np.float32)
        action = np.empty((transitions, action_dim), dtype=np.float32)
        episode_id = np.empty((transitions,), dtype=np.int64)
        step_in_episode = np.empty((transitions,), dtype=np.int64)
        source_row = np.empty((transitions,), dtype=np.int64)

        cursor = 0
        for ep, (offset, length) in enumerate(zip(ep_offset, ep_len)):
            n = int(length - 1)
            rows = slice(int(offset), int(offset + n))
            dst = slice(cursor, cursor + n)
            obs[dst] = latents[rows]
            next_obs[dst] = latents[int(offset) + 1 : int(offset) + int(length)]
            action[dst] = h5["action"][rows]
            episode_id[dst] = ep
            step_in_episode[dst] = np.arange(n, dtype=np.int64)
            source_row[dst] = np.arange(int(offset), int(offset + n), dtype=np.int64)
            cursor += n

    split_indices = {
        name: np.flatnonzero(np.isin(episode_id, episodes)).astype(np.int64)
        for name, episodes in split.items()
    }

    np.savez_compressed(
        out_path,
        obs=obs,
        next_obs=next_obs,
        action=action,
        episode_id=episode_id,
        step_in_episode=step_in_episode,
        source_row=source_row,
        train_idx=split_indices["train"],
        val_idx=split_indices["val"],
        test_idx=split_indices["test"],
        train_episodes=split["train"],
        val_episodes=split["val"],
        test_episodes=split["test"],
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hdf5_path": hdf5_path,
        "split_path": split_path,
        "config_path": config_path,
        "weights_path": weights_path,
        "output_npz": out_path,
        "latent_dim": latent_dim,
        "action_dim": action_dim,
        "frames_encoded": int(latents.shape[0]),
        "transitions": transitions,
        "episodes": int(len(ep_len)),
        "episode_length": int(ep_len[0]),
        "splits": {
            name: {
                "episodes": split[name].astype(int).tolist(),
                "transitions": int(len(split_indices[name])),
            }
            for name in ("train", "val", "test")
        },
        "encoder": {
            "model": encoder_label,
            "encoder_only_path": "encoder CLS token -> projector; no action_encoder or predictor",
            "img_size": img_size,
            "batch_size": batch_size,
            "device": str(device),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=as_builtin) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"wrote {summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, default=DEFAULT_HDF5)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--encoder-label", default="Cube LeWM Arm A action-conditioned encoder")
    args = parser.parse_args()

    summary = args.summary or args.out.with_name(args.out.stem + "_summary.json")
    build_transition_cache(
        args.hdf5,
        args.split,
        args.config,
        args.weights,
        args.out,
        summary,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device_name=args.device,
        encoder_label=args.encoder_label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
