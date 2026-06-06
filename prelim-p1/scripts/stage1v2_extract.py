#!/usr/bin/env python3
"""Stage 1 v2 latent extraction with larger trajectory split."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[2]
for rel in ("stable-worldmodel-readonly", "stable-pretraining-readonly", "le-wm"):
    path = str(ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import stable_pretraining as spt  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
import torch  # noqa: E402
from contact_fallback import contact_blocks_from_state  # noqa: E402
from torchvision.transforms import v2 as transforms  # noqa: E402


OUT_DIR = ROOT / "prelim-p1" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 4242
TARGET_EPISODES = 200
HISTORY_SIZE = 3
FRAMESKIP = 5
ENCODE_BATCH_FRAMES = 96


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def img_transform():
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
        ]
    )


def normalize_action_blocks(
    action_blocks: np.ndarray, action_mean: np.ndarray, action_std: np.ndarray
) -> np.ndarray:
    raw_dim = action_mean.shape[-1]
    blocks = action_blocks.reshape(action_blocks.shape[0], FRAMESKIP, raw_dim)
    normed = (blocks - action_mean.reshape(1, 1, raw_dim)) / action_std.reshape(1, 1, raw_dim)
    return normed.reshape(action_blocks.shape[0], FRAMESKIP * raw_dim).astype(np.float32)


def encode_pixels(model, pixels: torch.Tensor, device: torch.device) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for start in range(0, pixels.shape[0], ENCODE_BATCH_FRAMES):
            batch = pixels[start : start + ENCODE_BATCH_FRAMES].unsqueeze(0).to(device)
            info = model.encode({"pixels": batch})
            outputs.append(info["emb"].squeeze(0).detach().cpu())
    return torch.cat(outputs, dim=0).numpy().astype(np.float32)


def split_episodes(selected: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shuffled = selected.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, round(0.15 * n))
    n_val = max(1, round(0.15 * n))
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise RuntimeError("Not enough episodes for train/val/test split")
    train = np.sort(shuffled[:n_train]).astype(np.int64)
    val = np.sort(shuffled[n_train : n_train + n_val]).astype(np.int64)
    test = np.sort(shuffled[n_train + n_val :]).astype(np.int64)
    return train, val, test


def main() -> None:
    dataset_path = Path.home() / ".stable_worldmodel" / "datasets" / "pusht_expert_train.h5"
    object_ckpt = Path.home() / ".stable_worldmodel" / "pusht" / "lewm_object.ckpt"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing required dataset: {dataset_path}")
    if not object_ckpt.exists():
        raise FileNotFoundError(f"Missing required object checkpoint: {object_ckpt}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this machine.")

    device = torch.device("mps")
    model = swm.policy.AutoCostModel("pusht/lewm", cache_dir=swm.data.utils.get_cache_dir())
    model = model.to(device).eval()
    model.requires_grad_(False)

    pixel_transform = img_transform()
    dataset = swm.data.HDF5Dataset(
        "pusht_expert_train",
        frameskip=FRAMESKIP,
        num_steps=HISTORY_SIZE + 1,
        transform=lambda steps: {**steps, "pixels": pixel_transform(steps["pixels"])},
        keys_to_load=["pixels", "action", "state"],
        keys_to_cache=["action", "state"],
        cache_dir=swm.data.utils.get_cache_dir(),
    )

    raw_actions = dataset.get_col_data("action")
    finite_actions = raw_actions[~np.isnan(raw_actions).any(axis=1)]
    action_mean = finite_actions.mean(axis=0).astype(np.float32)
    action_std = finite_actions.std(axis=0, ddof=1).astype(np.float32)
    action_std = np.maximum(action_std, 1e-6)

    min_raw_len = (HISTORY_SIZE + 1) * FRAMESKIP
    valid = np.flatnonzero(dataset.lengths >= min_raw_len)
    num_episodes = min(TARGET_EPISODES, len(valid))
    if num_episodes < 20:
        raise RuntimeError(f"Only {num_episodes} usable episodes found; expected many more.")

    rng = np.random.default_rng(SEED)
    selected = np.sort(rng.choice(valid, size=num_episodes, replace=False)).astype(np.int64)
    train_eps, val_eps, test_eps = split_episodes(selected, rng)
    split_by_ep = {int(ep): 0 for ep in train_eps}
    split_by_ep.update({int(ep): 1 for ep in val_eps})
    split_by_ep.update({int(ep): 2 for ep in test_eps})
    split_names = {0: "train", 1: "val", 2: "test"}

    latent_chunks = []
    action_chunks = []
    target_chunks = []
    contact_chunks = []
    episode_chunks = []
    model_step_chunks = []
    split_chunks = []
    trajectory_records = []

    start_time = time.perf_counter()
    with h5py.File(dataset_path, "r", swmr=True) as h5:
        has_n_contacts = "n_contacts" in h5
        for idx, ep in enumerate(selected, start=1):
            ep = int(ep)
            raw_start = int(dataset.offsets[ep])
            raw_len = int(dataset.lengths[ep])
            model_steps = 1 + (raw_len - 1) // FRAMESKIP
            obs_rows = raw_start + np.arange(model_steps, dtype=np.int64) * FRAMESKIP
            action_rows = (model_steps - 1) * FRAMESKIP

            episode = dataset.load_episode(ep)
            pixels = episode["pixels"]
            actions_raw = np.asarray(h5["action"][raw_start : raw_start + action_rows], dtype=np.float32)
            action_blocks_raw = actions_raw.reshape(model_steps - 1, -1)
            action_blocks = normalize_action_blocks(action_blocks_raw, action_mean, action_std)

            states = np.asarray(h5["state"][obs_rows], dtype=np.float32)
            contact_blocks = contact_blocks_from_state(states, FRAMESKIP)
            contact_source = "geometry-from-state"
            if has_n_contacts:
                contact_source = "n_contacts-present-but-unused"

            latents = encode_pixels(model, pixels, device)
            if latents.shape[0] != model_steps:
                raise RuntimeError(
                    f"Episode {ep} latent length mismatch: got {latents.shape[0]}, expected {model_steps}"
                )

            n_pairs = model_steps - HISTORY_SIZE
            histories = np.stack([latents[i : i + HISTORY_SIZE] for i in range(n_pairs)], axis=0)
            action_histories = np.stack([action_blocks[i : i + HISTORY_SIZE] for i in range(n_pairs)], axis=0)
            targets = latents[HISTORY_SIZE:]
            pair_contacts = contact_blocks[HISTORY_SIZE - 1 : HISTORY_SIZE - 1 + n_pairs]
            if len(pair_contacts) < n_pairs:
                pair_contacts = np.pad(pair_contacts, (0, n_pairs - len(pair_contacts)), constant_values=0)

            split_code = split_by_ep[ep]
            latent_chunks.append(histories.astype(np.float32))
            action_chunks.append(action_histories.astype(np.float32))
            target_chunks.append(targets.astype(np.float32))
            contact_chunks.append(pair_contacts.astype(np.float32))
            episode_chunks.append(np.full(n_pairs, ep, dtype=np.int64))
            model_step_chunks.append(np.arange(HISTORY_SIZE, HISTORY_SIZE + n_pairs, dtype=np.int64))
            split_chunks.append(np.full(n_pairs, split_code, dtype=np.int64))
            trajectory_records.append(
                {
                    "episode_id": ep,
                    "split": split_names[split_code],
                    "raw_length": raw_len,
                    "model_steps": model_steps,
                    "pairs": n_pairs,
                    "latent_shape": list(latents.shape),
                    "action_block_shape": list(action_blocks.shape),
                    "contact_positive_blocks": int((contact_blocks > 0).sum()),
                    "contact_source": contact_source,
                }
            )
            if idx % 25 == 0:
                print(f"encoded {idx}/{num_episodes} episodes", flush=True)

    sync(device)
    elapsed = time.perf_counter() - start_time

    history = np.concatenate(latent_chunks, axis=0)
    actions = np.concatenate(action_chunks, axis=0)
    target = np.concatenate(target_chunks, axis=0)
    contact = np.concatenate(contact_chunks, axis=0)
    episode_id = np.concatenate(episode_chunks, axis=0)
    model_step = np.concatenate(model_step_chunks, axis=0)
    split = np.concatenate(split_chunks, axis=0)

    masks = {"train": split == 0, "val": split == 1, "test": split == 2}
    target_mean = target.mean(axis=0)
    target_std = target.std(axis=0, ddof=0)

    cache_path = OUT_DIR / "stage1v2_latents.npz"
    np.savez_compressed(
        cache_path,
        history=history,
        action=actions,
        target=target,
        contact=contact,
        episode_id=episode_id,
        model_step=model_step,
        split=split,
        train_episodes=train_eps,
        val_episodes=val_eps,
        test_episodes=test_eps,
        selected_episodes=selected,
        action_mean=action_mean,
        action_std=action_std,
        target_mean=target_mean.astype(np.float32),
        target_std=target_std.astype(np.float32),
    )

    summary = {
        "stage": "1v2_extraction",
        "seed": SEED,
        "target_episodes": TARGET_EPISODES,
        "used_episodes": int(num_episodes),
        "train_trajectory_count": int(len(train_eps)),
        "val_trajectory_count": int(len(val_eps)),
        "test_trajectory_count": int(len(test_eps)),
        "train_episodes": train_eps.tolist(),
        "val_episodes": val_eps.tolist(),
        "test_episodes": test_eps.tolist(),
        "total_pairs": int(history.shape[0]),
        "train_pairs": int(masks["train"].sum()),
        "val_pairs": int(masks["val"].sum()),
        "test_pairs": int(masks["test"].sum()),
        "history_shape": list(history.shape),
        "action_history_shape": list(actions.shape),
        "target_shape": list(target.shape),
        "latent_stats": {
            "target_per_dim_mean_mean": float(target_mean.mean()),
            "target_per_dim_mean_std": float(target_mean.std()),
            "target_per_dim_std_mean": float(target_std.mean()),
            "target_per_dim_std_min": float(target_std.min()),
            "target_per_dim_std_max": float(target_std.max()),
            "any_nan": bool(np.isnan(target).any() or np.isnan(history).any()),
        },
        "action_normalization": {
            "source": "le-wm/train.py get_column_normalizer(dataset, 'action', 'action') equivalent",
            "raw_action_mean": action_mean.tolist(),
            "raw_action_std": action_std.tolist(),
        },
        "indexing": {
            "frameskip": FRAMESKIP,
            "history_size": HISTORY_SIZE,
            "convention": "history latents/action blocks at model steps [i, i+1, i+2] target latent at [i+3]",
            "training_source": "le-wm/train.py: tgt_emb = emb[:, num_preds:] with num_preds=1; final predictor token is next latent",
        },
        "contact": {
            "source": "geometry-from-state",
            "positive_pairs": int((contact > 0).sum()),
            "non_contact_pairs": int((contact <= 0).sum()),
            "caveat": "HDF5 has no n_contacts; this uses the prior geometric fallback from 7D state.",
        },
        "elapsed_seconds": elapsed,
        "cache_path": str(cache_path),
        "trajectory_records": trajectory_records,
    }

    summary_path = OUT_DIR / "stage1v2_extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("STAGE 1 V2 EXTRACTION SUMMARY")
    print(
        f"episodes used: {num_episodes} "
        f"(train={len(train_eps)}, val={len(val_eps)}, test={len(test_eps)})"
    )
    print(
        f"pairs: train={int(masks['train'].sum())}, val={int(masks['val'].sum())}, "
        f"test={int(masks['test'].sum())}, total={history.shape[0]}"
    )
    print(f"history_shape: {tuple(history.shape)}")
    print(f"action_history_shape: {tuple(actions.shape)}")
    print(f"target_shape: {tuple(target.shape)}")
    print(
        "latent target stats: "
        f"mean(mean_dim)={target_mean.mean():.6f}, std(mean_dim)={target_mean.std():.6f}, "
        f"mean(std_dim)={target_std.mean():.6f}, min(std_dim)={target_std.min():.6f}, "
        f"max(std_dim)={target_std.max():.6f}, any_nan={summary['latent_stats']['any_nan']}"
    )
    print(
        f"contact labels: source=geometry-from-state, contact_pairs={int((contact > 0).sum())}, "
        f"non_contact_pairs={int((contact <= 0).sum())}"
    )
    print(f"wrote: {cache_path}")
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()
