#!/usr/bin/env python3
"""Stage 0 timing canary for Proposal 1.

This script intentionally writes only under prelim-p1/outputs and reads the
existing LeWM/checkpoint/dataset assets in place.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for rel in ("stable-worldmodel-readonly", "stable-pretraining-readonly", "le-wm"):
    path = str(ROOT / rel)
    if path not in sys.path:
        sys.path.insert(0, path)

import stable_pretraining as spt  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
import torch  # noqa: E402
from lewm_storage import require_external_storage  # noqa: E402
from torchvision.transforms import v2 as transforms  # noqa: E402


OUT_DIR = ROOT / "prelim-p1" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def time_call(fn, *, warmup: int = 2, repeat: int = 5, device: torch.device) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - start)
    return sum(times) / len(times), times


def img_transform():
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=224),
        ]
    )


def main() -> None:
    stablewm_root = require_external_storage()
    dataset_path = stablewm_root / "datasets" / "pusht_expert_train.h5"
    object_ckpt = stablewm_root / "pusht" / "lewm_object.ckpt"
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
    param_count = sum(p.numel() for p in model.parameters())

    dataset = swm.data.HDF5Dataset(
        "pusht_expert_train",
        frameskip=1,
        num_steps=8,
        transform=lambda steps: {**steps, "pixels": img_transform()(steps["pixels"])},
        keys_to_load=["pixels", "action", "state"],
        keys_to_cache=["action", "state"],
        cache_dir=swm.data.utils.get_cache_dir(),
    )
    sample = dataset[0]
    pixels = sample["pixels"].unsqueeze(0).to(device, non_blocking=True)

    with torch.no_grad():
        encoded = model.encode({"pixels": pixels})
    emb_shape = tuple(encoded["emb"].shape)

    with torch.no_grad():
        encode_mean, encode_times = time_call(
            lambda: model.encode({"pixels": pixels}),
            warmup=2,
            repeat=5,
            device=device,
        )

    # Predictor-only timing. The full model's prediction call includes pred_proj;
    # the action encoder is timed separately from the predictor pass.
    emb = torch.randn(1, 3, 192, device=device)
    action_block = torch.randn(1, 3, 10, device=device)
    with torch.no_grad():
        act_emb = model.action_encoder(action_block)
        pred = model.predict(emb, act_emb)
    pred_shape = tuple(pred.shape)
    act_emb_shape = tuple(act_emb.shape)

    with torch.no_grad():
        pred_fwd_mean, pred_fwd_times = time_call(
            lambda: model.predict(emb, act_emb),
            warmup=5,
            repeat=20,
            device=device,
        )

    # Freeze all model components except the predictor and pred projection for
    # the dummy optimizer step, then restore the original no-grad state.
    for p in model.parameters():
        p.requires_grad_(False)
    train_params = list(model.predictor.parameters()) + list(model.pred_proj.parameters())
    for p in train_params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(train_params, lr=1e-4)

    def train_step() -> torch.Tensor:
        opt.zero_grad(set_to_none=True)
        local_emb = torch.randn(1, 3, 192, device=device)
        local_act = torch.randn(1, 3, 10, device=device)
        local_act_emb = model.action_encoder(local_act).detach()
        out = model.predict(local_emb, local_act_emb)
        loss = out.square().mean()
        loss.backward()
        opt.step()
        return loss.detach()

    train_mean, train_times = time_call(train_step, warmup=3, repeat=20, device=device)
    for p in train_params:
        p.requires_grad_(False)

    summary = {
        "load_ok": True,
        "device": str(device),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "param_count": param_count,
        "param_count_millions": param_count / 1_000_000,
        "assets": {
            "dataset": str(dataset_path),
            "object_checkpoint": str(object_ckpt),
        },
        "api_verification": {
            "encode_signature": "encode(info)",
            "predict_signature": "predict(emb, act_emb)",
            "rollout_convention": (
                "For history H=3, jepa.JEPA.rollout uses emb[:, -HS:] and act_emb[:, -HS:]; "
                "the last output of predict(history_emb, action_emb) is appended as the next latent. "
                "Thus a 3-latent/3-action window ending at t predicts the latent at t+1."
            ),
            "live_source_discrepancies": [
                "No discrepancy for the loaded object checkpoint: AutoCostModel('pusht/lewm') resolves to jepa.JEPA from le-wm/jepa.py.",
                "Note: stable-worldmodel-readonly also contains stable_worldmodel.wm.lewm.lewm.LeWM with the same one-step predict signature but a different rollout implementation; it is not the loaded object checkpoint class in this environment.",
            ],
        },
        "input_shapes": {
            "pixels": tuple(pixels.shape),
            "encoded_emb": emb_shape,
            "action_block": tuple(action_block.shape),
            "action_embedding": act_emb_shape,
            "prediction": pred_shape,
        },
        "timing": {
            "encode_mean_seconds": encode_mean,
            "encode_times_seconds": encode_times,
            "encode_ms_per_frame": encode_mean * 1000.0 / pixels.shape[1],
            "predictor_fwd_mean_seconds": pred_fwd_mean,
            "predictor_fwd_times_seconds": pred_fwd_times,
            "predictor_fwd_ms": pred_fwd_mean * 1000.0,
            "predictor_fwd_bwd_step_mean_seconds": train_mean,
            "predictor_fwd_bwd_step_times_seconds": train_times,
            "predictor_fwd_bwd_step_ms": train_mean * 1000.0,
        },
        "mps_fallback": {
            "env_pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "script_observed_warning_status": "Check terminal log for PyTorch MPS fallback warnings emitted outside Python.",
        },
    }

    out_path = OUT_DIR / "stage0_timing.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print("STAGE 0 SUMMARY")
    print(f"load_ok: {summary['load_ok']}")
    print(f"device: {summary['device']}")
    print(f"model_class: {summary['model_class']}")
    print(f"param_count: {param_count:,} ({param_count / 1_000_000:.2f}M)")
    print(f"pixels_shape: {tuple(pixels.shape)} -> emb_shape: {emb_shape}")
    print(f"encode_ms_per_frame: {summary['timing']['encode_ms_per_frame']:.3f}")
    print(f"predictor_fwd_ms: {summary['timing']['predictor_fwd_ms']:.3f}")
    print(f"predictor_fwd_bwd_step_ms: {summary['timing']['predictor_fwd_bwd_step_ms']:.3f}")
    print("mps_fallback: inspect terminal log; no Python-level fallback exception was raised")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
