#!/usr/bin/env python3
import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import ViTConfig, ViTModel

import stable_pretraining as spt
import stable_worldmodel as swm
from jepa import JEPA
from module import ARPredictor, Embedder, MLP


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


REPO_ID = "quentinll/lewm-cube"
BATCH_SIZE = 2


def strip_target(d):
    return {k: v for k, v in d.items() if k != "_target_"}


def vit_hf_from_config(
    size="tiny",
    patch_size=16,
    image_size=224,
    pretrained=False,
    use_mask_token=True,
    **kwargs,
):
    size_configs = {
        "tiny": {"hidden_size": 192, "num_hidden_layers": 12, "num_attention_heads": 3},
        "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6},
        "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12},
        "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16},
        "huge": {"hidden_size": 1280, "num_hidden_layers": 32, "num_attention_heads": 16},
    }
    config_params = dict(size_configs[size])
    config_params["intermediate_size"] = config_params["hidden_size"] * 4
    config_params["image_size"] = image_size
    config_params["patch_size"] = patch_size
    config_params.update(kwargs)
    if pretrained:
        model = ViTModel.from_pretrained(
            f"google/vit-{size}-patch{patch_size}-{image_size}",
            add_pooling_layer=False,
            use_mask_token=use_mask_token,
        )
    else:
        model = ViTModel(
            ViTConfig(**config_params),
            add_pooling_layer=False,
            use_mask_token=use_mask_token,
        )
    model.config.interpolate_pos_encoding = True
    return model


def mlp_from_config(cfg, key):
    norm_target = cfg[key]["norm_fn"]["_target_"]
    if norm_target.endswith("BatchNorm1d"):
        norm_fn = torch.nn.BatchNorm1d
    elif norm_target.endswith("LayerNorm"):
        norm_fn = torch.nn.LayerNorm
    else:
        raise ValueError(f"Unsupported norm_fn in {key}: {norm_target}")

    params = {
        k: v for k, v in cfg[key].items() if k not in {"_target_", "norm_fn"}
    }
    return MLP(norm_fn=norm_fn, **params)


def remap_legacy_vit_keys(state_dict):
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("encoder.encoder.layer."):
            new_key = key.replace("encoder.encoder.layer.", "encoder.layers.", 1)
            new_key = new_key.replace(".attention.attention.query.", ".attention.q_proj.")
            new_key = new_key.replace(".attention.attention.key.", ".attention.k_proj.")
            new_key = new_key.replace(".attention.attention.value.", ".attention.v_proj.")
            new_key = new_key.replace(".attention.output.dense.", ".attention.o_proj.")
            new_key = new_key.replace(".intermediate.dense.", ".mlp.fc1.")
            new_key = new_key.replace(".output.dense.", ".mlp.fc2.")
        remapped[new_key] = value
    return remapped


def load_released_cube_model(root):
    local_dir = root / ".cache" / "sanity_cube_model"
    cfg_path = Path(
        hf_hub_download(REPO_ID, "config.json", local_dir=local_dir)
    )
    weights_path = Path(
        hf_hub_download(REPO_ID, "weights.pt", local_dir=local_dir)
    )

    cfg = json.loads(cfg_path.read_text())
    model = JEPA(
        encoder=vit_hf_from_config(**strip_target(cfg["encoder"])),
        predictor=ARPredictor(**strip_target(cfg["predictor"])),
        action_encoder=Embedder(**strip_target(cfg["action_encoder"])),
        projector=mlp_from_config(cfg, "projector"),
        pred_proj=mlp_from_config(cfg, "pred_proj"),
    )

    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    state = remap_legacy_vit_keys(state)
    result = model.load_state_dict(state, strict=True)
    return model, cfg, cfg_path, weights_path, result


def cube_pixels_and_action_space(batch_size, image_size):
    world = swm.World(
        env_name="swm/OGBCube-v0",
        num_envs=batch_size,
        max_episode_steps=50,
        env_type="single",
        ob_type="states",
        multiview=False,
        width=image_size,
        height=image_size,
        visualize_info=False,
        terminate_at_goal=True,
        image_shape=(image_size, image_size),
    )
    try:
        world.reset(seed=42)
        pixels = world.infos["pixels"].copy()
        action_space = world.envs.single_action_space
    finally:
        world.close()

    return pixels, action_space


def normalize_pixels(raw_pixels):
    pixels = torch.from_numpy(raw_pixels)
    pixels = pixels.permute(0, 1, 4, 2, 3).contiguous()
    pixels = pixels.float().div(255.0)

    mean = torch.tensor(spt.data.dataset_stats.ImageNet["mean"]).view(1, 1, 3, 1, 1)
    std = torch.tensor(spt.data.dataset_stats.ImageNet["std"]).view(1, 1, 3, 1, 1)
    return (pixels - mean) / std


def main():
    root = Path(__file__).resolve().parent
    model, cfg, cfg_path, weights_path, result = load_released_cube_model(root)
    model.eval().requires_grad_(False)

    raw_pixels, action_space = cube_pixels_and_action_space(
        BATCH_SIZE, cfg["encoder"]["image_size"]
    )
    pixels = normalize_pixels(raw_pixels)

    with torch.inference_mode():
        encoded = model.encode({"pixels": pixels})["emb"]

        flat_pixels = pixels.flatten(0, 1)
        cls = model.encoder(
            flat_pixels,
            interpolate_pos_encoding=True,
        ).last_hidden_state[:, 0]
        encoder_only_latents = model.projector(cls).reshape(
            pixels.shape[0], pixels.shape[1], -1
        )

    print(f"repo_id: {REPO_ID}")
    print(f"config_path: {cfg_path}")
    print(f"weights_path: {weights_path}")
    print(f"missing_keys: {result.missing_keys}")
    print(f"unexpected_keys: {result.unexpected_keys}")
    print("expected_input_image_format:")
    print(f"  raw_cube_pixels: shape {tuple(raw_pixels.shape)}, dtype {raw_pixels.dtype}, layout B,T,H,W,C, range [{raw_pixels.min()}, {raw_pixels.max()}]")
    print(f"  model_pixels: shape {tuple(pixels.shape)}, dtype {pixels.dtype}, layout B,T,C,H,W")
    print(f"  resolution: {cfg['encoder']['image_size']}x{cfg['encoder']['image_size']}")
    print("  channels: 3 RGB")
    print("  normalization: uint8/255.0, then ImageNet mean/std")
    print(f"  imagenet_mean: {spt.data.dataset_stats.ImageNet['mean']}")
    print(f"  imagenet_std: {spt.data.dataset_stats.ImageNet['std']}")
    print(f"latent_shape: {tuple(encoded.shape)}")
    print(f"encoder_only_latent_shape: {tuple(encoder_only_latents.shape)}")
    print(f"encoder_only_matches_model_encode: {torch.allclose(encoded, encoder_only_latents, atol=1e-6)}")
    print("action_space:")
    print(f"  env_step_dim: {action_space.shape[0]}")
    print(f"  env_step_low: {action_space.low.tolist()}")
    print(f"  env_step_high: {action_space.high.tolist()}")
    print(f"  model_action_encoder_input_dim: {cfg['action_encoder']['input_dim']}")
    print("  note: cube eval uses action_block=5, so the predictor action input is 5 env actions flattened to 25 values.")
    print("encoder_only_call:")
    print("  flat_pixels = pixels.flatten(0, 1)  # B,T,C,H,W -> B*T,C,H,W")
    print("  cls = model.encoder(flat_pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]")
    print("  latents = model.projector(cls).reshape(B, T, -1)")


if __name__ == "__main__":
    main()
