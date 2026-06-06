#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import ViTConfig, ViTModel

from jepa import JEPA
from module import ARPredictor, Embedder, MLP

def strip(d): return {k: v for k, v in d.items() if k != "_target_"}
def vit_hf_from_config(size="tiny", patch_size=16, image_size=224, pretrained=False, use_mask_token=True, **kwargs):
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
        model = ViTModel(ViTConfig(**config_params), add_pooling_layer=False, use_mask_token=use_mask_token)
    model.config.interpolate_pos_encoding = True
    return model

root = Path(__file__).resolve().parent
cfg_path = Path(hf_hub_download("quentinll/lewm-pusht", "config.json", local_dir=root / ".cache" / "sanity_model"))
w_path = Path(hf_hub_download("quentinll/lewm-pusht", "weights.pt", local_dir=root / ".cache" / "sanity_model"))
cfg = json.loads(cfg_path.read_text())
encoder = vit_hf_from_config(**strip(cfg["encoder"]))
norm = torch.nn.BatchNorm1d if cfg["projector"]["norm_fn"]["_target_"].endswith("BatchNorm1d") else torch.nn.LayerNorm
mlp = lambda k: MLP(norm_fn=norm, **strip({x: y for x, y in cfg[k].items() if x != "norm_fn"}))
model = JEPA(encoder, ARPredictor(**strip(cfg["predictor"])), Embedder(**strip(cfg["action_encoder"])), mlp("projector"), mlp("pred_proj"))
state = torch.load(w_path, map_location="cpu")
result = model.load_state_dict(state, strict=False)
print("missing_keys", result.missing_keys)
print("unexpected_keys", result.unexpected_keys)
print("parameter_count", sum(p.numel() for p in model.parameters()))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.eval().requires_grad_(False).to(device)
print("moved_to", device)
with torch.inference_mode():
    latent = model.encode({"pixels": torch.rand(1, 1, 3, 224, 224, device=device)})["emb"]
print("latent_shape", tuple(latent.shape))
