# harness/ — shared IDM readout (built once, used by all three rungs)

The probe, the gap_test, and curve's curve-3 head all read actions off encoder
latents with the SAME inverse-dynamics module: (z_t, z_{t+1}) → â_t.

Reuse before rebuilding:
- le-wm/inverse_dynamics.py — check whether this IS the IDM; wrap it if so.
- prelim-p1/scripts/stage1v2_extract.py — latent-extraction plumbing is reusable;
  only the ENCODER SOURCE changes (frozen-released → A/B/C-trained).

Files to land here:
  latents.py      — load encoder, extract (z_t, z_{t+1}) over a split
  idm_readout.py  — train/eval the IDM probe; report action-decoding error
  metrics.py      — per-seed aggregation, A-vs-B gap stats, chance baseline

## Released cube checkpoint sanity check

Script:

```bash
cd /Users/rishisim/Documents/research/lewm-p1-inversedynamics
le-wm/.venv/bin/python le-wm/sanity_check_lewm_cube.py
```

Verified on 2026-06-17 against `quentinll/lewm-cube` at HF commit
`b0747c5002e86d2ce8f3cd8178004b97524c587d`.

Result:

- `config.json` and `weights.pt` download under
  `le-wm/.cache/sanity_cube_model/`.
- The model loads with `strict=True` after applying the same legacy ViT key
  remap that this repo already carries in `train.py`.
- Real reset pixels from `swm/OGBCube-v0` encode successfully.
- `missing_keys: []`
- `unexpected_keys: []`
- `latent_shape: (2, 1, 192)`
- `encoder_only_matches_model_encode: True`

### Input image format

The cube environment emits raw pixels as uint8 RGB in `B,T,H,W,C` layout:

```text
(2, 1, 224, 224, 3), dtype uint8, range [1, 255]
```

The released LeWM encoder expects float tensors in `B,T,C,H,W` layout:

```text
(B, T, 3, 224, 224), dtype float32
```

Preprocessing matches `eval.py`:

1. Convert uint8 pixels to float by dividing by 255.
2. Normalize with ImageNet mean/std:
   - mean: `[0.485, 0.456, 0.406]`
   - std: `[0.229, 0.224, 0.225]`
3. Use 224x224 RGB images. The cube reset pixels already arrive at 224x224
   when `swm.World(..., image_shape=(224, 224), width=224, height=224)` is used.

### Action space

The cube simulator's per-step action space is:

```text
Box(-1.0, 1.0, (5,), float32)
```

So a single env action is 5D with all components in `[-1, 1]`.

The released cube checkpoint config has:

```text
action_encoder.input_dim = 25
```

That is consistent with the cube eval config's `action_block: 5`: the world
model predictor/action encoder consumes five 5D env actions flattened into one
25D action token per latent transition.

### Encoder-only call

Use this when extracting frozen latents for the IDM readout. This path bypasses
the predictor and action encoder:

```python
with torch.inference_mode():
    flat_pixels = pixels.flatten(0, 1)  # B,T,C,H,W -> B*T,C,H,W
    cls = model.encoder(
        flat_pixels,
        interpolate_pos_encoding=True,
    ).last_hidden_state[:, 0]
    latents = model.projector(cls).reshape(B, T, -1)
```

This is equivalent to `model.encode({"pixels": pixels})["emb"]` when no
`action` key is supplied. `model.encode` also skips `model.action_encoder`
unless `info["action"]` exists.

### Files inspected

- `le-wm/jepa.py`: `JEPA.encode` flattens `B,T` pixels, runs the ViT encoder,
  takes the CLS token, applies `projector`, reshapes to `B,T,D`, and only runs
  `action_encoder` if an `action` key is present.
- `le-wm/module.py`: provides `MLP`, `Embedder`, and `ARPredictor`. For this
  sanity check, only `MLP` is needed after the ViT to produce the 192D latent.
- `le-wm/eval.py`: defines the image preprocessing and runtime policy path.
  Pixel preprocessing is ImageNet normalization after uint8-to-float scaling.
- `le-wm/config/eval/cube.yaml`: cube uses `swm/OGBCube-v0`, 224x224 images,
  `dataset_name: ogbench/cube_single_expert`, and `action_block: 5`.
- `le-wm/sanity_check_lewm_pusht.py`: reconstructs the released HF LeWM model
  from `config.json` and `weights.pt`; the cube script follows the same shape
  but uses strict loading with the repo's legacy ViT key remap.
- `le-wm/inverse_dynamics.py`: currently contains only the module docstring
  `"Proposal 1 inverse-dynamics head placeholder."` It is not yet an IDM
  implementation and is not directly reusable as a readout, except as the
  intended file location/name for the shared inverse-dynamics head.
