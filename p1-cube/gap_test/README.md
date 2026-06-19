# p1-cube gap test

This note records the cube LeWM data path for the action-conditioned vs
action-free encoder comparison. The pretraining data is OGBench visual cube
play data, not expert data.

## Play HDF5 split created

Source files were already cached locally under `p1-cube/data/cache/ogbench/`.
Nothing was re-downloaded and no Hugging Face auth was needed.

Used source files:

- `visual-cube-single-play-v0.npz`
- `visual-cube-single-play-v0-val.npz`

I treated these as the two local shards of the `visual-cube-single-play-v0`
family. The main shard is the encoder pretraining pool. The val shard is the
held-out inverse-dynamics readout set and must not be used for encoder
pretraining.

Full local source counts:

| source | episodes | steps | episode length |
| --- | ---: | ---: | ---: |
| `visual-cube-single-play-v0.npz` | 1000 | 1,001,000 | 1001 |
| `visual-cube-single-play-v0-val.npz` | 100 | 100,100 | 1001 |
| total local play family | 1100 | 1,101,100 | 1001 |

Derived HDF5 outputs:

| split | HDF5 | episodes | steps |
| --- | --- | ---: | ---: |
| encoder pretrain pool | `p1-cube/data/lewm_hdf5/visual_cube_single_play_pretrain_pool.h5` | 1000 | 1,001,000 |
| held-out readout set | `p1-cube/data/lewm_hdf5/visual_cube_single_play_readout_heldout.h5` | 100 | 100,100 |

Manifest:

```text
p1-cube/data/lewm_hdf5/visual_cube_single_play_split_manifest.json
```

Disk size:

```text
12G  p1-cube/data/lewm_hdf5/visual_cube_single_play_pretrain_pool.h5
1.2G p1-cube/data/lewm_hdf5/visual_cube_single_play_readout_heldout.h5
13G  p1-cube/data/lewm_hdf5/
```

## HDF5 layout

Both derived files match the HDF5 layout that `train.py` and
`stable_worldmodel.data.HDF5Dataset` expect:

```text
ep_len       int64 episode lengths
ep_offset    int64 global row offsets
pixels       uint8 RGB frames, shape (N, 64, 64, 3), copied from NPZ observations
action       float32 env actions, shape (N, 5)
observation  float32 concat(qpos, qvel), shape (N, 41)
proprio      float32 concat(qpos, qvel), shape (N, 41)
qpos         float32, shape (N, 21)
qvel         float32, shape (N, 20)
terminal     bool, shape (N,)
```

Loader verification with `num_steps=4` and `frameskip=5`:

| split | HDF5 rows | episodes | LeWM windows | raw action dim | sample action |
| --- | ---: | ---: | ---: | ---: | --- |
| pretrain pool | 1,001,000 | 1000 | 982,000 | 5 | `(4, 25)` |
| readout heldout | 100,100 | 100 | 98,200 | 5 | `(4, 25)` |

The `(4, 25)` action shape is correct: each sample has
`num_steps = history_size + num_preds = 4`, and each step receives
`frameskip * env_action_dim = 5 * 5 = 25` action values.

## Disjointness check

The held-out readout set contains 100 whole trajectories, which is comfortably
in the "tens of episodes" target. The split is trajectory-disjoint by
`(source_file, source_episode_idx)`:

```text
pretrain source: visual-cube-single-play-v0.npz, episodes 0..999
readout source:  visual-cube-single-play-v0-val.npz, episodes 0..99
intersection_count: 0
trajectory_disjoint: true
```

The readout trajectories are not present in the pretraining HDF5.

## Train a play-data cube LeWM

Do not use the default expert dataset name from `config/train/data/ogb.yaml`
for this experiment. Override `data.dataset.name` to the play pretraining pool:

```bash
cd /Users/rishisim/Documents/research/lewm-p1-inversedynamics/le-wm

.venv/bin/python train.py \
  data=ogb \
  data.dataset.name=/Users/rishisim/Documents/research/lewm-p1-inversedynamics/p1-cube/data/lewm_hdf5/visual_cube_single_play_pretrain_pool.h5 \
  output_model_name=lewm_cube_play_action_conditioned \
  subdir=lewm_cube_play_action_conditioned
```

For the action-free companion run, keep every other knob identical and set only
the action toggle plus output names:

```bash
.venv/bin/python train.py \
  data=ogb \
  data.dataset.name=/Users/rishisim/Documents/research/lewm-p1-inversedynamics/p1-cube/data/lewm_hdf5/visual_cube_single_play_pretrain_pool.h5 \
  action_free=true \
  output_model_name=lewm_cube_play_action_free \
  subdir=lewm_cube_play_action_free
```

The held-out readout file should be used only later for inverse-dynamics readout
training:

```text
/Users/rishisim/Documents/research/lewm-p1-inversedynamics/p1-cube/data/lewm_hdf5/visual_cube_single_play_readout_heldout.h5
```

## Fixed training knobs

The base training config fixes the comparison-critical knobs:

- `seed: 3072`
- `history_size: 3`
- `num_preds: 1`
- `trainer.max_epochs: 100`
- `trainer.accelerator: gpu`
- `trainer.precision: bf16`
- `loader.batch_size: 128`
- `optimizer: AdamW, lr=5e-5, weight_decay=1e-3`
- scheduler: `LinearWarmupCosineAnnealingLR`, configured in `train.py`
- loss: prediction MSE plus `0.09 * SIGReg(knots=17, num_proj=1024)`

The pretrain HDF5 has 982,000 valid LeWM windows. With `train_split: 0.9`,
`loader.batch_size: 128`, and `drop_last=True`, the default training loop will
use about 883,800 train windows and 6,904 optimizer steps per epoch. At
`max_epochs: 100`, that is about 690,400 optimizer steps. Runtime should be
estimated from the first real epoch on the target single GPU.

## Action path

The action enters only through the predictor conditioning path:

1. `train.py` loads `batch["action"]` from the dataset and replaces NaNs with
   zero.
2. `JEPA.encode` receives the batch. It encodes `pixels` with the ViT encoder
   and separately calls `self.action_encoder(info["action"])`.
3. The HDF5 dataset uses `frameskip: 5`; it does not subsample `action`, then
   reshapes the action slice to `(num_steps, -1)`. With cube's 5D action and
   `num_steps=4`, one sample's action tensor is `(4, 25)`.
4. `train.py` sets `cfg.model.action_encoder.input_dim =
   cfg.data.dataset.frameskip * dataset.get_dim("action")`, so cube gets
   `5 * 5 = 25`.
5. `module.Embedder(input_dim=25, emb_dim=192)` maps `(B, T, 25)` to
   `(B, T, 192)` using a `Conv1d(25 -> 10, kernel_size=1)` followed by an MLP.
6. `lejepa_forward` slices `ctx_act = act_emb[:, :history_size]`.
7. `JEPA.predict(ctx_emb, ctx_act)` passes that action embedding into
   `ARPredictor.forward(x, c)`.
8. `ARPredictor` calls `Transformer(..., block_class=ConditionalBlock)`, and
   each `ConditionalBlock` uses `c` through AdaLN-zero modulation
   (`shift/scale/gate` for attention and MLP).

## Clean action-free toggle

The cleanest single toggle is already present in `train.py`:

```python
ctx_act = act_emb[:, :ctx_len]

if cfg.get("action_free", False):
    ctx_act = torch.zeros_like(ctx_act)

pred_emb = self.model.predict(ctx_emb, ctx_act)
```

Recommendation: keep the architecture identical and zero `ctx_act` at this
point, instead of removing the action pathway. This preserves parameter count,
module structure, optimizer groups, predictor shape, checkpoint compatibility,
and Hydra config shape. It ablates action information at the final predictor
conditioning input while still exercising the same action encoder code path.
Removing the pathway would require changing `JEPA.predict`, `ARPredictor`, and
the conditional transformer interface, making the comparison less isolated.

## Smoke train evidence

Earlier smoke training used a tiny temporary HDF5 fixture and confirmed the
training loop runs on cube-format HDF5 data. That run was intentionally not a
real training run.

Smoke checkpoint:

```text
p1-cube/gap_test/outputs/smoke_cache/checkpoints/lewm_cube_smoke/weights_epoch_1.pt
```

The real run should use the full play pretrain pool above, not the smoke file
and not the expert config default.

## Arm A preflight, 2026-06-17

Requested Arm A is one action-conditioned cube LeWM trained on:

```text
/Users/rishisim/Documents/research/lewm-p1-inversedynamics/p1-cube/data/lewm_hdf5/visual_cube_single_play_pretrain_pool.h5
```

Status: not completed locally. The full HDF5 loads, the model runs, gradients
reach all parameters, and bf16 MPS works, but the measured local throughput makes
a convergence run impractical on this Mac.

Evidence from the real file:

- fp32, batch 128, configured worker loader: validation ran and first optimizer
  step completed, but progress was only `1/4` tiny batches after about 3
  minutes; I interrupted it before wasting hours.
- bf16, batch 128, `num_workers=0`, two epochs with two train batches per epoch
  completed successfully.
- bf16 batch-128 timing run id:
  `/Users/rishisim/.cache/stable-pretraining/runs/20260617/143536/d086821b7685`
- timing-run metrics from four optimizer steps:

```text
step  fit/pred_loss  fit/sigreg_loss
0     0.2362         40.25
1     0.2330         40.25
2     0.2641         34.00
3     0.2794         32.50
```

This is not plateaued. It is only a real-file loader/optimizer smoke. With the
full pretrain pool, batch 128 gives about 6,904 optimizer steps per epoch after
the 0.9 train split. The measured bf16 batch-128 MPS timing was roughly 0.2 to
0.25 optimizer steps/sec for this tiny run, implying about 8 hours per epoch and
many days to weeks for a generous convergence budget. Run the real Arm A on a
CUDA GPU before launching the 10-run matrix.

No readout metrics were produced from Arm A, because there is no converged Arm A
encoder yet.

## Fixed held-out readout split

The readout split is independent of the encoder and is fixed here for all future
Gap arms. It uses only the 100-episode held-out file:

```text
/Users/rishisim/Documents/research/lewm-p1-inversedynamics/p1-cube/data/lewm_hdf5/visual_cube_single_play_readout_heldout.h5
```

Split file:

```text
p1-cube/gap_test/outputs/heldout_readout_split_seed2026.json
```

Rule: shuffle source episode ids `0..99` with `numpy.default_rng(2026)`, then
take 70 train, 15 val, and 15 test trajectories. Episode ids are sorted within
each split for readability.

```text
train episodes, n=70:
1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 22,
25, 28, 29, 30, 31, 32, 33, 35, 37, 38, 39, 41, 43, 44, 45, 46,
47, 48, 51, 52, 55, 57, 58, 59, 60, 61, 62, 64, 67, 68, 69, 70,
72, 73, 74, 75, 76, 77, 80, 81, 82, 83, 84, 85, 87, 88, 89, 91,
94, 96, 97, 98

val episodes, n=15:
0, 20, 21, 24, 27, 36, 50, 53, 54, 56, 65, 66, 86, 90, 99

test episodes, n=15:
4, 13, 23, 26, 34, 40, 42, 49, 63, 71, 78, 79, 92, 93, 95
```

This split must be reused unchanged for every future encoder so only the encoder
varies across Gap arms.

<!-- gap-test-overnight-results:start -->

## Overnight Vast 8x H100 results

### Arm A action-conditioned

- Local verified bundle: `p1-cube/gap_test/outputs/vast_8xh100_guard/arm_a_20260618T074538Z`
- Training epochs in CSV: `100`
- Fit prediction plateaued: `True`; last `0.032729`
- Fit SIGReg plateaued: `True`; at floor `False`; last `2.468750`
- Validation prediction plateaued: `False`; validation SIGReg plateaued: `False`
- Test R2 all dims: `0.686136`; excluding action_3: `0.878066`
- Test normalized MSE all dims: readout `0.302609` vs mean baseline `0.981831`
- Test normalized MSE excluding action_3: readout `0.121376` vs mean baseline `0.989777`
- Probe comparison: Probe all-dim R2 `0.435415`, delta `0.250721`; Probe excluding-action_3 R2 `0.712108`, delta `0.165959`; large discrepancy excluding action_3 `True`
- action_3 separately: R2 `-0.081587`, normalized MSE `1.027544`

| dim | R2 | normalized MSE |
| --- | ---: | ---: |
| action_0 | 0.836812 | 0.156913 |
| action_1 | 0.946565 | 0.053025 |
| action_2 | 0.950268 | 0.048761 |
| action_3 | -0.081587 | 1.027544 |
| action_4 | 0.778620 | 0.226804 |

<!-- gap-test-overnight-results:end -->
