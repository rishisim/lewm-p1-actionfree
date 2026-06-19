# probe/ - ceiling + harness smoke test (does NOT test P1)

Freeze the released LeWM cube encoder -> `harness/idm_readout.py` on a holdout
split. Outputs: the action-trained ceiling (high by construction) + proof the
pipeline runs. GATE: if the action-trained encoder cannot decode actions above
chance, the setup is broken; stop here.

Scripts: `run_probe.py`

Outputs:

- `outputs/probe_summary.json`
- `outputs/probe.log`
- `outputs/readout_best.pt`

## Result note - 2026-06-17

Input cache: `p1-cube/data/cube_single_play_probe_transitions.npz`.

Split:

| Split | Trajectories | Transitions |
|---|---:|---:|
| Train | 24 | 24,000 |
| Val | 3 | 3,000 |
| Test | 3 | 3,000 |

Readout choice:

- Features: `[z_t, z_{t+1}, z_{t+1} - z_t]`, all from frozen 192D released
  LeWM cube encoder latents. The encoder is not touched.
- Head: shallow MLP, `576 -> 256 -> 128 -> 5` with GELU activations.
- Loss: MSE on train-standardized action targets.
- Optimizer: AdamW, `lr=1e-3`, `weight_decay=1e-4`, batch size 512.
- Selection: early stop on validation normalized MSE. Best epoch 10; stopped at
  epoch 60 with patience 50.

Why this choice: the probe is a smoke test and ceiling estimate, so the head
should be expressive enough to catch nonlinear action information without
becoming the main experiment. Standardizing inputs/actions makes the objective
balanced across action dimensions, and the train-mean-action baseline keeps the
readout anchored to a trivial chance level.

Held-out test result:

| Model | MSE | Train-var-normalized MSE | Mean R2 |
|---|---:|---:|---:|
| Mean train action | 0.178389 | 0.981614 | -0.001411 |
| MLP readout | 0.088691 | 0.542089 | 0.435415 |

Gate: PASS. The readout improves held-out test normalized MSE by 44.8% over the
mean-action baseline, above the 10% smoke threshold.

Per-action held-out test result:

| Action dim | Baseline norm MSE | Readout norm MSE | Readout R2 |
|---|---:|---:|---:|
| action_0 | 1.049 | 0.220 | 0.791 |
| action_1 | 0.932 | 0.412 | 0.557 |
| action_2 | 0.987 | 0.191 | 0.807 |
| action_3 | 0.954 | 1.586 | -0.671 |
| action_4 | 0.986 | 0.301 | 0.695 |

Interpretation: the frozen released LeWM cube latents contain clearly decodable
action information overall, so the probe harness is not obviously broken.
However, action_3 does not decode on this holdout split. Downstream Gap tests
should compare both overall normalized MSE and the per-dimension profile rather
than relying on a single scalar.
