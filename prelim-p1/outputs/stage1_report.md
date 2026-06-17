# Stage 1 Report v2: Proposal 1 Preliminary Test

The previous Stage 1 report is preserved as `stage1_report_v1_INVALID.md`; its action-information and headroom numbers are discarded.

## Stage 0 Timing + Fallback

| Item | Value | Unit/Notes |
| --- | --- | --- |
| Load/device | True | mps |
| Param count | 18,034,478 | 18.03M |
| Encode | 3.344504 | ms/frame |
| Predictor fwd | 3.159125 | ms |
| Predictor fwd+bwd+step | 18.467694 | ms |

No explicit MPS CPU fallback warnings appeared in the captured Stage 0 terminal log.

## V2 Extraction Summary

- Episodes used: 200 total; 140 train, 30 validation, 30 test trajectories.
- Pairs: 3076 train, 591 validation, 668 test, 4335 total.
- History latents: `(4335, 3, 192)`; actions: `(4335, 3, 10)`; targets: `(4335, 192)`.
- Latent stats: mean(mean_dim)=0.007824, mean(std_dim)=0.998798, any_nan=False.
- Contact labels: geometry-from-state with 1559 contact pairs and 2776 non-contact pairs.

## B. Pipeline Sanity Check

| Quantity | Test MSE / Value |
| --- | --- |
| Persistence | 0.197046 |
| Last-latent ridge | 0.153607 |
| Ratio | 0.779551 |
| Accepted ratio range | 0.65 to 1.35 |
| Passed | True |

Result: passed. The last-latent ridge sits in the same scale as persistence, so the v1 failure is treated as MLP overfit rather than a gross indexing/normalization bug.

## 1b Three-Way/Four-Method Comparison

Ridge probe: full 3-latent history, validation-selected alpha `4.64159`, validation MSE `0.126085`.
MLP probe: 576->512->192 ReLU MLP, stopped at epoch 165, best validation epoch 105, best validation MSE `0.184634`.

| Method | Abs MSE | Normalized MSE | Fraction of Persistence | Per-Trajectory MSE Mean ± Std |
| --- | --- | --- | --- | --- |
| Persistence | 0.197046 | 0.197124 | 1.000000 | 0.203991 ± 0.061174 |
| Ridge history probe | 0.123972 | 0.124022 | 0.629155 | 0.129947 ± 0.054948 |
| MLP history probe | 0.167294 | 0.167360 | 0.849011 | 0.173683 ± 0.097620 |
| Full LeWM | 0.007169 | 0.007171 | 0.036381 | 0.007515 ± 0.003226 |

- Best history-only probe used for headlines: `ridge`
- Action information value: `0.942176`
- Predictive headroom over persistence: `0.370845`
- Ridge/MLP test MSE ratio: `1.349447`

## D. Ridge Data-Scaling Control

| Train Fraction | Train Pairs | Alpha | Val MSE | Test MSE |
| --- | --- | --- | --- | --- |
| 12.5% | 384 | 21.544347 | 0.281600 | 0.272613 |
| 25.0% | 769 | 21.544347 | 0.192874 | 0.187958 |
| 50.0% | 1538 | 21.544347 | 0.147476 | 0.145072 |
| 100.0% | 3076 | 4.641589 | 0.126085 | 0.123972 |

Scaling status: `still_dropping`; 50% to 100% relative test-MSE drop `0.145441`.

![Ridge scaling](stage1v2_ridge_scaling.png)

## 1c Contact Split

| Subset | Pairs | Method | Abs MSE | Normalized MSE | Fraction of Persistence | Action Info Value |
| --- | --- | --- | --- | --- | --- | --- |
| Non-contact | 422 | Persistence | 0.182626 | 0.181360 | 1.000000 |  |
| Non-contact | 422 | Ridge history probe | 0.117398 | 0.116584 | 0.642829 | 0.937918 |
| Non-contact | 422 | MLP history probe | 0.163024 | 0.161893 | 0.892662 |  |
| Non-contact | 422 | Full LeWM | 0.007288 | 0.007238 | 0.039908 |  |
| Contact | 246 | Persistence | 0.221781 | 0.231821 | 1.000000 |  |
| Contact | 246 | Ridge history probe | 0.135251 | 0.141373 | 0.609840 | 0.948515 |
| Contact | 246 | MLP history probe | 0.174620 | 0.182524 | 0.787351 |  |
| Contact | 246 | Full LeWM | 0.006963 | 0.007279 | 0.031397 |  |

## Plain-Language Reading

The ridge and MLP probes diverge by more than 25%, so the report uses ridge as the robust history-only anchor. The ridge scaling curve is still dropping from 50% to 100% train pairs, so the history-only result remains data-limited; any residual gap to Full LeWM is not clean evidence that actions are the sole missing signal. The best history-only probe beats persistence, so the one-step comparison is no longer invalidated by the v1 overfit failure. Full LeWM remains far lower-error than the best history-only probe on this one-step test.

## Divergences, Caveats, Confounds

- No discrepancy for the loaded object checkpoint: `AutoCostModel('pusht/lewm')` resolves to `jepa.JEPA` from `le-wm/jepa.py`.
- The installed `stable_worldmodel.wm.lewm.LeWM` source has a different rollout implementation, but it is not the loaded object checkpoint class here; one-step `predict(emb, act_emb)` shape is the same.
- The HDF5 dataset has no `n_contacts` column; contact/non-contact uses the prior geometric fallback from 7D state.
- Known confound: probes are trained on latents from an already action-conditioned trained encoder, so this remains a proxy rather than a direct action-free pretraining measurement.
- Ridge features were standardized using probe-train statistics before closed-form fitting; errors are reported in original latent units.

Stage 2 was not run.
