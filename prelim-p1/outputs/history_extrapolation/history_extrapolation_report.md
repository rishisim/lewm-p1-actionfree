# Pixel History Extrapolation Diagnostic

- History/window setup: history_size=3, num_preds=1, frameskip=5.
- Sampling: up to 20,000 valid windows per dataset, seed=3072.
- Pixel-space only: no trained encoder, checkpoint, action input, or ImageNet normalization.

## Ranking

1. TwoRoom: gap=-0.00101653, relative=-45.99%, copy_mse=0.00221041, extrap_mse=0.00322694
2. Reacher: gap=-0.00124058, relative=-41.27%, copy_mse=0.00300623, extrap_mse=0.00424680
3. PushT: gap=-0.00194791, relative=-82.84%, copy_mse=0.00235155, extrap_mse=0.00429946

## Dataset Results

- PushT: sampled 20,000 windows from 11052 episodes; linear sanity gap=-0.00098887; elapsed=50.7s.
- Reacher: sampled 20,000 windows from 8077 episodes; linear sanity gap=-0.00078095; elapsed=138.1s.
- TwoRoom: sampled 20,000 windows from 7244 episodes; linear sanity gap=-0.00073595; elapsed=73.1s.
- Cube: status=missing; fetch_status=missing; error=

## Recommendation

This is not yet a complete four-dataset result; do not make the final dataset recommendation until all four datasets complete.
