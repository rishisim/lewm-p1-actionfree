# P1 — Action-free pretraining + IDM under SIGReg (OGBench Cube)

**The one question this folder exists to answer:**
Under action-free SIGReg pretraining, is action information still recoverable
from the latent transitions — and if so, does it buy label efficiency on control?
Everything else (the data-efficiency curve, the workshop framing) is downstream
of that single fact.

## Why Cube, not PushT
- Cube is one of LeWM's three native envs (TwoRoom, PushT, OGBench Cube) → the
  SIGReg backbone applies with no porting, preserving the whole novelty claim.
- OGBench ships the diversity axis natively: `play` (temporally-correlated
  expert noise) vs `noisy` (uncorrelated Gaussian) datasets, i.e. narrow-vs-broad
  coverage — plus official data-gen scripts. PushT only ships narrow expert demos.
- Object physics gives real PASSIVE dynamics (the cube persists / settles after
  contact), so the action-free objective isn't starved AND the action is a
  meaningful-but-partial fraction of each transition → the IDM readout is a
  non-trivial test, unlike action-saturated envs (Reacher, point-mass).
- NOTE: "cube is eval-only" was a property of the *released LeWM checkpoint*
  (no training h5 shipped), NOT of the environment. OGBench provides the cube
  datasets + runnable sim + generation scripts.

## The ladder (rungs — NOT "stages"; see prelim-p1 for the retired stage* naming)

### probe/  — ceiling + harness smoke test  (does NOT test P1)
Freeze the *released* LeWM cube encoder, train the IDM readout
(z_t, z_{t+1}) → â_t. Two purposes: (a) confirm the extraction + IDM pipeline
runs end-to-end; (b) get the action-TRAINED ceiling (high by construction —
don't over-read it).
GATE: harness runs AND the action-trained encoder decodes actions above chance.
If even the ceiling can't decode → env/setup is broken → stop and fix.

### gap_test/  — action-recoverability gap  (THE CRUX)
Train two encoders, same data / size / hyperparameters, MULTIPLE SEEDS,
differing only in action conditioning:
  A = action-conditioned   B = action-free
Run the probe's IDM readout on both. The signal is the GAP (B vs A), not the
absolute number — identically-trained arms, so scale cancels and this can run
small.
  B ≈ A   → SIGReg already preserves action structure → IDM unnecessary → NEGATIVE.
  B ≪ A   → train C = B + IDM auxiliary head.
              C recovers toward A → mechanism is real → POSITIVE.
              C still ≪ A         → action-free SIGReg destroys it → P1 dies clean.
Single-seed gaps are noise; the result IS the gap, so seeds are non-negotiable.

### curve/  — data-efficiency curve  (HPC; POSITIVE branch only)
The attached figure, in full: scratch / pretrain+FT / pretrain+FT+IDM, plotted
as control success vs action-label fraction. This is the only rung that produces
the paper. Never run speculatively — only if gap_test comes back positive.

## Data principle (the load-bearing fix)
The action-free pool MUST be larger + more diverse than the labeled slice —
**not** expert demos with labels stripped. Two reasons:
  1. LeWM's own requirement: data must "sufficiently cover the environment
     dynamics" (exploratory/pseudo-expert OK, expertise not required). The
     released pipeline ignores this (it trains on pusht_expert_train.h5).
  2. The premise: "abundant cheap unlabeled data + few labels." If unlabeled =
     labeled-minus-labels, there's no abundance, no diversity, and the
     label-fraction x-axis is meaningless.
Build the action-free pool from OGBench play+noisy (or a generated
exploratory+pseudo-expert MIX — pure random starves the contact dynamics);
attach actions only to the small labeled fraction.

## Shared harness
The IDM readout is the SAME module in probe, gap_test, and curve's curve-3 head.
Build it once in harness/. Check le-wm/inverse_dynamics.py first (reuse if it is
the IDM); crib latent-extraction plumbing from
prelim-p1/scripts/stage1v2_extract.py — only the ENCODER SOURCE changes
(frozen-released → A/B/C-trained).

## Novelty positioning (workshop-scope; confirmed by lit check)
SIGReg (sliced isotropic-Gaussian matching via Epps-Pulley) is mechanistically
distinct from the regularizers in the nearest prior work:
  LAPO — VQ discrete bottleneck (unsupervised).
  ACT-JEPA — target-encoder/stop-gradient (the heuristic SIGReg removes).
  LAWM (Alles et al., arXiv:2512.10016, Dec 2025) — offline-RL latent-action
    world model, NOT a JEPA; ~10× fewer action labels on DMC.
None use SIGReg. Position P1 directly against LAWM: same label-efficiency claim,
different model class. The wedge — SIGReg's isotropic-Gaussian constraint is
more global than the VICReg family, so whether action structure survives it is
genuinely open (could go either way). This confirms (not lifts) the
"small workshop paper at most" calibration.

## Status
| rung      | state                          |
|-----------|--------------------------------|
| probe     | [ ] not started                |
| gap_test  | [ ] blocked on probe           |
| curve     | [ ] blocked on gap_test (POS)  |

## Relationship to prelim-p1/
prelim-p1 = the OLD confounded line (PushT, frozen action-trained encoder =
structurally blind to what action-free pretraining would do). Superseded by this
folder. Kept as record — stage1_report_v1_INVALID.md documents why the proxy
approach failed.
