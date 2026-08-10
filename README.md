# LeWM P1 Inverse Dynamics

Proposal 1 workspace for action-free pretraining plus inverse dynamics.

## Contents

- `le-wm/`: local source snapshot based on `lucas-maes/le-wm`, with a default-off `action_free` training stub and inverse-dynamics placeholder.
- `prelim-p1/`: Proposal 1 staged feasibility scripts, reports, logs, and compact result artifacts.
- `sweep/`: TODO harness area for action-label-fraction experiments.
- `docs/`: workspace notes and reports, including the LeWM reconnaissance report.

## Quick Map

| Path | Purpose |
| --- | --- |
| `le-wm/` | Model source, configs, and local training/eval entrypoints. |
| `prelim-p1/scripts/` | Reproducible preliminary-test scripts. |
| `prelim-p1/outputs/` | Tracked compact artifacts from the staged preliminary tests. |
| `docs/reports/` | Human-readable research notes and reconnaissance reports. |
| `sweep/` | Placeholder for the action-label-fraction sweep harness. |

## Excluded

This repository intentionally excludes local virtual environments, Python caches, large dataset caches, checkpoints, videos, and transient Hydra/output folders.

## External Storage

Authoritative source datasets and locally unique checkpoints live on the
governed USB volume at `/Volumes/ChildLens_Governed/lewm-storage`. The canonical
layout, checksums, and recovery procedure are recorded in
`docs/external_storage.md`, `storage.json`, and `storage_manifest.json`.

Run `python lewm_storage.py` as a preflight. The core training and evaluation
entrypoints and the local Cube readout-cache builder run the same preflight
automatically and stop with a clear message when the USB is not connected.
Prepared remote machines must set `LEWM_STORAGE_ROOT` to their governed copy.
