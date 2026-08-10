# External LeWM Storage

The authoritative local LeWM source datasets and unique checkpoints were
migrated to:

`/Volumes/ChildLens_Governed/lewm-storage`

The P1 Cube split is stored under
`lewm-p1-inversedynamics/p1-cube/data/lewm_hdf5`; shared Stable WorldModel
sources and checkpoints are under `stable-worldmodel`. The machine-readable
contract is `../storage.json` and the checksummed inventory is
`../storage_manifest.json`.

## Preflight

```bash
python lewm_storage.py
```

For a prepared remote copy:

```bash
export LEWM_STORAGE_ROOT=/path/to/lewm-storage
python lewm_storage.py
```

The guard exports `STABLEWM_HOME` for the current Python process and resolves
the P1 split through `lewm_storage.p1_hdf5_dir()`. A missing or incorrectly
marked root is rejected before data loading begins.

## Recovery policy

- Preserve the HDF5 files listed as `provenance_source` in the manifest.
- Preserve the split manifest and locally unique checkpoints.
- Re-download public Hugging Face artifacts from the sources recorded by the
  project when needed.
- Rebuild derived Lance datasets and caches; do not treat them as authoritative.
- Verify every restored artifact against `storage_manifest.json` before use.

The USB is not a backup. Replicate irreplaceable checkpoints to a second
governed location before retiring this device.
