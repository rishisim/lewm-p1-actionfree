"""Resolve the governed external storage used by local LeWM workflows."""

from __future__ import annotations

import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SPEC_PATH = REPO_ROOT / "storage.json"


class ExternalStorageUnavailable(RuntimeError):
    """Raised when a required governed storage root cannot be validated."""


def _spec() -> dict:
    return json.loads(SPEC_PATH.read_text())


def storage_root() -> Path:
    """Return the validated LeWM storage root.

    ``LEWM_STORAGE_ROOT`` is the explicit override for prepared remote storage.
    On the local Mac, the configured USB volume is mandatory.
    """

    spec = _spec()
    override = os.environ.get("LEWM_STORAGE_ROOT")
    root = Path(override).expanduser() if override else Path(spec["default_root"])
    marker = root / spec["marker"]
    if not root.is_dir() or not marker.is_file():
        raise ExternalStorageUnavailable(
            "LeWM external storage is unavailable. Connect the USB volume "
            f"'{spec['volume_name']}' (expected root: {root}) or set "
            "LEWM_STORAGE_ROOT to a prepared storage root."
        )
    marker_data = json.loads(marker.read_text())
    if (
        marker_data.get("storage_id") != spec["storage_id"]
        or marker_data.get("volume_uuid") != spec["volume_uuid"]
    ):
        raise ExternalStorageUnavailable(
            f"Refusing unrecognized LeWM storage at {root}; marker identity does not match."
        )
    manifest = root / spec["manifest_relative"]
    if not manifest.is_file():
        raise ExternalStorageUnavailable(f"LeWM storage manifest is missing: {manifest}")
    return root


def stablewm_home() -> Path:
    """Return the external stable-worldmodel home and export STABLEWM_HOME."""

    path = storage_root() / _spec()["stablewm_relative"]
    if not path.is_dir():
        raise ExternalStorageUnavailable(f"Stable WorldModel storage is missing: {path}")
    os.environ["STABLEWM_HOME"] = str(path)
    return path


def p1_hdf5_dir() -> Path:
    """Return the authoritative P1 Cube split directory."""

    path = storage_root() / _spec()["p1_hdf5_relative"]
    if not path.is_dir():
        raise ExternalStorageUnavailable(f"P1 HDF5 storage is missing: {path}")
    return path


def require_external_storage() -> Path:
    """Preflight the USB and configure stable-worldmodel for this process."""

    return stablewm_home()


if __name__ == "__main__":
    print(require_external_storage())
    print(p1_hdf5_dir())
