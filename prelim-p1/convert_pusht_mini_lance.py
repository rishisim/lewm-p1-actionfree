from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers compressed HDF5 filters.
import numpy as np

from stable_worldmodel.data.formats.lance import LanceWriter


SRC = Path("/Users/rishisim/.stable_worldmodel/datasets/pusht_expert_train.h5")
OUT = Path("/Users/rishisim/.stable_worldmodel/datasets/pusht_expert_train_mini.lance")
N_ROWS = 500
DATA_KEYS = ["pixels", "action", "proprio", "state"]


def iter_episodes():
    with h5py.File(SRC, "r") as f:
        episode_idx = np.asarray(f["episode_idx"][:N_ROWS])
        for ep in np.unique(episode_idx):
            rows = np.flatnonzero(episode_idx == ep)
            start = int(rows[0])
            stop = int(rows[-1]) + 1
            episode = {}
            for key in DATA_KEYS:
                values = np.asarray(f[key][start:stop])
                episode[key] = [values[i] for i in range(len(values))]
            yield episode


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with LanceWriter(OUT, mode="overwrite") as writer:
        episodes = list(iter_episodes())
        row_count = sum(len(ep["action"]) for ep in episodes)
        writer.write_episodes(episodes)

    print(f"Wrote {row_count} rows, columns: {DATA_KEYS}")


if __name__ == "__main__":
    main()
