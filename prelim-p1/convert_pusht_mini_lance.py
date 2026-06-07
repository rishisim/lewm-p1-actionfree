from pathlib import Path
import argparse

import h5py
import hdf5plugin  # noqa: F401 - registers compressed HDF5 filters.
import numpy as np

from stable_worldmodel.data.formats.lance import LanceWriter


DATASET_DIR = Path("/Users/rishisim/.stable_worldmodel/datasets")
PRESETS = {
    "pusht": {
        "src": DATASET_DIR / "pusht_expert_train.h5",
        "out": DATASET_DIR / "pusht_expert_train_mini.lance",
        "keys": ["pixels", "action", "proprio", "state"],
        "episode_key": "episode_idx",
        "nan_to_num_keys": [],
    },
    "reacher": {
        "src": DATASET_DIR / "reacher.h5",
        "out": DATASET_DIR / "reacher_mini.lance",
        "keys": ["pixels", "action", "observation"],
        "episode_key": "ep_idx",
        "nan_to_num_keys": ["action"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 episodes to Lance using the stable-worldmodel writer."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        choices=PRESETS,
        default="pusht",
        help="Dataset preset. Defaults to the historical PushT conversion.",
    )
    parser.add_argument("--src", type=Path, help="Override source HDF5 path.")
    parser.add_argument("--out", type=Path, help="Override output Lance path.")
    parser.add_argument(
        "--n-rows",
        type=int,
        help="Optional row limit. Omit to convert the full HDF5 dataset.",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=1000,
        help="Approximate rows per Lance write/progress update.",
    )
    return parser.parse_args()


def iter_episode_batches(src, data_keys, episode_key, nan_to_num_keys, n_rows, chunk_rows):
    with h5py.File(src, "r") as f:
        missing = [key for key in [episode_key, *data_keys] if key not in f]
        if missing:
            raise KeyError(f"{src} is missing required HDF5 keys: {missing}")

        total_rows = int(f[episode_key].shape[0])
        if n_rows is not None:
            total_rows = min(total_rows, n_rows)

        if "ep_offset" in f and "ep_len" in f:
            episode_spans = (
                (int(offset), int(offset) + int(length))
                for offset, length in zip(f["ep_offset"], f["ep_len"])
                if int(offset) < total_rows
            )
        else:
            episode_idx = f[episode_key]
            episode_spans = []
            start = 0
            while start < total_rows:
                ep = int(episode_idx[start])
                stop = start + 1
                while stop < total_rows and int(episode_idx[stop]) == ep:
                    stop += 1
                episode_spans.append((start, stop))
                start = stop

        batch = []
        batch_rows = 0
        for start, stop in episode_spans:
            stop = min(stop, total_rows)
            if start >= stop:
                continue
            episode = {}
            for key in data_keys:
                values = np.asarray(f[key][start:stop])
                if key in nan_to_num_keys:
                    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
                episode[key] = [values[i] for i in range(len(values))]
            batch.append(episode)
            batch_rows += stop - start

            if batch_rows >= chunk_rows:
                yield batch, batch_rows, total_rows
                batch = []
                batch_rows = 0

        if batch:
            yield batch, batch_rows, total_rows


def main():
    args = parse_args()
    preset = PRESETS[args.dataset]
    src = args.src or preset["src"]
    out = args.out or preset["out"]
    data_keys = preset["keys"]
    episode_key = preset["episode_key"]
    nan_to_num_keys = set(preset["nan_to_num_keys"])
    if args.chunk_rows < 1:
        raise ValueError("--chunk-rows must be at least 1")

    out.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with LanceWriter(out, mode="overwrite") as writer:
        for batch_idx, (episodes, batch_rows, total_rows) in enumerate(
            iter_episode_batches(
                src, data_keys, episode_key, nan_to_num_keys, args.n_rows, args.chunk_rows
            ),
            start=1,
        ):
            writer.write_episodes(episodes)
            row_count += batch_rows
            print(
                f"Chunk {batch_idx}: wrote {batch_rows} rows "
                f"({row_count}/{total_rows})",
                flush=True,
            )

    print(f"Wrote {row_count} rows to {out}, columns: {data_keys}")


if __name__ == "__main__":
    main()
