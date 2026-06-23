"""Compute episodes_stats.jsonl from parquet episode files.

This script reads all episode parquet files from a LeRobot v2.1 dataset,
computes per-episode min/max/mean/std statistics for each numeric field,
and outputs episodes_stats.jsonl.

Usage:
    python /chenhaiying/iagcloud/chenhaiying/openpi/scripts/generate_episodes_stats.py \
        --dataset-dir /home/ma-user/work/pyten/Programs/data/Ego_data/ST260527_lerobot
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm


ARRAY_FIELDS = {"observation.state", "action"}
SCALAR_FIELDS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
IMAGE_FIELDS = {"observation.images.egocentric"}


def _stats_for_array(arr: np.ndarray) -> dict:
    """Compute element-wise min/max/mean/std for a 2D array (frames x dim).

    NaN values are ignored (using nan-aware numpy functions).
    """
    n = arr.shape[0]
    return {
        "min": np.nanmin(arr, axis=0).tolist(),
        "max": np.nanmax(arr, axis=0).tolist(),
        "mean": np.nanmean(arr, axis=0).tolist(),
        "std": np.nanstd(arr, axis=0).tolist(),
        "count": [n],
    }


def _stats_for_scalar(series: pd.Series) -> dict:
    """Compute min/max/mean/std for a scalar pandas Series."""
    n = len(series)
    vals = series.to_numpy()
    return {
        "min": [float(vals.min())],
        "max": [float(vals.max())],
        "mean": [float(vals.mean())],
        "std": [float(vals.std(ddof=0))],
        "count": [n],
    }


def _placeholder_image_stats(num_frames: int) -> dict:
    return {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": [[[0.5]], [[0.5]], [[0.5]]],
        "std": [[[0.25]], [[0.25]], [[0.25]]],
        "count": [num_frames],
    }


def compute_stats(df: pd.DataFrame) -> dict[str, dict]:
    """Compute stats for all fields in a single-episode DataFrame."""
    stats: dict[str, dict] = {}

    for field in ARRAY_FIELDS:
        if field in df.columns:
            arr = np.stack(df[field].to_numpy())
            stats[field] = _stats_for_array(arr)

    for field in SCALAR_FIELDS:
        if field in df.columns:
            stats[field] = _stats_for_scalar(df[field])

    for field in IMAGE_FIELDS:
        num_frames = len(df)
        stats[field] = _placeholder_image_stats(num_frames)

    return stats


def main(dataset_dir: str | Path):
    dataset_dir = Path(dataset_dir)
    data_dir = dataset_dir / "data"
    meta_dir = dataset_dir / "meta"
    output_path = meta_dir / "episodes_stats.jsonl"

    # Collect all episode parquet files across all chunk directories
    parquet_files = sorted(
        data_dir.glob("chunk-*/episode_*.parquet"),
        key=lambda p: int(p.stem.split("_")[1]),
    )

    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found under {data_dir}/")

    print(f"Found {len(parquet_files)} episode parquet files across {data_dir}/")

    results: list[dict] = []
    for pf in tqdm.tqdm(parquet_files, desc="Computing episode stats"):
        df = pd.read_parquet(pf)

        episode_index = int(pf.stem.split("_")[1])
        stats = compute_stats(df)

        results.append({
            "episode_index": episode_index,
            "stats": stats,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    print(f"Written {len(results)} episode stat entries to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute episodes_stats.jsonl from parquet episode files"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to the dataset root (must contain data/chunk-*/episode_*.parquet and meta/)",
    )
    args = parser.parse_args()
    main(args.dataset_dir)
