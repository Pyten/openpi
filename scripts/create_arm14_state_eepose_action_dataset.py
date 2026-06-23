"""Create a LeRobot dataset with arm-joint state and eepose actions.

The output dataset keeps images and frame metadata from a 32-dim joint
dataset, slices `observation.state` to the 14 arm joint dimensions, and
replaces `action` with the paired 14-dim eepose action dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import tqdm


ARM14_IDXS = [*range(2, 9), *range(15, 22)]
TIMESTAMP_ATOL = 1e-4


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _slice_list(values: list, idxs: list[int] = ARM14_IDXS) -> list:
    return [values[i] for i in idxs]


def _slice_stats(stats: dict, idxs: list[int] = ARM14_IDXS) -> dict:
    sliced = {}
    for key, value in stats.items():
        if key == "count":
            sliced[key] = value
        elif isinstance(value, list):
            sliced[key] = _slice_list(value, idxs)
        else:
            sliced[key] = value
    return sliced


def _extract_arm14_state_column(series: pd.Series) -> list[np.ndarray]:
    state32 = np.stack(series.to_numpy()).astype(np.float32)
    if state32.shape[-1] != 32:
        raise ValueError(f"Expected 32-dim joint state, got shape {state32.shape}")
    return [row for row in state32[:, ARM14_IDXS]]


def _as_float32_array_column(series: pd.Series, *, expected_dim: int, name: str) -> list[np.ndarray]:
    values = np.stack(series.to_numpy()).astype(np.float32)
    if values.shape[-1] != expected_dim:
        raise ValueError(f"Expected {expected_dim}-dim {name}, got shape {values.shape}")
    return [row for row in values]


def _check_dataset_meta(joint_root: Path, eepose_root: Path) -> tuple[dict, dict]:
    joint_info = _load_json(joint_root / "meta/info.json")
    eepose_info = _load_json(eepose_root / "meta/info.json")

    for key in ("total_episodes", "total_frames", "fps"):
        if joint_info.get(key) != eepose_info.get(key):
            raise ValueError(f"Dataset metadata mismatch for {key}: {joint_info.get(key)} != {eepose_info.get(key)}")

    joint_state_shape = joint_info["features"]["observation.state"]["shape"]
    eepose_action_shape = eepose_info["features"]["action"]["shape"]
    if joint_state_shape != [32]:
        raise ValueError(f"Expected joint observation.state shape [32], got {joint_state_shape}")
    if eepose_action_shape != [14]:
        raise ValueError(f"Expected eepose action shape [14], got {eepose_action_shape}")

    return joint_info, eepose_info


def _validate_episode_alignment(joint_df: pd.DataFrame, eepose_df: pd.DataFrame, episode_index: int) -> None:
    if len(joint_df) != len(eepose_df):
        raise ValueError(f"Episode {episode_index}: row count mismatch {len(joint_df)} != {len(eepose_df)}")

    for col in ("episode_index", "frame_index", "index", "task_index"):
        if col in joint_df.columns and col in eepose_df.columns:
            if not np.array_equal(joint_df[col].to_numpy(), eepose_df[col].to_numpy()):
                raise ValueError(f"Episode {episode_index}: column {col} is not aligned")

    if "timestamp" in joint_df.columns and "timestamp" in eepose_df.columns:
        if not np.allclose(joint_df["timestamp"].to_numpy(), eepose_df["timestamp"].to_numpy(), atol=TIMESTAMP_ATOL):
            raise ValueError(f"Episode {episode_index}: timestamp is not aligned")


def _copy_meta_files(joint_root: Path, output_root: Path) -> None:
    meta_out = output_root / "meta"
    meta_out.mkdir(parents=True, exist_ok=True)
    for filename in ("episodes.jsonl", "tasks.jsonl"):
        shutil.copy2(joint_root / "meta" / filename, meta_out / filename)


def _build_info(joint_info: dict, eepose_info: dict) -> dict:
    info = json.loads(json.dumps(joint_info))
    features = info["features"]

    joint_names = joint_info["features"]["observation.state"]["names"][0]
    arm14_names = _slice_list(joint_names)
    eepose_action_feature = eepose_info["features"]["action"]

    features["observation.state"] = {
        "dtype": "float32",
        "shape": [14],
        "names": [arm14_names],
    }
    features["action"] = eepose_action_feature
    features.pop("observation.velocity", None)
    return info


def _build_norm_stats(joint_root: Path, eepose_root: Path) -> dict:
    joint_stats = _load_json(joint_root / "norm_stats.json")["norm_stats"]
    eepose_stats = _load_json(eepose_root / "norm_stats.json")["norm_stats"]
    return {
        "norm_stats": {
            "state": _slice_stats(joint_stats["state"]),
            "actions": eepose_stats["actions"],
        }
    }


def _build_episodes_stats(joint_root: Path, eepose_root: Path) -> list[dict]:
    joint_records = _read_jsonl(joint_root / "meta/episodes_stats.jsonl")
    eepose_records = _read_jsonl(eepose_root / "meta/episodes_stats.jsonl")

    if len(joint_records) != len(eepose_records):
        raise ValueError(f"episodes_stats length mismatch: {len(joint_records)} != {len(eepose_records)}")

    output_records = []
    for joint_record, eepose_record in zip(joint_records, eepose_records, strict=True):
        if joint_record["episode_index"] != eepose_record["episode_index"]:
            raise ValueError(
                "episodes_stats episode mismatch: "
                f"{joint_record['episode_index']} != {eepose_record['episode_index']}"
            )

        joint_stats = joint_record["stats"]
        eepose_stats = eepose_record["stats"]
        stats = {key: value for key, value in joint_stats.items() if key != "observation.velocity"}
        stats["observation.state"] = _slice_stats(joint_stats["observation.state"])
        stats["action"] = eepose_stats["action"]
        output_records.append({"episode_index": joint_record["episode_index"], "stats": stats})

    return output_records


def _episode_index_from_path(path: Path) -> int:
    return int(path.stem.split("_")[1])


def create_dataset(joint_dataset: Path, eepose_dataset: Path, output_dataset: Path, *, overwrite: bool) -> None:
    joint_info, eepose_info = _check_dataset_meta(joint_dataset, eepose_dataset)

    if output_dataset.exists():
        if not overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_dataset}. Use --overwrite to replace it.")
        shutil.rmtree(output_dataset)

    joint_files = sorted(joint_dataset.glob("data/chunk-*/episode_*.parquet"), key=_episode_index_from_path)
    eepose_files = sorted(eepose_dataset.glob("data/chunk-*/episode_*.parquet"), key=_episode_index_from_path)
    if len(joint_files) != len(eepose_files):
        raise ValueError(f"Episode file count mismatch: {len(joint_files)} != {len(eepose_files)}")
    if not joint_files:
        raise FileNotFoundError(f"No joint parquet files found under {joint_dataset / 'data'}")

    for joint_path, eepose_path in tqdm.tqdm(
        list(zip(joint_files, eepose_files, strict=True)),
        desc="Creating arm14/eepose14 dataset",
    ):
        episode_index = _episode_index_from_path(joint_path)
        if episode_index != _episode_index_from_path(eepose_path):
            raise ValueError(f"Episode file mismatch: {joint_path.name} != {eepose_path.name}")

        joint_df = pd.read_parquet(joint_path)
        eepose_df = pd.read_parquet(eepose_path)
        _validate_episode_alignment(joint_df, eepose_df, episode_index)

        out_df = joint_df.copy()
        out_df["observation.state"] = _extract_arm14_state_column(joint_df["observation.state"])
        out_df["action"] = _as_float32_array_column(eepose_df["action"], expected_dim=14, name="eepose action")
        if "observation.velocity" in out_df.columns:
            out_df = out_df.drop(columns=["observation.velocity"])

        relative_path = joint_path.relative_to(joint_dataset)
        out_path = output_dataset / relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(out_path, index=False)

    _copy_meta_files(joint_dataset, output_dataset)
    _write_json(output_dataset / "meta/info.json", _build_info(joint_info, eepose_info))
    _write_json(output_dataset / "norm_stats.json", _build_norm_stats(joint_dataset, eepose_dataset))
    _write_jsonl(output_dataset / "meta/episodes_stats.jsonl", _build_episodes_stats(joint_dataset, eepose_dataset))

    print(f"Created dataset: {output_dataset}")
    print("observation.state: 14 arm joints; action: 14 eepose values")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joint-dataset",
        type=Path,
        default=Path("/home/ma-user/work/wkq/robot_data/2026_grab_train"),
        help="Path to the 32-dim joint LeRobot dataset.",
    )
    parser.add_argument(
        "--eepose-dataset",
        type=Path,
        default=Path("/home/ma-user/work/wkq/robot_data/2026_grab_train_eepose"),
        help="Path to the 14-dim eepose LeRobot dataset.",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=Path("/home/ma-user/work/wkq/robot_data/2026_grab_train_arm14_state_eepose_action"),
        help="Path where the mixed 14-14 dataset will be written.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace the output dataset if it already exists.")
    args = parser.parse_args()

    create_dataset(args.joint_dataset, args.eepose_dataset, args.output_dataset, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
