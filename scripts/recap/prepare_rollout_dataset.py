"""Audit RECAP rollout files and create episode-level split manifests."""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = {
    "task_id",
    "task_name",
    "prompt",
    "ep_idx",
    "success",
    "length",
    "images",
    "wrist_imgs",
    "states",
    "actions",
    "rewards",
    "dones",
    "collection_seed",
    "init_state_index",
    "policy_id",
    "action_rng_mode",
    "protocol",
}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values)
    return {
        "min": int(array.min()),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p95": float(np.percentile(array, 95)),
        "max": int(array.max()),
        "mean": float(array.mean()),
    }


def _check_episode(
    episode: dict[str, Any], task_id: int, task_index: int, source_file: Path
) -> tuple[list[str], dict[str, Any]]:
    prefix = f"{source_file.name}[{task_index}]"
    errors: list[str] = []
    missing = sorted(REQUIRED_KEYS - episode.keys())
    if missing:
        errors.append(f"{prefix}: missing keys {missing}")

    length = int(episode.get("length", -1))
    if length <= 0:
        errors.append(f"{prefix}: invalid length {length}")
    if episode.get("task_id") != task_id:
        errors.append(f"{prefix}: task_id={episode.get('task_id')} expected {task_id}")

    expected_shapes = {
        "images": (224, 224, 3),
        "wrist_imgs": (224, 224, 3),
        "states": (8,),
        "actions": (7,),
        "rewards": (),
        "dones": (),
    }
    for key, trailing_shape in expected_shapes.items():
        value = episode.get(key)
        if not isinstance(value, np.ndarray):
            errors.append(f"{prefix}: {key} is not an ndarray")
            continue
        if value.shape != (length, *trailing_shape):
            errors.append(
                f"{prefix}: {key}.shape={value.shape}, expected {(length, *trailing_shape)}"
            )
        if key in {"states", "actions", "rewards"} and not np.isfinite(value).all():
            errors.append(f"{prefix}: {key} contains non-finite values")

    for key in ("images", "wrist_imgs"):
        value = episode.get(key)
        if isinstance(value, np.ndarray) and value.dtype != np.uint8:
            errors.append(f"{prefix}: {key}.dtype={value.dtype}, expected uint8")

    return errors, {
        "episode_id": f"task_{task_id:02d}/ep_{int(episode.get('ep_idx', -1)):03d}",
        "task_id": task_id,
        "task_name": episode.get("task_name"),
        "ep_idx": int(episode.get("ep_idx", -1)),
        "task_index": task_index,
        "source_file": source_file.name,
        "success": bool(episode.get("success", False)),
        "length": length,
        "collection_seed": int(episode.get("collection_seed", -1)),
        "init_state_index": int(episode.get("init_state_index", -1)),
    }


def audit_dataset(data_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    task_files = sorted(data_dir.glob("task_*.pkl"))
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    task_summaries = []

    if len(task_files) != manifest.get("num_tasks"):
        errors.append(
            f"found {len(task_files)} task files, manifest expects {manifest.get('num_tasks')}"
        )

    for task_id, task_file in enumerate(task_files):
        try:
            with task_file.open("rb") as stream:
                episodes = pickle.load(stream)
        except Exception as exc:
            errors.append(f"{task_file.name}: pickle load failed: {exc}")
            continue
        if not isinstance(episodes, list):
            errors.append(f"{task_file.name}: root object is not a list")
            continue

        task_records = []
        protocols = set()
        policy_ids = set()
        rng_modes = set()
        for task_index, episode in enumerate(episodes):
            if not isinstance(episode, dict):
                errors.append(f"{task_file.name}[{task_index}]: episode is not a dict")
                continue
            episode_errors, record = _check_episode(
                episode, task_id, task_index, task_file
            )
            errors.extend(episode_errors)
            task_records.append(record)
            protocols.add(json.dumps(episode.get("protocol", {}), sort_keys=True))
            policy_ids.add(str(episode.get("policy_id")))
            rng_modes.add(str(episode.get("action_rng_mode")))

        ep_indices = [record["ep_idx"] for record in task_records]
        if len(ep_indices) != len(set(ep_indices)):
            errors.append(f"{task_file.name}: duplicate ep_idx values")
        expected_count = manifest.get("episodes_per_task")
        if len(task_records) != expected_count:
            errors.append(
                f"{task_file.name}: {len(task_records)} episodes, expected {expected_count}"
            )

        successes = sum(record["success"] for record in task_records)
        task_summaries.append(
            {
                "task_id": task_id,
                "source_file": task_file.name,
                "episodes": len(task_records),
                "successes": successes,
                "success_rate": successes / max(len(task_records), 1),
                "length": _quantiles([record["length"] for record in task_records]),
                "policy_ids": sorted(policy_ids),
                "action_rng_modes": sorted(rng_modes),
                "protocols": [json.loads(value) for value in sorted(protocols)],
            }
        )
        records.extend(task_records)
        del episodes
        gc.collect()

    total_successes = sum(record["success"] for record in records)
    if len(records) != manifest.get("num_episodes"):
        errors.append(
            f"audited {len(records)} episodes, manifest expects {manifest.get('num_episodes')}"
        )
    if total_successes != manifest.get("num_successes"):
        errors.append(
            f"audited {total_successes} successes, manifest expects {manifest.get('num_successes')}"
        )

    audit = {
        "data_dir": str(data_dir.resolve()),
        "manifest": manifest,
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "summary": {
            "episodes": len(records),
            "successes": total_successes,
            "success_rate": total_successes / max(len(records), 1),
            "length": _quantiles([record["length"] for record in records]),
            "task_counts": dict(sorted(Counter(r["task_id"] for r in records).items())),
        },
        "tasks": task_summaries,
    }
    return audit, records


def _target_counts(total: int) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in SPLIT_RATIOS.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    for name in sorted(raw, key=lambda key: raw[key] - counts[key], reverse=True):
        if sum(counts.values()) == total:
            break
        counts[name] += 1
    return counts


def _allocate_successes(successes: int, targets: dict[str, int], total: int) -> dict[str, int]:
    raw = {name: successes * count / total for name, count in targets.items()}
    allocation = {name: min(math.floor(raw[name]), targets[name]) for name in targets}
    while sum(allocation.values()) < successes:
        candidates = [name for name in targets if allocation[name] < targets[name]]
        name = max(candidates, key=lambda key: (raw[key] - allocation[key], targets[key]))
        allocation[name] += 1
    return allocation


def create_splits(
    records: list[dict[str, Any]], seed: int
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    splits = {name: [] for name in SPLIT_RATIOS}
    summaries = []
    task_ids = sorted({record["task_id"] for record in records})
    for task_id in task_ids:
        task_records = [record for record in records if record["task_id"] == task_id]
        positive = [record for record in task_records if record["success"]]
        negative = [record for record in task_records if not record["success"]]
        rng = np.random.default_rng(seed + task_id)
        rng.shuffle(positive)
        rng.shuffle(negative)

        targets = _target_counts(len(task_records))
        positive_counts = _allocate_successes(len(positive), targets, len(task_records))
        pos_offset = neg_offset = 0
        task_summary = {"task_id": task_id, "splits": {}}
        for name in SPLIT_RATIOS:
            pos_count = positive_counts[name]
            neg_count = targets[name] - pos_count
            selected = (
                positive[pos_offset : pos_offset + pos_count]
                + negative[neg_offset : neg_offset + neg_count]
            )
            rng.shuffle(selected)
            splits[name].extend(selected)
            pos_offset += pos_count
            neg_offset += neg_count
            task_summary["splits"][name] = {
                "episodes": len(selected),
                "successes": sum(record["success"] for record in selected),
            }
        summaries.append(task_summary)

    for name in splits:
        splits[name].sort(key=lambda record: (record["task_id"], record["ep_idx"]))
    return splits, summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit, records = audit_dataset(args.data_dir)
    _json_dump(output_dir / "audit.json", audit)
    if audit["errors"]:
        for error in audit["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    splits, task_summaries = create_splits(records, args.seed)
    for name, split_records in splits.items():
        _json_dump(
            output_dir / f"{name}.json",
            {"split": name, "seed": args.seed, "episodes": split_records},
        )
    summary = {
        "seed": args.seed,
        "ratios": SPLIT_RATIOS,
        "splits": {
            name: {
                "episodes": len(values),
                "successes": sum(record["success"] for record in values),
                "success_rate": sum(record["success"] for record in values)
                / max(len(values), 1),
            }
            for name, values in splits.items()
        },
        "tasks": task_summaries,
    }
    _json_dump(output_dir / "split_summary.json", summary)
    print(json.dumps({"audit": audit["summary"], "splits": summary["splits"]}, indent=2))


if __name__ == "__main__":
    main()
