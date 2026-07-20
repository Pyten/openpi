"""Audit RECAP rollout files and create episode-level split manifests."""

from __future__ import annotations

import argparse
import gc
import json
import math
import pickle
from collections import Counter
from collections import defaultdict
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


def _group_cost(
    counts: dict[str, int],
    successes: dict[str, int],
    targets: dict[str, int],
    success_targets: dict[str, float],
) -> float:
    cost = 0.0
    for name in SPLIT_RATIOS:
        count_scale = max(targets[name], 1)
        success_scale = max(success_targets[name], 1.0)
        cost += ((counts[name] - targets[name]) / count_scale) ** 2
        cost += ((successes[name] - success_targets[name]) / success_scale) ** 2
        overflow = max(counts[name] - targets[name], 0)
        cost += 4.0 * (overflow / count_scale) ** 2
    return cost


def _assign_groups(
    task_records: list[dict[str, Any]], seed: int
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in task_records:
        grouped[record["init_state_index"]].append(record)

    rng = np.random.default_rng(seed)
    groups = list(grouped.values())
    group_targets = _target_counts(len(groups))
    targets = _target_counts(len(task_records))
    success_rate = sum(record["success"] for record in task_records) / len(task_records)
    success_targets = {name: targets[name] * success_rate for name in targets}

    best_assignment = None
    best_cost = float("inf")
    # Group constraints make a simple class-wise shuffle invalid. Search fixed group
    # quotas instead, selecting the partition whose episode counts and success rates
    # best match the source task. The seeded search is deterministic and cheap here.
    for _ in range(20_000):
        permutation = list(rng.permutation(groups))
        cursor = 0
        candidate = {}
        for name in SPLIT_RATIOS:
            count = group_targets[name]
            candidate[name] = [item for group in permutation[cursor : cursor + count] for item in group]
            cursor += count
        counts = {name: len(values) for name, values in candidate.items()}
        successes = {
            name: sum(record["success"] for record in values)
            for name, values in candidate.items()
        }
        cost = _group_cost(counts, successes, targets, success_targets)
        if cost < best_cost:
            best_cost = cost
            best_assignment = candidate

    assert best_assignment is not None
    return best_assignment


def validate_splits(
    splits: dict[str, list[dict[str, Any]]], expected_records: int
) -> dict[str, Any]:
    episode_owners: dict[str, str] = {}
    group_owners: dict[tuple[int, int], str] = {}
    errors = []
    for split_name, records in splits.items():
        for record in records:
            episode_id = record["episode_id"]
            previous_split = episode_owners.setdefault(episode_id, split_name)
            if previous_split != split_name:
                errors.append(
                    f"episode {episode_id} appears in {previous_split} and {split_name}"
                )
            group = (record["task_id"], record["init_state_index"])
            previous_split = group_owners.setdefault(group, split_name)
            if previous_split != split_name:
                errors.append(
                    f"init-state group {group} appears in {previous_split} and {split_name}"
                )

    if len(episode_owners) != expected_records:
        errors.append(
            f"split contains {len(episode_owners)} unique episodes, expected {expected_records}"
        )
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "unique_episodes": len(episode_owners),
        "unique_init_state_groups": len(group_owners),
        "group_overlap_count": 0 if not errors else sum("init-state group" in e for e in errors),
        "group_key": ["task_id", "init_state_index"],
    }


def create_splits(
    records: list[dict[str, Any]], seed: int
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    splits = {name: [] for name in SPLIT_RATIOS}
    summaries = []
    task_ids = sorted({record["task_id"] for record in records})
    for task_id in task_ids:
        task_records = [record for record in records if record["task_id"] == task_id]
        task_splits = _assign_groups(task_records, seed + task_id)
        task_summary = {"task_id": task_id, "splits": {}}
        for name in SPLIT_RATIOS:
            selected = task_splits[name]
            splits[name].extend(selected)
            task_summary["splits"][name] = {
                "episodes": len(selected),
                "successes": sum(record["success"] for record in selected),
                "init_state_groups": len(
                    {record["init_state_index"] for record in selected}
                ),
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
    split_validation = validate_splits(splits, len(records))
    if split_validation["errors"]:
        for error in split_validation["errors"]:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    for name, split_records in splits.items():
        _json_dump(
            output_dir / f"{name}.json",
            {"split": name, "seed": args.seed, "episodes": split_records},
        )
    summary = {
        "seed": args.seed,
        "ratios": SPLIT_RATIOS,
        "validation": split_validation,
        "storage": {
            "mode": "task_shard_references",
            "record_fields": ["source_file", "task_index"],
            "duplicates_rollout_payloads": False,
        },
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
