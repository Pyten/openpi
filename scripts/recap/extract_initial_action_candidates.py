"""Extract compact same-initial-state action-candidate data from rollout shards."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-steps", type=int, default=5)
    args = parser.parse_args()

    shards = sorted(args.rollout_dir.glob("task_*.pkl"))
    if not shards:
        raise FileNotFoundError(f"no task shards under {args.rollout_dir}")

    records: list[dict] = []
    task_prompts: dict[str, str] = {}
    for shard_path in shards:
        with shard_path.open("rb") as handle:
            episodes = pickle.load(handle)
        print(f"loaded {shard_path.name}: {len(episodes)} episodes", flush=True)
        for episode in episodes:
            if len(episode["actions"]) < args.chunk_steps:
                continue
            task_id = int(episode["task_id"])
            task_prompts[str(task_id)] = str(episode["prompt"])
            records.append(
                {
                    "task_id": task_id,
                    "init_state_index": int(episode["init_state_index"]),
                    "episode_index": int(episode["ep_idx"]),
                    "success": int(bool(episode["success"])),
                    "image": np.asarray(episode["images"][0], dtype=np.uint8),
                    "wrist_image": np.asarray(episode["wrist_imgs"][0], dtype=np.uint8),
                    "state": np.asarray(episode["states"][0], dtype=np.float32),
                    "action_chunk": np.asarray(
                        episode["actions"][: args.chunk_steps], dtype=np.float32
                    ),
                }
            )
        del episodes

    records.sort(key=lambda record: (record["task_id"], record["episode_index"]))
    if not records:
        raise RuntimeError("no eligible rollout records")

    task_ids = np.asarray([record["task_id"] for record in records], dtype=np.int16)
    init_state_indices = np.asarray(
        [record["init_state_index"] for record in records], dtype=np.int16
    )
    successes = np.asarray([record["success"] for record in records], dtype=np.int8)
    episode_indices = np.asarray(
        [record["episode_index"] for record in records], dtype=np.int16
    )
    images = np.stack([record["image"] for record in records])
    wrist_images = np.stack([record["wrist_image"] for record in records])
    states = np.stack([record["state"] for record in records])
    action_chunks = np.stack([record["action_chunk"] for record in records])

    group_keys = list(zip(task_ids.tolist(), init_state_indices.tolist(), strict=True))
    groups = {key: [] for key in group_keys}
    for index, key in enumerate(group_keys):
        groups[key].append(index)
    mixed_groups = sum(
        any(successes[index] for index in indices)
        and any(not successes[index] for index in indices)
        for indices in groups.values()
    )
    pair_count = sum(
        int(successes[indices].sum()) * int(len(indices) - successes[indices].sum())
        for indices in groups.values()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "initial_candidates.npz",
        task_ids=task_ids,
        init_state_indices=init_state_indices,
        episode_indices=episode_indices,
        successes=successes,
        images=images,
        wrist_images=wrist_images,
        states=states,
        action_chunks=action_chunks,
    )
    manifest = {
        "source_dir": str(args.rollout_dir),
        "records": int(len(records)),
        "tasks": sorted(int(task_id) for task_id in np.unique(task_ids)),
        "chunk_steps": args.chunk_steps,
        "groups": int(len(groups)),
        "mixed_success_failure_groups": int(mixed_groups),
        "success_failure_pairs": int(pair_count),
        "task_prompts": task_prompts,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
