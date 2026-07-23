import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[2] / "scripts" / "recap" / "sidecar_rollout_loader.py"
SPEC = importlib.util.spec_from_file_location("sidecar_rollout_loader", MODULE_PATH)
loader_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(loader_module)


def _episode(task_id: int, ep_idx: int, success: bool):
    length = 2
    return {
        "task_id": task_id,
        "ep_idx": ep_idx,
        "length": length,
        "success": success,
        "prompt": f"task {task_id}",
        "images": np.zeros((length, 224, 224, 3), dtype=np.uint8),
        "wrist_imgs": np.zeros((length, 224, 224, 3), dtype=np.uint8),
        "states": np.zeros((length, 8), dtype=np.float32),
        "actions": np.zeros((length, 7), dtype=np.float32),
    }


def test_sidecar_loader_reads_train_split_and_balances_tasks(tmp_path):
    rollout = tmp_path / "rollout"
    sidecars = tmp_path / "sidecars"
    metadata = tmp_path / "metadata"
    rollout.mkdir(); sidecars.mkdir(); metadata.mkdir()
    episodes = [_episode(5, 0, True), _episode(9, 0, False)]
    with (rollout / "task_05.pkl").open("wb") as stream:
        pickle.dump([episodes[0]], stream)
    with (rollout / "task_09.pkl").open("wb") as stream:
        pickle.dump([episodes[1]], stream)
    for task_id, success in ((5, True), (9, False)):
        with (sidecars / f"task_{task_id:02d}_value_labels.pkl").open("wb") as stream:
            pickle.dump([{
                "episode_id": f"task_{task_id:02d}/ep_000",
                "return_only_labels": np.array([int(success), int(success)], dtype=np.int8),
                "mc_labels": np.array([0, 1], dtype=np.int8),
                "n50_labels": np.array([1, 0], dtype=np.int8),
            }], stream)
    payload = {"episodes": [
        {"episode_id": "task_05/ep_000", "task_id": 5},
        {"episode_id": "task_09/ep_000", "task_id": 9},
    ]}
    for split in ("train", "val", "test"):
        (metadata / f"{split}.json").write_text(json.dumps(payload if split == "train" else {"episodes": []}))

    loader = loader_module.SidecarRolloutDataLoader(
        rollout, sidecars, metadata, "mc", batch_size=8, seed=1
    )
    assert loader.num_samples == 4
    assert set(loader.task_ids) == {5, 9}

    def tokenize(prompts):
        return np.zeros((len(prompts), 3), dtype=np.int32), np.ones((len(prompts), 3), dtype=bool)

    batch = loader.sample_batch(tokenize)
    assert batch[0].shape == (8, 224, 224, 3)
    assert batch[3].shape == (8, 50, 7)
    assert set(batch[4].tolist()) <= {0, 1}
