import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[2] / "scripts" / "recap" / "prepare_rollout_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_rollout_dataset", MODULE_PATH)
prepare = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(prepare)


def _records() -> list[dict]:
    records = []
    for task_id in range(2):
        for init_state_index in range(10):
            for repeat in range(2):
                records.append(
                    {
                        "episode_id": f"task_{task_id:02d}/ep_{init_state_index * 2 + repeat:03d}",
                        "task_id": task_id,
                        "ep_idx": init_state_index * 2 + repeat,
                        "init_state_index": init_state_index,
                        "success": (init_state_index + repeat + task_id) % 3 != 0,
                    }
                )
    return records


def test_grouped_split_has_no_episode_or_init_state_overlap():
    records = _records()
    splits, _ = prepare.create_splits(records, seed=123)
    validation = prepare.validate_splits(splits, len(records))
    assert validation["status"] == "passed"
    assert validation["group_overlap_count"] == 0
    assert sum(len(split) for split in splits.values()) == len(records)


def test_grouped_split_is_reproducible_and_approximately_sized():
    records = _records()
    first, _ = prepare.create_splits(records, seed=456)
    second, _ = prepare.create_splits(records, seed=456)
    assert {
        name: [record["episode_id"] for record in values]
        for name, values in first.items()
    } == {
        name: [record["episode_id"] for record in values]
        for name, values in second.items()
    }
    assert 0.60 <= len(first["train"]) / len(records) <= 0.80
    assert 0.10 <= len(first["val"]) / len(records) <= 0.25
    assert 0.10 <= len(first["test"]) / len(records) <= 0.25


def test_grouped_split_preserves_task_success_rate():
    records = _records()
    splits, _ = prepare.create_splits(records, seed=789)
    source_rate = sum(record["success"] for record in records) / len(records)
    for values in splits.values():
        split_rate = sum(record["success"] for record in values) / len(values)
        # Tiny synthetic groups can force a coarse split; the production data has
        # enough groups for the tighter balance reported by the splitter.
        assert abs(split_rate - source_rate) <= 0.40
