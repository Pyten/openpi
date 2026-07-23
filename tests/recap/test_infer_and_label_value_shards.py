import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[2] / "scripts" / "recap" / "infer_and_label_value_shards.py"
SPEC = importlib.util.spec_from_file_location("infer_and_label_value_shards", MODULE_PATH)
labels = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(labels)


def test_signed_rewards_sets_terminal_outcome_without_mutating_source():
    source = np.array([0.0, 0.0], dtype=np.float32)
    failure = labels.signed_rewards({"rewards": source, "success": False})
    success = labels.signed_rewards({"rewards": source, "success": True})
    assert np.array_equal(failure, np.array([0.0, -1.0]))
    assert np.array_equal(success, np.array([0.0, 1.0]))
    assert np.array_equal(source, np.zeros(2))


def test_monte_carlo_returns_are_reverse_cumulative_rewards():
    result = labels.monte_carlo_returns(np.array([0.0, 0.0, 1.0]))
    assert np.array_equal(result, np.ones(3))


def test_nstep_advantage_bootstraps_at_exact_horizon():
    rewards = np.zeros(4, dtype=np.float32)
    values = np.array([0.1, 0.2, 0.7, 0.9], dtype=np.float32)
    advantages, returns = labels.nstep_advantages(
        rewards, np.zeros(4, dtype=bool), values, n_step=2
    )
    assert np.isclose(returns[0], values[2])
    assert np.isclose(advantages[0], values[2] - values[0])
    assert returns[2] == 0.0


def test_task_threshold_uses_only_values_passed_by_caller():
    train_values = [np.arange(10, dtype=np.float32)]
    threshold = labels.task_threshold(train_values, positive_fraction=0.4)
    assert np.isclose(threshold, np.percentile(np.arange(10), 60))
