import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[2] / "scripts" / "recap" / "train_value_prototype.py"
SPEC = importlib.util.spec_from_file_location("train_value_prototype", MODULE_PATH)
value = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(value)


def test_binary_metrics_are_exact_for_perfect_predictions():
    metrics = value.binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["auroc"] == 1.0
    assert np.isclose(metrics["brier"], 0.025)


def test_binary_metrics_handle_ties():
    metrics = value.binary_metrics(np.array([0, 1]), np.array([0.5, 0.5]))
    assert metrics["auroc"] == 0.5


def test_temperature_scaling_is_fit_without_changing_rank_order():
    logits = np.array([-4.0, -1.0, 1.0, 4.0])
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    temperature = value.fit_temperature(logits, labels)
    assert temperature > 0.0
    assert np.array_equal(np.argsort(logits), np.argsort(logits / temperature))


def test_progress_heuristic_uses_episode_level_counts():
    def episode(success: bool):
        return {"success": success, "actions": np.zeros((4, 1))}

    samples = [
        ("a", episode(True), 0),
        ("a", episode(True), 1),
        ("b", episode(False), 0),
        ("b", episode(False), 1),
    ]
    estimates = value.progress_heuristic(samples, bins=2, smoothing=0.0)
    assert estimates[0] == 0.5
