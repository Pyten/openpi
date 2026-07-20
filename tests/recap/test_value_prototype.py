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
