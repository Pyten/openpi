import importlib.util
import pathlib

import numpy as np

MODULE_PATH = pathlib.Path(__file__).parents[2] / "scripts" / "recap" / "label_advantages.py"
SPEC = importlib.util.spec_from_file_location("label_advantages", MODULE_PATH)
label_advantages = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(label_advantages)


def test_nstep_advantage_subtracts_current_value_and_bootstraps():
    rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dones = np.array([False, False, True])
    values = np.array([0.2, 0.5, 0.9], dtype=np.float32)
    advantages, returns = label_advantages.compute_nstep_advantages(
        rewards, dones, values, n=1, gamma=1.0
    )
    np.testing.assert_allclose(returns, [0.5, 0.9, 1.0])
    np.testing.assert_allclose(advantages, [0.3, 0.4, 0.1], atol=1e-6)


def test_terminal_transition_does_not_bootstrap():
    advantages, returns = label_advantages.compute_nstep_advantages(
        np.array([-1.0], dtype=np.float32),
        np.array([True]),
        np.array([-0.25], dtype=np.float32),
        n=50,
        gamma=1.0,
    )
    np.testing.assert_allclose(returns, [-1.0])
    np.testing.assert_allclose(advantages, [-0.75])
