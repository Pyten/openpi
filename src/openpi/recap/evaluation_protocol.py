"""Canonical LIBERO-Spatial evaluation protocol for RECAP comparisons."""

SUITE_NAME = "libero_spatial"
RESIZE = 224
NUM_WAIT_STEPS = 10
MAX_STEPS = 220
REPLAN_STEPS = 5
ACTION_HORIZON = 50
ACTION_DIM = 32
FLOW_STEPS = 10
EPISODES_PER_TASK = 50
EVALUATION_SEEDS = (0, 1, 2, 3, 4)


def action_rng_seed(seed: int, episode: int, step: int) -> int:
    # Keep the fields disjoint for the supported evaluation grid.
    return seed * 1_000_000 + episode * 10_000 + step
