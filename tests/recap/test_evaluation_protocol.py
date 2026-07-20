from openpi.recap import evaluation_protocol as protocol


def test_protocol_matches_official_libero_defaults():
    assert protocol.SUITE_NAME == "libero_spatial"
    assert protocol.REPLAN_STEPS == 5
    assert protocol.NUM_WAIT_STEPS == 10
    assert protocol.MAX_STEPS == 220
    assert protocol.EPISODES_PER_TASK == 50


def test_action_rng_seed_is_deterministic_and_unique_for_small_grid():
    seeds = {
        protocol.action_rng_seed(seed, episode, step)
        for seed in range(5)
        for episode in range(50)
        for step in range(220)
    }
    assert len(seeds) == 5 * 50 * 220


def test_action_rng_seed_matches_collector_replan_steps():
    seed = 3
    episode = 17
    wait_steps = protocol.NUM_WAIT_STEPS
    collector_steps = range(wait_steps, wait_steps + protocol.MAX_STEPS, protocol.REPLAN_STEPS)
    collector_keys = [
        protocol.action_rng_seed(seed, episode, step - wait_steps)
        for step in collector_steps
    ]
    evaluator_keys = [
        protocol.action_rng_seed(seed, episode, step)
        for step in range(0, protocol.MAX_STEPS, protocol.REPLAN_STEPS)
    ]
    assert collector_keys == evaluator_keys
