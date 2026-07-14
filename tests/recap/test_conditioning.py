import jax.numpy as jnp
import numpy as np

from openpi.recap.conditioning import ConditioningState
from openpi.recap.conditioning import apply_condition_dropout
from openpi.recap.conditioning import combine_cfg


def test_condition_dropout_uses_distinct_null_state():
    labels = jnp.array([ConditioningState.NEGATIVE, ConditioningState.POSITIVE])
    actual = apply_condition_dropout(labels, jnp.array([False, True]))
    np.testing.assert_array_equal(
        actual,
        jnp.array([ConditioningState.UNCONDITIONAL, ConditioningState.POSITIVE]),
    )


def test_cfg_endpoints():
    null = jnp.array([1.0, 2.0])
    positive = jnp.array([3.0, 6.0])
    np.testing.assert_allclose(combine_cfg(null, positive, 0.0), null)
    np.testing.assert_allclose(combine_cfg(null, positive, 1.0), positive)
