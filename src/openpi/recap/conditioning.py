"""Shared advantage-conditioning and classifier-free guidance primitives."""

from enum import IntEnum

import jax.numpy as jnp


class ConditioningState(IntEnum):
    NEGATIVE = 0
    POSITIVE = 1
    UNCONDITIONAL = 2


def apply_condition_dropout(labels: jnp.ndarray, keep_mask: jnp.ndarray) -> jnp.ndarray:
    """Map dropped conditions to the null state, never to the negative state."""
    labels = jnp.asarray(labels, dtype=jnp.int32)
    keep_mask = jnp.asarray(keep_mask, dtype=bool)
    return jnp.where(keep_mask, labels, int(ConditioningState.UNCONDITIONAL))


def combine_cfg(
    unconditional_velocity: jnp.ndarray,
    positive_velocity: jnp.ndarray,
    beta: float | jnp.ndarray,
) -> jnp.ndarray:
    """Combine flow fields so beta=0 is null and beta=1 is positive."""
    return unconditional_velocity + beta * (positive_velocity - unconditional_velocity)
