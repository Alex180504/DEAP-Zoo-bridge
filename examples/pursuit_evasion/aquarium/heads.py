"""Train/eval action-head selection for evolved (EA/CMA/PSO/QD) aquarium policies.

This module holds only the *methodological convention* for the aquarium
comparison; the head implementations themselves live in the library
(:mod:`deap_pettingzoo.policies.action_heads`).

* **Training / search** uses the library's stochastic
  :class:`~deap_pettingzoo.policies.action_heads.SoftmaxSamplingHead`
  (discrete) via ``make_head(..., stochastic=True)``.  The default argmax
  head gives gradient-free search a step-function fitness landscape (small
  weight perturbations almost never flip an argmax), while sampling from
  ``softmax(logits / temperature)`` keeps the landscape smooth at decision
  boundaries.  Continuous mode uses :class:`BoxHead` (tanh) — already
  smooth, so ``stochastic`` is a no-op there.
* **Scored evaluation** uses the deterministic argmax head
  (:class:`~deap_pettingzoo.policies.action_heads.DiscreteHead` /
  :class:`BoxHead`) via plain ``make_head``.
"""

from __future__ import annotations

import numpy as np

from deap_pettingzoo.policies.action_heads import ActionHead, make_head


def make_train_head(action_space, temperature: float, rng_seed: int) -> ActionHead:
    """Return the EA/CMA *training* head for either aquarium action space.

    Discrete spaces get the stochastic
    :class:`~deap_pettingzoo.policies.action_heads.SoftmaxSamplingHead`
    (RNG seeded from ``rng_seed`` so identical genomes under the same env
    seed produce identical action streams — common-random-numbers
    evaluation at the policy layer); continuous spaces fall through to the
    smooth :class:`BoxHead`.
    """
    return make_head(
        action_space,
        stochastic=True,
        temperature=temperature,
        rng=np.random.default_rng(rng_seed & 0xFFFF_FFFF),
    )


def make_eval_head(action_space) -> ActionHead:
    """Return the deterministic champion-evaluation head (argmax / tanh)."""
    return make_head(action_space)


__all__ = [
    "make_train_head",
    "make_eval_head",
]
