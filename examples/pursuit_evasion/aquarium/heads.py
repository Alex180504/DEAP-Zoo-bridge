"""Action heads for evolved (EA/CMA) aquarium policies.

Discrete mode uses :class:`SoftmaxSamplingHead`: the default argmax head
gives gradient-free search a step-function fitness landscape (small weight
perturbations almost never flip an argmax), while sampling from
``softmax(logits / temperature)`` keeps the landscape smooth at decision
boundaries.  Continuous mode uses the library :class:`BoxHead` (tanh) —
already smooth, no sampling needed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from deap_pettingzoo.policies.action_heads import ActionHead, BoxHead, DiscreteHead


class SoftmaxSamplingHead(ActionHead):
    """Stochastic softmax-sampling head for Discrete spaces (EA/CMA training).

    Seed the RNG deterministically (e.g. from the agent id) so two identical
    genomes evaluated under the same env seed produce the same action stream
    — this preserves common-random-numbers evaluation at the policy layer.
    """

    def __init__(self, n: int, temperature: float = 0.2,
                 rng: Optional[np.random.Generator] = None) -> None:
        if n < 1:
            raise ValueError(f"SoftmaxSamplingHead requires n >= 1, got {n}")
        if temperature <= 0:
            raise ValueError(
                f"SoftmaxSamplingHead requires temperature > 0, got {temperature}"
            )
        self.n = int(n)
        self.temperature = float(temperature)
        self.rng = rng if rng is not None else np.random.default_rng()

    @property
    def output_size(self) -> int:
        return self.n

    def decode(self, raw_output: np.ndarray) -> int:
        arr = np.asarray(raw_output, dtype=np.float64).ravel()
        if arr.size != self.n:
            raise ValueError(
                f"SoftmaxSamplingHead expected output of size {self.n}, "
                f"got size {arr.size}"
            )
        scaled = arr / self.temperature
        scaled -= scaled.max()
        exp = np.exp(scaled)
        probs = exp / exp.sum()
        return int(self.rng.choice(self.n, p=probs))


def make_train_head(action_space, temperature: float, rng_seed: int) -> ActionHead:
    """Return the EA/CMA *training* head for either aquarium action space."""
    if hasattr(action_space, "n"):
        return SoftmaxSamplingHead(
            n=int(action_space.n),
            temperature=temperature,
            rng=np.random.default_rng(rng_seed & 0xFFFF_FFFF),
        )
    return BoxHead(low=action_space.low, high=action_space.high)


def make_eval_head(action_space) -> ActionHead:
    """Return the deterministic champion-evaluation head."""
    if hasattr(action_space, "n"):
        return DiscreteHead(int(action_space.n))
    return BoxHead(low=action_space.low, high=action_space.high)


def head_output_size(action_space) -> int:
    """Network output width required for *action_space*."""
    if hasattr(action_space, "n"):
        return int(action_space.n)
    return int(np.prod(action_space.shape))


__all__ = [
    "SoftmaxSamplingHead",
    "make_train_head",
    "make_eval_head",
    "head_output_size",
]
