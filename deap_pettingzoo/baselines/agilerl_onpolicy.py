"""Shared helpers for single-env on-policy (IPPO/PPO) rollout collection.

Centralises the tensor-shape conventions AgileRL's ``IPPO.learn`` expects so
that the standalone on-policy trainer (:class:`IPPOTrainer`) and the
co-evolution RL side (:class:`RLSideState`) assemble identical ``experiences``
8-tuples. The scope here is the single-environment case (``num_envs == 1``)
used by co-evolution; the vectorised (``num_envs > 1``) path stays in the
trainer.

The 8-tuple consumed by ``agent.learn`` is::

    (states, actions, log_probs, rewards, dones, values, next_states, next_dones)

where the first six are per-agent dicts of per-step lists and the last two are
per-agent dicts holding the final bootstrap observation / done flag. Every
stored entry carries the leading ``num_envs == 1`` axis.
"""

from typing import Any, Dict, List, Tuple

import numpy as np


def sanitize_obs(o: Any) -> np.ndarray:
    """Replace all-NaN inactive-agent placeholders with zeros.

    PettingZoo workers emit all-NaN observations for agents that have gone
    inactive; AgileRL networks need finite inputs. Finite observations pass
    through unchanged.
    """
    arr = np.asarray(o, dtype=np.float32)
    if not np.isnan(arr).any():
        return arr

    clean = np.array(arr, copy=True)
    if clean.ndim <= 1:
        if np.isnan(clean).all():
            clean[...] = 0.0
        return clean

    inactive = np.isnan(clean).all(axis=tuple(range(1, clean.ndim)))
    clean[inactive] = 0.0
    return clean


def batch_obs(o: Any) -> np.ndarray:
    """Sanitize then add the leading ``num_envs == 1`` axis to a single-env obs."""
    return np.asarray(sanitize_obs(o))[None, ...]


class OnPolicyRolloutBuffer:
    """Accumulate per-agent on-policy transitions for one ``learn`` call.

    Usage per env step (after ``agent.get_action`` and ``env.step``)::

        buf.add(obs, action, log_prob, value, reward, done)

    where ``obs``/``reward``/``done`` are per-agent dicts for the learner's
    agents, ``action``/``log_prob``/``value`` are the per-agent dicts returned
    by ``agent.get_action`` (already carrying the ``(1,)`` axis), and ``done``
    holds the *pre-step* done flags. After collecting, call
    :meth:`experiences` with the final bootstrap obs + done to get the 8-tuple.
    """

    def __init__(self, agent_ids: List[str]) -> None:
        self.agent_ids = list(agent_ids)
        self.states: Dict[str, List] = {a: [] for a in self.agent_ids}
        self.actions: Dict[str, List] = {a: [] for a in self.agent_ids}
        self.log_probs: Dict[str, List] = {a: [] for a in self.agent_ids}
        self.rewards: Dict[str, List] = {a: [] for a in self.agent_ids}
        self.dones: Dict[str, List] = {a: [] for a in self.agent_ids}
        self.values: Dict[str, List] = {a: [] for a in self.agent_ids}

    def __len__(self) -> int:
        return len(self.states[self.agent_ids[0]])

    @staticmethod
    def _as_vec(x: Any) -> np.ndarray:
        """Coerce a per-agent ``get_action`` output to a ``(1, ...)`` array."""
        if x is None:
            return np.array([np.nan], dtype=np.float32)
        return np.asarray(x)

    def add(
        self,
        obs: Dict[str, Any],
        action: Dict[str, Any],
        log_prob: Dict[str, Any],
        value: Dict[str, Any],
        reward: Dict[str, Any],
        done: Dict[str, Any],
    ) -> None:
        """Record one transition step for every learner agent."""
        for a in self.agent_ids:
            self.states[a].append(batch_obs(obs[a]))
            self.actions[a].append(self._as_vec(action.get(a)))
            self.log_probs[a].append(self._as_vec(log_prob.get(a)))
            self.values[a].append(self._as_vec(value.get(a)))
            self.rewards[a].append(
                np.array([float(reward.get(a, np.nan))], dtype=np.float32)
            )
            self.dones[a].append(
                np.asarray(done[a], dtype=np.float32).reshape(1)
            )

    def experiences(
        self, bootstrap_obs: Dict[str, Any], next_done: Dict[str, Any]
    ) -> Tuple[Any, ...]:
        """Return the 8-tuple ``agent.learn`` expects.

        ``bootstrap_obs`` is the final next-obs per-agent dict (used by GAE to
        bootstrap the last step); ``next_done`` the final per-agent done dict.
        """
        boot = {a: batch_obs(bootstrap_obs[a]) for a in self.agent_ids}
        nd = {
            a: np.asarray(next_done[a], dtype=np.float32).reshape(1)
            for a in self.agent_ids
        }
        return (
            self.states,
            self.actions,
            self.log_probs,
            self.rewards,
            self.dones,
            self.values,
            boot,
            nd,
        )
