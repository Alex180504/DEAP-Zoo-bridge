from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .registry import (
    DEFAULT_EPISODE_LEN_FALLBACK,
    build_env,
    get_env_spec,
    is_registered,
    list_envs,
)


def _patch_fixed_cam_range(wrapper: "PettingZooWrapper") -> None:
    """Pin MPE's per-frame ``cam_range`` to a stable value for recordings.

    MPE's ``SimpleEnv.draw()`` recomputes ``cam_range`` every frame as the
    maximum absolute entity coordinate.  As agents move the zoom level
    fluctuates, making static obstacles appear to drift in screen-space
    even though they are motionless in world coordinates.  (The
    ``dynamic_rescaling`` flag only controls entity *radius*, not the
    viewport computation.)

    This function replaces ``draw`` on the underlying ``SimpleEnv``
    instance with a version that uses ``original_cam_range`` — set once
    at env construction — so all frames share the same pixel-per-world-unit
    scale.  Safe to call on non-MPE wrappers; silently fails if the
    expected attribute chain is absent.
    """
    try:
        # PettingZooWrapper.env  - aec_to_parallel_wrapper
        # .aec_env               - OrderEnforcingWrapper
        # .env                   - AssertOutOfBoundsWrapper
        # .env                   - SimpleEnv (owns .draw, .world, .screen)
        base = wrapper.env.aec_env.env.env
    except AttributeError:
        return
    if not (hasattr(base, "original_cam_range") and hasattr(base, "draw")):
        return

    try:
        import pygame as _pygame
    except ImportError:
        import warnings

        warnings.warn(
            "pygame is not installed; _patch_fixed_cam_range will be skipped. "
            "Install it to enable stable-viewport frame recording.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    _cam = base.original_cam_range  # fixed at env construction

    def _fixed_draw(b=base):
        b.screen.fill((255, 255, 255))
        for entity in b.world.entities:
            x, y = entity.state.p_pos
            y = -y  # flip y to match MPE display convention
            x = (x / _cam) * b.width // 2 * 0.9 + b.width // 2
            y = (y / _cam) * b.height // 2 * 0.9 + b.height // 2
            # Clamp so agents that stray beyond cam_range don't crash.
            x = float(max(1, min(b.width - 1, x)))
            y = float(max(1, min(b.height - 1, y)))
            radius = entity.size * 350  # dynamic_rescaling disabled
            _pygame.draw.circle(b.screen, entity.color * 200, (x, y), radius)
            _pygame.draw.circle(b.screen, (0, 0, 0), (x, y), radius, 1)

    base.draw = _fixed_draw


class PettingZooWrapper:
    """
    Wrapper for PettingZoo environments.

    Provides a unified interface for PettingZoo multi-agent environments,
    handling agent coordination, observation/action space management,
    and episode tracking.

    Any environment registered via
    :func:`deap_pettingzoo.environments.registry.register_env` can be
    wrapped by name. MPE2 environments and ``marl-aquarium``'s
    ``aquarium_v0`` are auto-registered on package import if the
    corresponding packages are installed.

    Attributes:
        env_name: Name of the PettingZoo environment
        env: The wrapped PettingZoo environment
        env_config: Configuration dict for environment
        render_mode: Rendering mode ('human', 'rgb_array', None)
        possible_agents: List of all possible agent names
        agents: List of currently active agents
        episode_length: Current episode step count
        cumulative_rewards: Dict of cumulative rewards per agent
    """

    @classmethod
    def supported_envs(cls) -> List[str]:
        """Return the sorted list of currently-registered environment names."""
        return list_envs()

    def __init__(
        self,
        env_name: str,
        env_config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """
        Initialize environment wrapper.

        Args:
            env_name: Registered PettingZoo environment name.
            env_config: Optional configuration dict for environment.
            render_mode: Rendering mode ('human', 'rgb_array', None).

        Raises:
            ValueError: If ``env_name`` is not registered. See
                :func:`deap_pettingzoo.environments.registry.register_env`
                for how to register third-party envs.

        Example:
            >>> wrapper = PettingZooWrapper(
            ...     env_name='simple_tag_v3',
            ...     env_config={'max_cycles': 25}
            ... )
        """
        if not is_registered(env_name):
            # get_env_spec raises a descriptive ValueError listing known names and recovery suggestions; delegate to it.
            get_env_spec(env_name)

        self.env_name = env_name
        self.env_config = env_config or {}
        self.render_mode = render_mode

        # Build environment via the registered factory.
        self.env = build_env(
            env_name,
            env_config=self.env_config,
            render_mode=render_mode,
        )

        if render_mode == "rgb_array":
            _patch_fixed_cam_range(self)

        # Store agent information
        self.possible_agents = self.env.possible_agents

        # Episode tracking
        self.episode_length = 0
        self.cumulative_rewards = {}
        #: Most recent infos dict returned by the underlying env's ``reset``/``step``.
        #: Evaluators read ``last_infos[agent].get("action_mask")`` to forward legal-action
        #: masks to mask-aware policies (see :class:`~deap_pettingzoo.policies.action_heads.MaskedDiscreteHead`).
        self.last_infos: Dict[str, Dict[str, Any]] = {}

    @property
    def env_spec(self):
        """Return the :class:`EnvSpec` describing this environment."""
        return get_env_spec(self.env_name)

    @property
    def default_episode_len(self) -> int:
        """Fallback episode length used for budget translation."""
        try:
            return self.env_spec.default_episode_len
        except ValueError:  # pragma: no cover - registry mutated post-init
            return DEFAULT_EPISODE_LEN_FALLBACK

    @property
    def agents(self) -> List[str]:
        """Get list of currently active agents."""
        return self.env.agents if hasattr(self.env, "agents") else []

    @property
    def num_agents(self) -> int:
        """Get number of currently active agents."""
        return len(self.agents)

    def observation_space(self, agent: str):
        """
        Get observation space for agent.

        Args:
            agent: Agent name

        Returns:
            Observation space for the agent
        """
        return self.env.observation_space(agent)

    def action_space(self, agent: str):
        """
        Get action space for agent.

        Args:
            agent: Agent name

        Returns:
            Action space for the agent
        """
        return self.env.action_space(agent)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Optional reset options

        Returns:
            Dict mapping agent names to initial observations

        Example:
            >>> observations = wrapper.reset(seed=42)
            >>> print(observations.keys())
            dict_keys(['adversary_0', 'agent_0', 'agent_1'])
        """
        # Reset environment
        observations, infos = self.env.reset(seed=seed, options=options)

        # Reset episode tracking
        self.episode_length = 0
        self.cumulative_rewards = {agent: 0.0 for agent in observations.keys()}
        self.last_infos = dict(infos) if infos is not None else {}

        return observations

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        """
        Execute one environment step.

        Args:
            actions: Dict mapping agent names to action arrays

        Returns:
            Tuple of (observations, rewards, dones, truncated, infos)

        Raises:
            ValueError: If actions missing for active agents or wrong shape

        Example:
            >>> actions = {agent: policy.forward(obs) for agent, obs in observations.items()}
            >>> obs, rewards, dones, truncated, infos = wrapper.step(actions)
        """
        # Validate all active agents have actions
        active_agents = set(self.agents)
        action_agents = set(actions.keys())

        if not action_agents.issuperset(active_agents):
            missing = active_agents - action_agents
            raise ValueError(
                f"Missing actions for agents: {missing}. "
                f"Expected actions for: {active_agents}"
            )

        # Convert and validate actions
        converted_actions = {}
        for agent, action in actions.items():
            if agent in active_agents:
                action_space = self.action_space(agent)

                # Convert numpy array to appropriate format
                if hasattr(action, "shape"):
                    # If action is array, convert based on space type
                    if hasattr(action_space, "n"):  # Discrete space
                        # Take argmax for discrete actions
                        action_value = (
                            int(np.argmax(action))
                            if action.size > 1
                            else int(action.item())
                        )
                    else:  # Box space
                        action_value = action
                else:
                    # Already a scalar
                    action_value = action

                # Validate action
                if not action_space.contains(action_value):
                    raise ValueError(
                        f"Action for agent '{agent}' not in action space: "
                        f"got {action_value}, space is {action_space}"
                    )

                converted_actions[agent] = action_value

        # Execute step
        observations, rewards, terminations, truncations, infos = self.env.step(
            converted_actions
        )

        # Update episode tracking
        self.episode_length += 1
        for agent, reward in rewards.items():
            if agent in self.cumulative_rewards:
                self.cumulative_rewards[agent] += reward
            else:
                self.cumulative_rewards[agent] = reward
        self.last_infos = dict(infos) if infos is not None else {}

        return observations, rewards, terminations, truncations, infos

    def render(self) -> Optional[np.ndarray]:
        """
        Render environment.

        Returns:
            RGB array if render_mode='rgb_array', otherwise None
        """
        if self.render_mode == "rgb_array":
            return self.env.render()
        elif self.render_mode == "human":
            self.env.render()
            return None
        else:
            return None

    def close(self) -> None:
        """Close environment and cleanup resources.

        Some third-party PettingZoo envs (notably ``marl-aquarium``'s
        ``aquarium_v0``) call ``sys.exit()`` from their ``close()`` method,
        which would otherwise kill the whole Python process. We catch
        ``SystemExit`` (and defensively ``Exception``) so closing is safe.
        """
        if self.env is not None and hasattr(self.env, "close"):
            try:
                self.env.close()
            except SystemExit:
                pass
            except Exception:
                pass
        self.env = None

    def __repr__(self) -> str:
        """String representation of wrapper."""
        return (
            f"PettingZooWrapper("
            f"env='{self.env_name}', "
            f"agents={len(self.possible_agents)}, "
            f"episode_length={self.episode_length})"
        )
