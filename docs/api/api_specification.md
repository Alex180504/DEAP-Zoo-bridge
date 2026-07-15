# API Specification

Authoritative reference for every public class, function, and dataclass in
`deap_pettingzoo/`. All signatures, defaults, and behaviours match the
current code (see the [project overview](../index.md) for the narrative
overview). Private helpers (leading underscore) are omitted unless they
are contractually relevant.

---

## Table of contents

1. [Package surface](#package-surface)
2. [Genomes (`deap_pettingzoo.genomes`)](#genomes)
3. [Policies (`deap_pettingzoo.policies`)](#policies)
4. [Environments (`deap_pettingzoo.environments`)](#environments)
5. [Evaluators (`deap_pettingzoo.evaluators`)](#evaluators)
6. [Integration (`deap_pettingzoo.integration`)](#integration)
7. [Baselines (`deap_pettingzoo.baselines`)](#baselines)
8. [Co-evolution (`baselines.coevolution`)](#co-evolution-baselinescoevolution)
9. [Catalog (`deap_pettingzoo.catalog`)](#catalog) — see also [configuration_schemas.md](configuration_schemas.md)
10. Experiment Service (`deap_pettingzoo.service`) — see [service_api.md](service_api.md)
11. [Logging & Analysis](#logging--analysis) — see [logging_and_analysis.md](logging_and_analysis.md)
12. [Type & protocol reference](#type--protocol-reference)
13. [Error handling conventions](#error-handling-conventions)

---

## Package surface

Top-level package re-exports the six sub-packages only:

```python
import deap_pettingzoo
deap_pettingzoo.genomes
deap_pettingzoo.policies
deap_pettingzoo.evaluators
deap_pettingzoo.environments
deap_pettingzoo.utils       # reserved; currently empty
```

Symbols are imported from their sub-packages (e.g.
`from deap_pettingzoo.genomes import FixedTopologyGenome`). The
`deap_pettingzoo.baselines` sub-package is not re-exported at the top
level but is fully public and stable.

---

## Genomes

Module: `deap_pettingzoo.genomes.fixed_topology`

### `class FixedTopologyGenome`

Flat-weight-vector neural network genome compatible with DEAP operators.
There is no separate `GenomeBase` ABC — this is the only concrete
genome class.

```python
class FixedTopologyGenome:
    SUPPORTED_ACTIVATIONS = {"relu", "tanh", "sigmoid", "linear"}

    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "tanh",
                output_activation: Optional[str] = None,
        weight_range: Tuple[float, float] = (-1.0, 1.0),
        random_seed: Optional[int] = None,
    ) -> None: ...
```

**Raises:**
- `ValueError` if `len(layer_sizes) < 2`.
- `ValueError` if `activation` not in `SUPPORTED_ACTIVATIONS`.
- `ValueError` if `output_activation` is not in `SUPPORTED_ACTIVATIONS`.

**Attributes:**
- `layer_sizes: List[int]` — `[input, hidden…, output]`.
- `activation: str` — hidden-layer activation.
- `output_activation: str` — final-layer activation, defaulting to `activation` when not provided.
- `weights: np.ndarray` — flat `float64` weight+bias vector initialised
  uniformly in `weight_range`.
- `rng: np.random.Generator` — per-instance RNG seeded by `random_seed`.

**Properties:**
- `input_size -> int` — `layer_sizes[0]`.
- `output_size -> int` — `layer_sizes[-1]`.
- `hidden_layers -> List[int]` — `layer_sizes[1:-1]`.

**Methods:**

| Signature | Behaviour |
| --- | --- |
| `get_weight_count() -> int` | Total scalar count: `Σ (in_i · out_i + out_i)`. |
| `get_weight_ranges() -> List[Tuple[int, int]]` | `(start, end)` slice into `weights` per layer. |
| `get_weights() -> np.ndarray` | Returns `self.weights` (no copy). |
| `set_weights(weights: np.ndarray) -> None` | Assigns after shape check; raises `ValueError` on mismatch. |
| `validate() -> bool` | `True` iff no NaN/Inf and length matches `get_weight_count()`. |
| `clone() -> FixedTopologyGenome` | Deep copy; clone gets an independent freshly-seeded RNG. |
| `to_dict() -> Dict[str, Any]` | Serialises layer sizes, activation, output_activation, weights (as list), range, seed. |
| `from_dict(data) -> FixedTopologyGenome` | Inverse of `to_dict`; `@staticmethod`. Missing `output_activation` is backfilled from `activation` for legacy payloads. |
| `mutate(mutation_rate=0.1, mutation_std=0.2, inplace=True) -> Optional[FixedTopologyGenome]` | Gaussian mutation with per-weight Bernoulli mask; returns `None` in-place, clone otherwise. |
| `__repr__()` | `FixedTopologyGenome(layers=…, activation=…, weights=…)`. |

---

## Policies

Module: `deap_pettingzoo.policies`. Both policies satisfy
[`PolicyProtocol`](#policyprotocol).

### `class FeedforwardNeuralPolicy`

Stateless MLP forward pass driven by a `FixedTopologyGenome`. Holds a
**reference** to the genome, so weight updates take effect immediately.

```python
class FeedforwardNeuralPolicy:
    def __init__(self, genome: FixedTopologyGenome) -> None: ...
```

**Raises:**
- `ValueError` if `genome is None`.
- `TypeError` if `genome` is not a `FixedTopologyGenome` instance.

**Properties:** `input_size`, `output_size`, `layer_sizes` (all delegate
to the genome).

**Methods:**

| Signature | Behaviour |
| --- | --- |
| `forward(observation: np.ndarray) -> np.ndarray` | Dense forward pass; hidden layers use `genome.activation`, the final layer uses `genome.output_activation`, and a `ValueError` is raised if `len(obs) != input_size`. |
| `get_action(observation, deterministic=True, action_mask=None) -> np.ndarray \| int` | Returns `forward(obs)` (with `N(0, 0.1)` noise if `deterministic=False`) and, if the policy was built with an `ActionHead`, pipes it through `head.decode(raw, action_mask=...)`. `action_mask` is forwarded to `MaskedDiscreteHead` and ignored by `DiscreteHead` / `MultiDiscreteHead` / `BoxHead`. |
| `__repr__()` | `FeedforwardNeuralPolicy(input_size=…, output_size=…, layers=…)`. |

Note: `get_action` returns the **raw output vector** unless an
`ActionHead` was attached at construction time. Discrete-action
environments must apply `argmax` externally —
[`PettingZooWrapper.step`](#class-pettingzoowrapper) does this
automatically when the action space is `Discrete`.

For the scripted aquarium pursuit-evasion comparison, the registered env applies
an experiment-local observation wrapper before the policy sees each predator
observation. The wrapper maps raw absolute-position / wrapped-angle inputs onto
relative geometric features and exposes cumulative catches through
`info['prey_caught_total']` for evaluation.

### Action heads (`deap_pettingzoo.policies.action_heads`)

```python
class ActionHead(ABC):
    @abstractmethod
    def decode(self, raw: np.ndarray, action_mask=None): ...

class DiscreteHead(ActionHead):
    def __init__(self, n: int): ...
    # decode: argmax over raw[:n]; ignores action_mask.

class MaskedDiscreteHead(DiscreteHead):
    # decode: zero illegal logits to -inf before argmax.
    # action_mask: np.ndarray of shape (n,) with {0, 1} entries.

class MultiDiscreteHead(ActionHead):
    def __init__(self, nvec): ...
    # decode: split sum(nvec) logits into one chunk per sub-action and
    # argmax each -> np.int64 array of shape (len(nvec),). ignores action_mask.

class BoxHead(ActionHead):
    def __init__(self, low: np.ndarray, high: np.ndarray): ...
    # decode: tanh(raw) -> rescale into [low, high].

def required_output_size(space) -> int: ...
def make_head(space, *, masked: bool = False) -> ActionHead: ...
```

`make_head(space, masked=True)` is the idiomatic factory: it returns
`MaskedDiscreteHead` (or `DiscreteHead`) for `Discrete` spaces,
`MultiDiscreteHead` for `MultiDiscrete`, `BoxHead` for `Box`, and raises
`NotImplementedError` on `Tuple` / `Dict` / other spaces. `required_output_size`
returns `n` for `Discrete`, `sum(nvec)` for `MultiDiscrete`, and `prod(shape)`
for `Box`. `SelfPlayEvaluator` calls `make_head` with `masked=True` so
`infos[agent]["action_mask"]` flows end-to-end (the `masked` flag is a no-op
for non-`Discrete` spaces).

### `@dataclass APFConfig`

Hyperparameters for `APFEvaderPolicy`.

| Field | Default | Meaning |
| --- | --- | --- |
| `k_pursuer: float` | `1.0` | Gain on inverse-square pursuer repulsion. |
| `k_landmark: float` | `0.3` | Gain on inverse-square landmark repulsion. |
| `k_boundary: float` | `2.0` | Gain on (linear) boundary pushback. |
| `boundary_threshold: float` | `0.9` | Per-axis distance beyond which boundary term activates. |
| `no_op_threshold: float` | `1e-3` | Minimum projection magnitude to emit a non-zero action. |
| `softening: float` | `1e-3` | Added to `‖r‖³` to avoid singularities. |

### `class APFEvaderPolicy`

Scripted, stateless evader. Reads the agent's observation through a
supplied [`ObsLayout`](#dataclass-obslayout), computes a resultant repulsive
force, then quantises it to one of the five discrete MPE actions.

```python
class APFEvaderPolicy:
    def __init__(
        self,
        layout: ObsLayout,
        config: Optional[APFConfig] = None,
    ) -> None: ...

    def get_action(
        self,
        observation: np.ndarray,
        deterministic: bool = True,   # accepted for symmetry; ignored
    ) -> int: ...

    def reset(self) -> None: ...      # no-op
```

Action space index order:
`0 = no-op, 1 = -x, 2 = +x, 3 = -y, 4 = +y`.

---

## Environments

Module: `deap_pettingzoo.environments`.

### `class PettingZooWrapper`

Thin unified wrapper over any registered PettingZoo parallel environment.
`env_name` is resolved through the open registry
(`deap_pettingzoo.environments.registry`).  MPE2 envs
and `aquarium_v0` are auto-registered at import time (see
`environments/builtin.py`); third-party envs are added with
`register_env(...)`.

```python
class PettingZooWrapper:
    def __init__(
        self,
        env_name: str,
        env_config: Optional[Dict[str, Any]] = None,
        render_mode: Optional[str] = None,
    ) -> None: ...

    # `SUPPORTED_ENVS` is a lazy, dict-like view over the registry
    # (module-style factories only).
    SUPPORTED_ENVS: Mapping[str, Any]
```

**Raises:** `ValueError` (starts with "Unsupported / unregistered
environment") if `env_name` is not registered; the message lists all
known envs and suggests `register_env` or installing extras.

**Attributes:** `env_name`, `env` (raw parallel env), `env_config`,
`render_mode`, `possible_agents`, `episode_length: int`,
`cumulative_rewards: Dict[str, float]`, `env_spec: EnvSpec`,
`default_episode_len: int`.

**Properties:** `agents -> List[str]` (active agents), `num_agents -> int`.

**Methods:**

| Signature | Behaviour |
| --- | --- |
| `observation_space(agent) -> gym.Space` | Delegates to raw env. |
| `action_space(agent) -> gym.Space` | Delegates to raw env. |
| `reset(seed=None, options=None) -> Dict[str, np.ndarray]` | Resets the env; resets `episode_length` and `cumulative_rewards`. |
| `step(actions) -> Tuple[obs, rewards, terms, truncs, infos]` | Validates that every active agent has an action; converts arrays to discrete via `argmax` when action space is `Discrete`; raises `ValueError` on missing actions or out-of-space values. |
| `render() -> Optional[np.ndarray]` | Returns RGB array when `render_mode='rgb_array'`. |
| `close() -> None` | Closes the underlying env; sets `self.env = None`. |
| `PettingZooWrapper.supported_envs() -> List[str]` | Classmethod; same as `registry.list_envs()`. |

### `registry` — open environment registry

Module: `deap_pettingzoo.environments.registry`.

```python
EnvApi = Literal[
    "parallel", "aec", "auto",
    "aec_masked", "aec_sequential",
    "jax_parallel",
]

@dataclass
class EnvSpec:
    name: str
    factory: Any            # module with .parallel_env/.env/.raw_env, or callable
    default_episode_len: int = 25
    continuous_default: bool = False
    notes: str = ""
    tags: frozenset = frozenset()
    api: EnvApi = "parallel"
    wrappers: Tuple[Callable[[Any], Any], ...] = ()

def register_env(name, factory, *, default_episode_len=25,
                 continuous_default=False, notes="",
                 tags=None, api="parallel",
                 wrappers=(), overwrite=False) -> EnvSpec
def unregister_env(name) -> None
def get_env_spec(name) -> EnvSpec          # raises ValueError if missing
def is_registered(name) -> bool
def list_envs(tag=None, api=None) -> List[str]
def build_env(name, env_config=None, render_mode=None) -> ParallelEnv
def build_env_from_spec(spec: EnvSpec, env_config=None, render_mode=None) -> ParallelEnv
```

**API semantics.** ``api`` declares the shape the factory produces:

- `"parallel"` (default) takes the legacy fast path and raises
  `TypeError` if the constructed env is actually an `AECEnv`.
- `"aec"` calls the factory's `env`/`raw_env` attribute (or the
  factory itself) and converts through
  `pettingzoo.utils.conversions.aec_to_parallel` — the upstream env
  must set `metadata["is_parallelizable"] = True`.
- `"auto"` prefers `parallel_env` and falls back to converting an
  `AECEnv`. Converted envs get a trust-but-verify wrapper that
  warns on the first reset when active agents differ from
  `possible_agents` and raises `ValueError` if a later step
  silently drops a previously-active agent.
- `"aec_masked"` wraps a strictly turn-based AEC env
  (`is_parallelizable = False`) with
  :class:`~deap_pettingzoo.environments.MaskedSimultaneousAdapter`:
  every active agent appears each step and
  `infos[agent]["action_mask"]` is populated; only the current
  player's action actually executes. Preferred for self-play
  EA / CMA-ES workflows — `SingleEpisodeEvaluator` /
  `SelfPlayEvaluator` work unchanged.
- `"aec_sequential"` wraps the same envs with
  :class:`~deap_pettingzoo.environments.SequentialTurnAdapter`:
  single-agent-per-step dicts, preserving true turn semantics for
  future TD-learning / AlphaZero-style experiments.

Passing `api="aec"` for a non-parallelizable env raises a clear
`TypeError` naming `aec_masked` and `aec_sequential` as the correct
alternatives.

- `"jax_parallel"` declares that the factory returns a raw
  JAX-native env satisfying
  :class:`~deap_pettingzoo.environments.JaxEnvProtocol`
  (e.g. a JaxMARL env). :func:`build_env` wraps it in
  :class:`~deap_pettingzoo.environments.JaxEnvToPettingZooAdapter`
  so the rest of the stack sees a regular PettingZoo `ParallelEnv`.
  QDax trainers (`MAPElitesTrainer`, `CMAMETrainer`) additionally
  branch on this api to take the fast JAX-native scoring path
  (`jit`+`vmap`) instead of the numpy-boundary path. See
  :mod:`deap_pettingzoo.contrib.jaxmarl` for the one-liner
  registration helper.

**`wrappers` chain.** `EnvSpec.wrappers` is an ordered tuple of
plain `(env -> env)` callables applied **after** api coercion, in
left-to-right order. Each entry must be callable; non-callables
raise `TypeError` eagerly at registration. This is the hook for
composing [SuperSuit](https://pypi.org/project/SuperSuit/)
transforms (observation padding, agent-ID channels, reward /
observation normalization, black-death) or any other PettingZoo
wrapper. Canonical MPE ordering for heterogeneous-obs envs:

```python
from supersuit import (
    pad_observations_v0, pad_action_space_v0, agent_indicator_v0,
)
register_env(
    "simple_adversary_padded_v3",
    simple_adversary_v3,
    api="parallel",
    wrappers=(
        pad_observations_v0,
        pad_action_space_v0,
        agent_indicator_v0,
    ),
    overwrite=True,
)
```

Install the optional extra with
`pip install 'deap-pettingzoo[supersuit]'`.

Side-effect module `environments/builtin.py` auto-registers every
available MPE2 env (tagged `{"mpe2", ...}`) and, if installed,
`aquarium_v0` (tagged `{"aquarium"}`).  The pursuit-evasion contrib
subpackage registers `"pursuit_evasion"` on import. The
`deap_pettingzoo.contrib.openspiel` module is **opt-in** — nothing
is registered on import; see the "Turn-based games via Shimmy"
subsection below.

### `@dataclass PursuitEvasionConfig`

*(Experiment-side research code; lives in
`examples.pursuit_evasion.simple_tag.env`, not in the library.)*

Single source of truth for the pursuit-evasion scenario.

| Field | Default | Meaning |
| --- | --- | --- |
| `num_adversaries: int` | `10` | Pursuer count (`adversary_*`). |
| `num_good: int` | `10` | Evader count (`agent_*`). |
| `num_obstacles: int` | `3` | Static landmarks. |
| `max_cycles: int` | `200` | Steps per episode. |
| `equal_ability: bool` | `True` | Overwrite every agent's `max_speed`/`accel` after each reset. |
| `equal_max_speed: float` | `1.0` | Applied when `equal_ability=True`. |
| `equal_accel: float` | `3.0` | Applied when `equal_ability=True`. |
| `num_agent_neighbors: Optional[int]` | `5` | K-nearest agents visible; `None` => full observability. |
| `num_landmark_neighbors: Optional[int]` | `3` | K-nearest landmarks visible; `None` => full. |
| `curriculum: bool` | `False` | Enable MPE2 prey-speed curriculum (stages 0/1/2 = 50/75/100 %). |
| `terminate_on_success: bool` | `False` | End episode when all evaders caught simultaneously. |
| `boundary_threshold: float` | `0.9` | Per-axis position magnitude beyond which boundary penalty activates. |
| `boundary_scale: float` | `1.0` | Multiplicative coefficient on the exponential penalty. |
| `boundary_sharpness: float` | `2.0` | Exponent coefficient — higher is a sharper wall. |

### `@dataclass ObsLayout`

Slice metadata for a single agent's MPE2 observation vector. Returned by
`PursuitEvasionEnv.get_layout(agent)`.

| Field | Type | Meaning |
| --- | --- | --- |
| `self_vel` | `slice` | Own velocity (length 2). |
| `self_pos` | `slice` | Own position (length 2). |
| `landmark_rel` | `slice` | Visible landmark relative positions. |
| `other_rel` | `slice` | Visible other-agent relative positions. |
| `adversary_rel` | `slice` | Sub-slice of `other_rel` guaranteed to be adversary positions; equal to `other_rel` under PO. |
| `good_rel` | `slice` | Sub-slice of `other_rel` for good agents; empty under PO. |
| `good_vel` | `slice` | Good-agent velocities. |
| `is_adversary` | `bool` | `True` iff this layout describes an adversary. |
| `num_landmarks` | `int` | Landmark slot count. |
| `num_others` | `int` | Other-agent slot count. |
| `num_adv_in_other` | `int` | Adversary slot count at the start of `other_rel`. |
| `num_good_in_other` | `int` | Good-agent slot count in `other_rel` (0 under PO). |
| `num_good_vel` | `int` | Good-agent velocity slot count. |
| `partial_obs` | `bool` | `True` iff K-nearest agent masking is enabled. |

**Observation layout formulae:**

Partial observability (`num_agent_neighbors=K`, `num_landmark_neighbors=L`):

```
[ self_vel(2) | self_pos(2) | landmark_rel(2L) | other_rel(2K) | good_vel(2K) ]
```

Full observability (`num_agent_neighbors=None`) — for an observer of
type `adv?` with `A = num_adversaries`, `G = num_good`:

```
[ self_vel(2) | self_pos(2) | landmark_rel(2·num_obstacles) |
  adversary_rel(2·(A-1 if adv else A)) |
  good_rel(2·(G if adv else G-1)) |
  good_vel(2·(G if adv else G-1)) ]
```

### `class PursuitEvasionEnv` <- `PettingZooWrapper`

Specialisation of `simple_tag_v3` for equal-ability predator-prey
research with native K-nearest partial observability, optional
curriculum, optional early termination, and an exponential boundary
penalty added to each agent's reward every step.

```python
class PursuitEvasionEnv(PettingZooWrapper):
    def __init__(
        self,
        config: Optional[PursuitEvasionConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None: ...
```

**Warns:** `UserWarning` when `config.curriculum=True` and
`config.equal_ability=True` — equal-ability overwrites the curriculum
prey speeds on every reset. Does **not** raise.

**Extra attributes on top of `PettingZooWrapper`:**
- `pe_config: PursuitEvasionConfig`.

**Extra properties:**
- `adversary_agents -> List[str]` — names starting with `adversary`.
- `good_agents -> List[str]` — the complement.
- `curriculum_stage -> int` — pass-through to the underlying MPE2 env.

**Extra methods:**

| Signature | Behaviour |
| --- | --- |
| `get_layout(agent: str) -> ObsLayout` | Static slice metadata for `agent`. |
| `advance_curriculum() -> None` | Increment stage; effective on next reset. |
| `set_curriculum_stage(stage: int) -> None` | Jump to stage (clamped by MPE2). |

**Overridden behaviours:**
- `reset` additionally applies equal-ability after the base reset.
- `step` additionally subtracts the exponential boundary penalty
  `scale · Σ_axis (exp(sharpness · max(0, |xᵢ| - threshold)) - 1)`
  from each live agent's reward (and from `cumulative_rewards`).

### Turn-based adapters (`deap_pettingzoo.environments.turn_based`)

Two small wrappers that let the PettingZoo **parallel** evaluator
stack drive a strictly turn-based **AEC** env (`tictactoe_v3`,
`connect_four_v3`, OpenSpiel games via Shimmy, ...). Both are
returned implicitly by `build_env` when the registered `api` is
`"aec_masked"` or `"aec_sequential"` respectively; they are also
exported from `deap_pettingzoo.environments` for direct use.

```python
class MaskedSimultaneousAdapter:
    """Parallel-shaped facade over an AEC env.

    Every active agent appears in ``observations`` / ``rewards`` /
    ``infos`` each step, but only the current player's action from
    the ``actions`` dict actually executes — the others are ignored.
    ``infos[agent]["action_mask"]`` is populated on every step so
    ``MaskedDiscreteHead`` can zero illegal logits.
    """
    def __init__(self, aec_env): ...
    possible_agents: List[str]
    metadata: Dict[str, Any]
    def reset(self, seed=None) -> Tuple[Dict, Dict]: ...
    def step(self, actions) -> Tuple[Dict, Dict, Dict, Dict, Dict]: ...
    def close(self) -> None: ...
    def render(self) -> Any: ...

class SequentialTurnAdapter:
    """Single-agent-per-step parallel facade.

    Only the current player appears in the returned dicts; true turn
    semantics are preserved for TD-learning / AlphaZero-style
    experiments where only the mover gets an observation.
    """
    # same interface as MaskedSimultaneousAdapter
```

Use `api="aec_masked"` for EA / CMA-ES self-play (all agents show
up each step so shared-weights evaluators work unchanged); use
`api="aec_sequential"` when a downstream algorithm needs strict
turn semantics.

### Turn-based games via Shimmy (`deap_pettingzoo.contrib.openspiel`)

Opt-in bridge to [OpenSpiel](https://github.com/deepmind/open_spiel)
turn-based games via
[Shimmy](https://shimmy.farama.org/environments/open_spiel/). Nothing
is registered on import; call `register_openspiel_game` explicitly.

```python
def register_openspiel_game(
    name: str,
    *,
    game_name: str,
    api: Literal["aec_masked", "aec_sequential"] = "aec_masked",
    default_episode_len: int = 50,
    wrappers: Sequence[Callable[[Any], Any]] = (),
    overwrite: bool = False,
    notes: str = "",
    tags: Optional[Iterable[str]] = None,
    **extra_reg_kwargs,
) -> EnvSpec
```

- `name` — registry key (e.g. `"openspiel_connect_four"`).
- `game_name` — OpenSpiel short name (`"connect_four"`,
  `"tic_tac_toe"`, `"kuhn_poker"`, ...).
- `api` — validated against `{"aec_masked", "aec_sequential"}`;
  any other value raises `ValueError`.
- Extras: install with `pip install 'deap-pettingzoo[shimmy]'`. If
  `shimmy` or `pyspiel` are missing, the factory raises `ImportError`
  with an install hint.

Example:

```python
from deap_pettingzoo.contrib.openspiel import register_openspiel_game
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.evaluators import SelfPlayEvaluator

register_openspiel_game(
    "openspiel_connect_four",
    game_name="connect_four",
    api="aec_masked",
    tags=["openspiel", "turn_based", "zero_sum"],
)
wrapper = PettingZooWrapper("openspiel_connect_four")
evaluator = SelfPlayEvaluator(wrapper, aggregation="first_agent")
```

### JAX envs (`deap_pettingzoo.environments.jax_parallel`)

```python
@runtime_checkable
class JaxEnvProtocol(Protocol):
    agents: List[str]
    def reset(self, key) -> Tuple[Dict[str, Any], Any]: ...
    def step(self, key, state, actions) -> Tuple[Dict, Any, Dict, Dict, Dict]: ...
    def observation_space(self, agent) -> Any: ...
    def action_space(self, agent) -> Any: ...

class JaxEnvToPettingZooAdapter(ParallelEnv):
    def __init__(self, jax_env: JaxEnvProtocol, seed: int = 0) -> None: ...
```

Wraps any pure-functional JAX env (JaxMARL, Brax-MARL, ...) in
PettingZoo's `ParallelEnv` interface using a **per-step numpy
boundary**. The adapter owns a `jax.random.PRNGKey`, splits it on
every `reset`/`step`, and marshals dicts between JAX and numpy.
Intentionally un-`jit`ted: the boundary is where QDax's fast path
stops and existing torch/numpy learners (AgileRL, DEAP) take over.

### JaxMARL registration (`deap_pettingzoo.contrib.jaxmarl`)

Opt-in; nothing auto-registers on import.

```python
def register_jaxmarl_env(
    name: str,
    env_id: str,
    *,
    default_episode_len: int = 25,
    tags: Optional[Iterable[str]] = None,
    overwrite: bool = False,
    **jaxmarl_kwargs,
) -> EnvSpec
```

Wraps `jaxmarl.make(env_id, **jaxmarl_kwargs)` as a factory and
registers it with `api="jax_parallel"`. Raises `ImportError` with
install hint if `jax` or `jaxmarl` are missing. Install the extras
with `pip install 'deap-pettingzoo[jaxmarl]'` (pulls `jax` too).

```python
from deap_pettingzoo.contrib.jaxmarl import register_jaxmarl_env

register_jaxmarl_env(
    "jaxmarl_simple_tag",
    "MPE_simple_tag_v3",
    default_episode_len=25,
    num_good_agents=1,
    num_adversaries=3,
)
```

### QDax contrib (`deap_pettingzoo.contrib.qdax`)

Quality-diversity training via
[QDax](https://github.com/adaptive-intelligent-robotics/QDax).
Requires the `[qdax]` extra (also installs `jax` and `jaxmarl`).

#### Policy (`deap_pettingzoo.contrib.qdax.policy`)

```python
def make_flax_mlp(layer_sizes: Sequence[int], activation: str = "tanh"): ...
def init_flax_mlp_params(module, *, input_size: int, seed: int = 0): ...

def flat_count(layer_sizes: Sequence[int]) -> int: ...
def flat_to_pytree(flat: np.ndarray, layer_sizes: Sequence[int]): ...
def pytree_to_flat(params, layer_sizes: Sequence[int]) -> np.ndarray: ...

class FlaxPolicyAdapter(PolicyProtocol):
    def __init__(self, module, params, *,
                 action_space=None, masked: bool = False) -> None: ...
    def get_action(self, obs, action_mask=None): ...
```

`FlaxPolicyAdapter` satisfies `PolicyProtocol` so QDax individuals
drop into the same evaluators / co-evolution machinery as DEAP
genomes. `flat_to_pytree` / `pytree_to_flat` bridge the flat
vectors `FixedTopologyGenome` uses and the Flax pytree shape CMA-ME
and DEAP-style flat operators expect.

#### Descriptors (generic contract)

The library ships only the generic `DescriptorProtocol`
(`deap_pettingzoo.contrib.qdax.descriptors`): a callable
`(trajectory) -> array` exposing `.output_shape`, `.min_bd`, `.max_bd`
attributes that the trainer threads into `compute_euclidean_centroids`.
The caller injects a concrete descriptor via `descriptor_fn` (see
"Algorithm params" below). Example pursuit-evasion descriptors live
outside the library in `examples/qdax/descriptors.py`:

```python
# examples/qdax/descriptors.py
def formation_spread_and_mean_distance(
    num_pursuers: int, num_evaders: int
) -> Callable  # output_shape=2, min/max = (0,0)/(1,1)

def with_capture_efficiency(
    num_pursuers: int, num_evaders: int, capture_radius: float = 0.1
) -> Callable  # output_shape=3

def single_team_formation(num_agents: int) -> Callable  # output_shape=2
```

Both expect `trajectory["positions"]` of shape `(T, N, 2)` where
the first `num_pursuers` indices are pursuers. The shipped scoring
functions respect this ordering.

#### Scoring function builders

```python
# Fast path (requires a jax_parallel env; jit+vmap)
def make_jax_scoring_function(
    jax_env,
    module,
    *,
    episode_length: int,
    descriptor_fn,
    action_type: Literal["discrete", "continuous"] = "discrete",
    agents: Optional[List[str]] = None,
    state_extractor: Optional[Callable] = None,
) -> Callable  # (genotypes, key) -> (fitness, descriptor, extra_scores)

# Slow path (any PettingZoo env; Python rollout loop)
def make_numpy_scoring_function(
    env_name: str,
    *,
    layer_sizes: Sequence[int],
    episode_length: int,
    descriptor_fn,
    env_config: Optional[dict] = None,
    action_type: Literal["discrete", "continuous"] = "discrete",
    state_extractor: Optional[Callable] = None,
    activation: str = "tanh",
    num_workers: int = 1,
    parallel_backend: str = "process",
) -> Callable

# Tear down the persistent worker pool (if any) on the slow path:
def close_scoring_function(scoring_function: Callable) -> None
```

Both return the QDax 0.5.0 scoring contract: `(genotypes, key) ->
(fitness, descriptor, extra_scores)`. The slow-path callable
additionally exposes `.close()`, `._num_workers`, and
`._parallel_backend` for pool lifecycle management. With
`num_workers > 1` the executor is spun up lazily on first call and
reused across QD iterations; the `MAPElitesTrainer` /
`CMAMETrainer` wrap their loops in `try / finally` to call
`close_scoring_function` on exit. Worker-side rollouts ship a
picklable `_RolloutSpec` (env name, layer sizes, activation,
descriptor) and rebuild the env / Flax module via a module-global
cache (`_NUMPY_WORKER_CACHE`).

> **Fast path constraint.** `num_workers > 1` is **rejected** on the
> JAX fast path with a `ValueError` — the JAX rollout already
> batches the whole population through a single `jax.vmap`, so use
> `batch_size` to scale instead. Process-pool parallelism is only
> useful on the numpy path.

#### Trainers (`deap_pettingzoo.contrib.qdax.trainer`)

```python
class MAPElitesTrainer(BaselineTrainer): ...  # algorithm="map_elites"
class CMAMETrainer(BaselineTrainer):    ...  # algorithm="cma_me"
```

Both accept `TrainingConfig` and auto-dispatch between the fast
path (when the env's registered `api == "jax_parallel"`) and the
slow numpy path (with a `UserWarning`). Supported budget modes:
`BudgetMode.ENV_STEPS`, `BudgetMode.EVAL_EPISODES`.

`TrainingConfig.algorithm_params` keys:

| key | default | notes |
|-----|---------|-------|
| `hidden_sizes` | `(32, 32)` MAP-Elites / `(16, 16)` CMA-ME | Flax MLP hidden layer widths |
| `activation` | `"tanh"` | One of `"tanh"`, `"relu"` |
| `batch_size` | `32` | QDax emitter batch size |
| `init_batch_size` | `64` (MAP-Elites only) | Initial population |
| `grid_shape` | `(10, 10)` | Euclidean centroid grid |
| `num_centroids` | `None` | If set, use CVT centroids instead of grid |
| `iso_sigma` / `line_sigma` | `0.005` / `0.05` | MAP-Elites isoline variation |
| `sigma_g` | `0.1` (CMA-ME only) | CMA step size |
| `episode_length` | `25` | Rollout horizon |
| `descriptor_fn` | `None` (**required**) | Caller-supplied descriptor callable with `.min_bd`/`.max_bd`/`.output_shape` (e.g. from `examples/qdax/descriptors.py`) |
| `save_repertoire` | `True` | Pickle repertoire to `output_dir/repertoire.pkl` |
| `seed` | `0` | JAX PRNG seed |
| `num_workers` | `1` | Per-individual parallel rollouts on the **numpy slow path only**. `>1` spawns a `ProcessPoolExecutor` reused across QD iterations and torn down via `close_scoring_function` in a `try/finally`. Rejected on the JAX fast path with `ValueError`. |
| `parallel_backend` | `"process"` | One of `"process"` / `"dask"` (slow path only). |

`TrainingResult.metadata` contains `descriptor`, `grid_shape`,
`num_iterations`, `layer_sizes`, and (when `output_dir` and
`save_repertoire` are set) `repertoire_path`.

#### Co-evolution plug-in (`deap_pettingzoo.contrib.qdax.coevolution`)

Two classes let QDax repertoires participate in
`CoEvolutionTrainer`:

```python
class QDaxRepertoirePool(OpponentPool):
    def __init__(
        self,
        repertoire,
        policy_factory: Callable[[Any], PolicyProtocol],
        *,
        mode: Literal["uniform", "fitness"] = "uniform",
        temperature: float = 1.0,
    ) -> None: ...

class QDaxSideState(SideState):
    """Used implicitly when Side.optimizer is 'qdax_map_elites'
    or 'qdax_cma_me'. Runs one QDax iteration per co-evolution
    generation; publishes a QDaxRepertoirePool via current_pool().

    Experimental: the scoring path currently uses a centre-of-range
    placeholder descriptor (CoEvolutionEvaluator does not expose
    trajectories). Fitness is real, descriptor diversity is not.
    """
```

`Side` now accepts `optimizer="qdax_map_elites"` or
`optimizer="qdax_cma_me"`; `make_side_state` soft-imports this
module and raises `ImportError` with an install hint if the
`[qdax]` extra is missing.

#### Plugin seam — adding a co-evolution side

`make_side_state` and `Side.__post_init__` validation are both driven by the
**open** `_SIDE_STATE_FACTORIES` registry (a token -> zero-arg loader map).
Register a new side without editing `sides.py`:

```python
from deap_pettingzoo.baselines.coevolution import register_side_state
register_side_state("myopt", MySideState)          # class, dotted str, or loader
Side("s", ("agent",), optimizer="myopt")           # now validates
```

Or register the standalone algorithm and its side in one call — the side is
forwarded to `register_side_state` under `coevolution_optimizer`:

```python
from deap_pettingzoo import register_algorithm
register_algorithm(
    "myalgo", label="My Algorithm", family="evolutionary",
    trainer=MyTrainer, capabilities=MyTrainer.CAPS,
    coevolution_optimizer="myopt", side_state=MySideState,
)
```

A programmatic call affects only the calling process (like a programmatic
`register_env`); for worker-safe registration ship a `register()` entry point
under `deap_pettingzoo.algorithms` / `deap_pettingzoo.coevolution_sides`
(discovered once at catalog import in every process). See
[architecture.md](../architecture.md#adding-an-algorithm--side-plugin-seam).

---

### NEAT / HyperNEAT / ES-HyperNEAT (`deap_pettingzoo.contrib.neat`)

Install with `pip install deap-pettingzoo[neat]` (requires `neat-python>=2.0`).
All public classes are lazy-imported via PEP 562; importing the package
without `neat` installed does **not** fail until a class is accessed.

#### Config (`deap_pettingzoo.contrib.neat.config`)

```python
def build_neat_config(input_size: int, output_size: int, *,
                      pop_size: int = 50,
                      seed: Optional[int] = None,
                      overrides: Optional[dict] = None) -> neat.Config
```
Build a `neat.Config` for standard neuroevolution.  Creates a temporary
`.ini` config file with sensible defaults and returns the parsed `neat.Config`.

```python
def build_cppn_config(input_size: int, output_size: int, *,
                      pop_size: int = 50,
                      seed: Optional[int] = None,
                      overrides: Optional[dict] = None) -> neat.Config
```
Build a `neat.Config` for **CPPN** evolution (HyperNEAT / ES-HyperNEAT).
Adds periodic/Gaussian activation functions (`sin`, `gauss`, `abs`, `square`)
and wider weight ranges appropriate for pattern-producing networks.
CPPN inputs are always `[x1, y1, x2, y2, bias]` (5 inputs), output is a
single weight value (1 output).

#### Substrate (`deap_pettingzoo.contrib.neat.substrate`)

```python
class Substrate:  # re-exported from _pureples.substrate
    input_coordinates: List[Tuple[float, float]]
    output_coordinates: List[Tuple[float, float]]
    hidden_coordinates: List[List[Tuple[float, float]]]

def make_uniform_substrate(num_inputs: int, num_outputs: int, *,
                           hidden_layers: Optional[Sequence[int]] = None) -> Substrate
def make_2d_grid_substrate(grid_shape: Tuple[int, int], num_outputs: int, *,
                           hidden_layers: ...) -> Substrate
def substrate_for_env(env_obs_size: int, env_act_size: int, *,
                      layout: str = "uniform", hidden_layers: ...) -> Substrate
```

#### Policies

```python
class NeatPolicy(PolicyProtocol):
    """Wraps neat.nn.FeedForwardNetwork + ActionHead."""
    def __init__(self, genome, config, *, action_space=None, masked=False)
    def get_action(obs, *, action_mask=None) -> Any
    def reset() -> None

class HyperNeatPolicy(PolicyProtocol):
    """CPPN -> substrate-query -> RecurrentNetwork phenotype."""
    def __init__(self, genome, config, substrate, *,
                 action_space=None, masked=False,
                 activation="sigmoid", weight_threshold=0.1)

class ESHyperNeatPolicy(PolicyProtocol):
    """CPPN -> ESNetwork quadtree -> RecurrentNetwork phenotype."""
    def __init__(self, genome, config, substrate, *,
                 action_space=None, masked=False, es_params=None)
```

``weight_threshold`` controls the minimum absolute CPPN output below
which phenotype connection weights are zeroed.  Lower values let through
more connections at the cost of noise.  The default 0.1 allows ~90 % of
random CPPN outputs to create non-zero weights (the original pureples
value 0.2 allows only ~60 %).  Pass it to HyperNEAT via
``algorithm_params["weight_threshold"]``; for ES-HyperNEAT it belongs
inside the ``es_params`` dict.

#### Trainers

```python
class NEATTrainer(BaselineTrainer):
    """NEAT population trainer.  Budget -> generations.

    Parameters
    ----------
    num_workers : int (default 1)
        When > 1, genome evaluation is distributed across a
        ``ProcessPoolExecutor``.  Each worker rebuilds the
        environment, NEAT config, and policy independently.
        When 1, evaluation runs sequentially in the main process.
    """

class HyperNEATTrainer(NEATTrainer):
    """HyperNEAT trainer (CPPN + substrate).

    Inherits ``num_workers`` from :class:`NEATTrainer`.
    """

class ESHyperNEATTrainer(NEATTrainer):
    """ES-HyperNEAT trainer (CPPN + quadtree substrate).

    Inherits ``num_workers`` from :class:`NEATTrainer`.
    """
```

All three accept standard `TrainingConfig` with `algorithm_params`
including:

| Field | Type | Default | Applies to |
|---|---|---|---|
| ``pop_size`` | ``int`` | 50 | all |
| ``n_generations`` | ``int`` | auto | all |
| ``eval_episodes_per_genome`` | ``int`` | 1 | all |
| ``seed`` | ``int`` | None | all |
| ``neat_config_overrides`` | ``dict`` | {} | all (NEAT / CPPN config) |
| ``substrate_hidden_layers`` | ``list[int] \| None`` | ``[4, 4]`` | HyperNEAT |
| ``substrate_layout`` | ``"uniform" \| "2d_grid"`` | ``"uniform"`` | HyperNEAT |
| ``phenotype_activation`` | ``str`` | ``"sigmoid"`` | HyperNEAT |
| ``weight_threshold`` | ``float`` | 0.1 | HyperNEAT |
| ``es_params`` | ``dict`` | ``ES_HYPERNEAT_PARAMS_DEFAULTS`` | ES-HyperNEAT |

``es_params`` accepts the same keys as
:data:`~deap_pettingzoo.contrib.neat._pureples.es_hyperneat.ES_HYPERNEAT_PARAMS_DEFAULTS`
including ``initial_depth``, ``max_depth``, ``variance_threshold``,
``band_threshold``, ``division_threshold``, ``max_weight``,
``activation`` (hidden-node activation), and ``weight_threshold``.

All three support the standard ``RecordingConfig`` on
``TrainingConfig.recording``.  When ``enabled=True``, a custom
``neat.reporting.BaseReporter`` hooks ``post_evaluate`` and forwards each
generation's champion to the shared ``RecordingController``, which decides —
per ``RecordingConfig.mode`` — whether to capture it.  Frames are written to
``{output_dir}/{run_id}/frames/{episode_prefix}episode_NNNN.{fmt}``
via :class:`UnifiedEvaluator`.

#### Co-evolution sides

```python
class NeatSideState(SideState):
    """Side.optimizer = "neat" — NEAT population as co-evolution participant."""

class HyperNeatSideState(NeatSideState):
    """Side.optimizer = "hyperneat"."""

class ESHyperNeatSideState(NeatSideState):
    """Side.optimizer = "es_hyperneat"."""
```

Snapshot kinds for parallel evaluation: `"neat"`, `"hyperneat"`,
`"es_hyperneat"` — handled by `rebuild_opponent_policy`.

#### AgileRL HPO (`deap_pettingzoo.baselines.agilerl_hpo_trainer`)

```python
class EvolveHPOTrainer(AgileRLTrainer):       # abstract base for HPO
class MADDPGHPOTrainer(EvolveHPOTrainer):     # MADDPG (off-policy) + evolvable topology
class MATD3HPOTrainer(EvolveHPOTrainer):      # MATD3 (off-policy) + evolvable topology
class IPPOHPOTrainer(EvolveHPOTrainer):       # IPPO (on-policy) + evolvable topology
class GroupedIPPOHPOTrainer(IPPOHPOTrainer):  # prefix-grouped IPPO (on-policy)
```

Population-based training with `agilerl.hpo.Mutations` and
`agilerl.hpo.tournament.TournamentSelection`.  Hyperparameter ranges
configured via `algorithm_params["hp_config"]`.  All four are catalog
algorithms (`maddpg-hpo`, `matd3-hpo`, `ippo-hpo`, `ippo-grouped-hpo`)
exposed in the experiment-setup menu; the off-policy variants use a
replay-buffer rollout loop and the on-policy IPPO variants share the
`agilerl_onpolicy.train_agent_on_policy` helper.  Co-evolution side
is `RLEvolveHPOSideState` (optimizer `"rl_hpo"`).

---

## Evaluators

Module: `deap_pettingzoo.evaluators`.

### `class BaseEvaluator` (ABC)

```python
class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, individual: Any, **kwargs) -> Tuple[float, ...]: ...
```

Contract: return a DEAP-style fitness tuple. `individual` must expose a
`.genome` attribute.

### `class SingleEpisodeEvaluator`

```python
class SingleEpisodeEvaluator(BaseEvaluator):
    VALID_AGGREGATIONS = ["sum", "mean", "min", "max", "team"]

    def __init__(
        self,
        env_wrapper: PettingZooWrapper,
        max_steps: int = 500,
        aggregation_method: str = "sum",
        team_mapping: Optional[Dict[str, str]] = None,
    ) -> None: ...

    def evaluate(
        self,
        individual: Any,
        seed: Optional[int] = None,
    ) -> Tuple[float]: ...
```

**Behaviour:** shares one genome across **all** agents. All agents
must therefore have **identical** observation sizes; heterogeneous
envs (e.g. `mpe2.simple_adversary_v3`: adversaries at 16-dim, good
agent at 14-dim) raise `ValueError` pointing at two migration paths
— register the env with SuperSuit padding wrappers
(`pad_observations_v0`, `pad_action_space_v0`, optionally
`agent_indicator_v0`) via `EnvSpec.wrappers`, **or** evolve per-role
with `MixedPolicyEvaluator`. Runs a single episode (up to
`max_steps`), then aggregates cumulative per-agent rewards according
to `aggregation_method`. For `team`, requires `team_mapping` and
returns the signed two-team reward difference.

**Raises:** `ValueError` on `None` individual/genome, invalid
aggregation, or heterogeneous observation sizes; `RuntimeError`
wraps any exception raised during rollout.

### `class SelfPlayEvaluator`

```python
class SelfPlayEvaluator(BaseEvaluator):
    VALID_AGGREGATIONS = ("sum", "mean", "first_agent", "team_mapping")

    def __init__(
        self,
        env_wrapper: PettingZooWrapper,
        max_steps: int = 200,
        aggregation: str = "first_agent",
        team_mapping: Optional[Dict[str, str]] = None,
        team: Optional[str] = None,
        masked: bool = True,
        flatten_obs: bool = True,
    ) -> None: ...

    def evaluate(
        self,
        individual: Any,
        seed: Optional[int] = None,
    ) -> Tuple[float]: ...
```

Canonical EA evaluator for strictly turn-based AEC games (tic-tac-toe,
connect-four, kuhn-poker, ...). Drives **every** agent with a single
shared-weights `FeedforwardNeuralPolicy`. When `masked=True` (default),
the policy is built with a `MaskedDiscreteHead` that consumes
`infos[agent]["action_mask"]` (zeroing illegal logits to `-inf` before
`argmax`); when `flatten_obs=True` (default), observations are
flattened to 1-D via `np.ravel` so the MLP can consume 2-D / 3-D board
representations without a separate wrapper env.

Pair with `api="aec_masked"` on registration (or
`deap_pettingzoo.contrib.openspiel.register_openspiel_game` for
OpenSpiel games via Shimmy). Aggregation is the natural axis for
self-play fitness:

- `sum` / `mean` — cooperative aggregation across all agents.
- `first_agent` — reward of `possible_agents[0]` only; the right
  choice for symmetric zero-sum games ("how good a player-1 am I?").
- `team_mapping` — sum rewards for agents whose
  `team_mapping[agent] == team`. Requires both kwargs.

**Raises:** `ValueError` on invalid aggregation or missing team
args; `RuntimeError` wraps any exception raised during rollout.

### `class MultiEpisodeEvaluator`

```python
class MultiEpisodeEvaluator(BaseEvaluator):
    VALID_AGGREGATIONS = ["mean", "median", "min", "max"]

    def __init__(
        self,
        env_wrapper: PettingZooWrapper,
        num_episodes: int = 5,
        aggregation_method: str = "mean",
        base_seed: int = 42,
        **single_eval_kwargs,
    ) -> None: ...

    def evaluate(self, individual: Any) -> Tuple[float]: ...
    def evaluate_with_stats(self, individual: Any) -> Tuple[float, float, float, float]:
        """Return (mean, std, min, max) across episodes."""
```

Runs `num_episodes` deterministic rollouts with seeds
`base_seed, base_seed+1, …` via an internally owned
`SingleEpisodeEvaluator` constructed from `single_eval_kwargs`.

### `@dataclass MixedPolicyEvalResult`

```python
@dataclass
class MixedPolicyEvalResult:
    fitness: float
    learner_reward: float    # == fitness
    scripted_reward: float
    steps: int
```

### `class MixedPolicyEvaluator`

```python
# The factory takes the per-agent action space so the constructed
# policy can attach the correct ActionHead (DiscreteHead / BoxHead).
LearnerPolicyFactory = Callable[[Any, str, Any], PolicyProtocol]

# Factory aliases (factory-path construction)
EnvFactory            = Callable[[], PettingZooWrapper]
ScriptedPolicyFactory = Callable[[PettingZooWrapper], Dict[str, PolicyProtocol]]

class MixedPolicyEvaluator(BaseEvaluator):
    def __init__(
        self,
        learner_agent_prefixes: List[str],
        learner_factory: LearnerPolicyFactory,
        max_steps: int = 500,
        # --- live-object path (default; not picklable across workers) ---
        env_wrapper: Optional[PettingZooWrapper] = None,
        scripted_policies: Optional[Dict[str, PolicyProtocol]] = None,
        # --- factory path (picklable; required for parallel eval) ---
        env_factory: Optional[EnvFactory] = None,
        scripted_policy_factory: Optional[ScriptedPolicyFactory] = None,
    ) -> None: ...

    def evaluate(
        self,
        individual: Any,
        seed: Optional[int] = None,
    ) -> Tuple[float]: ...

    def run_episode(
        self,
        individual: Any,
        seed: Optional[int] = None,
    ) -> MixedPolicyEvalResult: ...
```

**Two construction paths:**

| Path | When to use | Params |
|------|-------------|--------|
| **Live-object** | Single-process (default) | `env_wrapper`, `scripted_policies` |
| **Factory** | Parallel evaluation (picklable) | `env_factory`, `scripted_policy_factory` |

Exactly one path must be used; mixing both raises `ValueError`.

**Live-object path:**

```python
evaluator = MixedPolicyEvaluator(
    env_wrapper=wrapper,
    scripted_policies={"agent_0": my_policy},
    learner_agent_prefixes=["adversary"],
    learner_factory=_factory,
)
```

**Factory path** (required for `make_mixed_parallel_evaluate_map`):

```python
evaluator = MixedPolicyEvaluator(
    env_factory=build_env,                      # module-level def
    scripted_policy_factory=build_scripted,     # module-level def
    learner_agent_prefixes=["adversary"],
    learner_factory=_factory,
    max_steps=200,
)
```

> **Pickle constraint**: `env_factory`, `scripted_policy_factory`, and
> `learner_factory` must be **module-level** callables (not lambdas or
> closures) to be picklable on Windows spawn workers.

**Semantics:** agents whose name starts with any entry in
`learner_agent_prefixes` are the learner side; everyone else is driven
by the scripted policies. The learner factory is called with
`(individual.genome, agent_id, env_wrapper.action_space(agent_id))`
for each learner agent at the start of every episode. DEAP fitness is the
**sum of learner-side rewards only**; scripted rewards are returned
separately on `MixedPolicyEvalResult` for diagnostics.

With the **factory path**, `run_episode` builds a **fresh env** per call
and closes it on exit — safe for concurrent subprocess workers. The
constructor also builds one validation env (stored as `self.env_wrapper`)
to check agent coverage; callers should close it when done:
`evaluator.env_wrapper.close()`.

**Raises:**
- `ValueError` at construction time if neither `env_wrapper` nor
  `env_factory` is supplied.
- `ValueError` at construction time if both `env_wrapper` and
  `env_factory` are supplied simultaneously.
- `ValueError` at construction time if `env_factory` is given without
  `scripted_policy_factory`.
- `ValueError` at construction time if `scripted_policies` is missing
  entries for any non-learner agent.
- `ValueError` at construction time if `learner_agent_prefixes` matches
  no agents.
- `ValueError` in `evaluate`/`run_episode` if `individual.genome` is
  missing or `None`.

---

## Integration

Module: `deap_pettingzoo.integration.deap_integration`. Plumbing
between `FixedTopologyGenome` and DEAP's `base`/`creator`/`tools`.

### Functions

```python
def setup_deap_creators() -> None:
    """Idempotently create `creator.FitnessMax` (weights=(1.0,)) and
    `creator.Individual` (object with `genome=None` + fitness)."""
```

```python
def create_deap_individual(
    layer_sizes: List[int],
    activation: str = "tanh",
    weight_range: tuple = (-1.0, 1.0),
    random_seed: Optional[int] = None,
) -> creator.Individual:
    """Build a `FixedTopologyGenome` and attach it to a new individual."""
```

```python
def evaluate_individual(
    individual: Any,
    evaluator: BaseEvaluator,
    **kwargs,
) -> Tuple[float, ...]:
    """Thin wrapper over `evaluator.evaluate(individual, **kwargs)`."""
```

```python
def create_toolbox(
    genome_config: Dict[str, Any],
    evaluator: BaseEvaluator,
    selection_config: Optional[Dict[str, Any]] = None,
    crossover_config: Optional[Dict[str, Any]] = None,
    mutation_config: Optional[Dict[str, Any]] = None,
) -> deap.base.Toolbox:
    """Wire a full DEAP toolbox (individual/population/evaluate/mate/mutate/select/clone)."""
```

Expected keys:

- `genome_config`: `layer_sizes` (required), `activation` (`"tanh"`),
  `weight_range` (`(-1.0, 1.0)`), `random_seed` (`None`).
- `selection_config`: `method` (``\"tournament\"``), ``tournament_size``
  (``3``). Supported ``method`` values:

  * Single-objective: ``\"tournament\"``, ``\"roulette\"``, ``\"best\"``,
    ``\"random\"``, ``\"sus\"``.
  * Multi-objective: ``\"nsga2\"``, ``\"spea2\"`` (requires multi-weight
    ``setup_deap_creators`` + ``FitnessSpec``).
  * Lexicase: ``\"lexicase\"``, ``\"epsilon_lexicase\"`` (requires
    multi-dimensional fitness vectors).
- `crossover_config`: `method` (`"blend"`), `alpha` (`0.5`), `indpb`
  (`0.5` for `"uniform"`), `eta` (`15.0` for `"sbx"`). Supported
  `method` values: `"blend"`, `"one_point"`, `"two_point"`,
  `"uniform"`, `"sbx"` (simulated-binary-bounded), `"es_blend"`.
- `mutation_config`: `method` (`"gaussian"`), `mu` (`0.0`), `sigma`
  (`0.2`), `indpb` (`0.1`), `eta` (`15.0` for `"polynomial_bounded"`).
  Supported `method` values: `"gaussian"`, `"polynomial_bounded"`.

Non-`blend`/`gaussian` operators are dispatched to DEAP built-ins via
internal `_cx_on_weights` / `_mut_on_weights` adapters that extract the
genome's flat weight vector, mutate it in place, then write it back.
Omitting any `method` key preserves the historical
tournament/blend/gaussian defaults.

Registered operators:

| Key | Function |
| --- | --- |
| `individual` | Builds a fresh `create_deap_individual`; per-call seed increment if `random_seed` given. |
| `population` | `tools.initRepeat(list, toolbox.individual)`. |
| `evaluate` | `evaluate_individual(evaluator=evaluator)`. |
| `mate` | Dispatched by `crossover_config['method']`. |
| `mutate` | Dispatched by `mutation_config['method']`. |
| `select` | Dispatched by `selection_config['method']`. |
| `clone` | `clone_individual`. |

```python
def crossover_blend(ind1, ind2, alpha: float = 0.5) -> tuple:
    """BLX-α blend crossover on flat weight vectors; in-place on both."""

def mutate_gaussian(individual, mu=0.0, sigma=0.2, indpb=0.1) -> tuple:
    """Per-weight Bernoulli mask + N(mu, sigma) additive noise; in-place."""

def clone_individual(individual) -> creator.Individual:
    """Deep copy via `genome.clone()`; preserves fitness if valid."""
```

### Parallel evaluation helpers

```python
def make_parallel_evaluate_map(
    executor: Any,
    evaluator: SingleEpisodeEvaluator,
    eval_episodes: int = 1,
    seed: Optional[int] = None,
) -> Callable:
    """Build a ``map``-shaped callable for parallel ``SingleEpisodeEvaluator`` eval.

    Compatible with ``toolbox.register("map", pmap)``."""
```

```python
def make_mixed_parallel_evaluate_map(
    executor: Any,
    evaluator: MixedPolicyEvaluator,
    *,
    eval_episodes: int = 1,
    step_sink: Optional[List[int]] = None,
) -> Callable:
    """Build a ``map``-shaped callable for parallel ``MixedPolicyEvaluator`` eval.

    The evaluator **must** have been constructed with the factory path
    (``env_factory`` + ``scripted_policy_factory`` + ``learner_factory``
    as module-level callables). Live-object construction raises ``ValueError``.

    The returned callable has signature
    ``pmap(evaluate_fn, individuals) -> list`` — compatible with
    ``toolbox.register("map", pmap)``.

    When ``step_sink`` (a 1-element mutable list) is provided, the real
    per-task env-step deltas returned by each worker are summed into
    ``step_sink[0]`` — the consumed-step (``budget_on_consumed``)
    budgeting hook, mirroring ``make_parallel_evaluate_map``.

    Example::

        evaluator = MixedPolicyEvaluator(
            env_factory=build_env,
            scripted_policy_factory=build_scripted_evaders,
            learner_agent_prefixes=["adversary"],
            learner_factory=_learner_factory,
            max_steps=200,
        )
        with make_executor(n_workers) as executor:
            pmap = make_mixed_parallel_evaluate_map(
                executor, evaluator, eval_episodes=3
            )
            toolbox.register("map", pmap)
    """
```

**Raises:**
- `TypeError` if `evaluator` is not a `MixedPolicyEvaluator`.
- `ValueError` if the evaluator was built with the live-object path.
- `ValueError` if any factory is not a module-level callable (lambda or
  closure cannot be pickled across processes).

### CMA-ES helpers — `integration.cma_integration`

Companion module that wires a `FixedTopologyGenome` flat weight vector
into DEAP's built-in CMA-ES (`deap.cma.Strategy`). Creates a dedicated
`creator.CMAIndividual` (list subclass with `fitness=creator.FitnessMax`
and `genome=None`) so that `deap.cma.Strategy.update` can rebuild the
sample matrix via `numpy.array(population)` while each individual still
exposes the `.genome` attribute required by evaluators.

```python
def setup_cma_strategy(
    genome_config: Dict[str, Any],
    sigma0: float = 0.3,
    population_size: Optional[int] = None,
    centroid: Optional[np.ndarray] = None,
) -> Tuple[deap.cma.Strategy, int]:
    """Build a CMA strategy sized to the genome's weight count.

    Returns (strategy, dimension). Defaults: zero centroid; DEAP's
    default lambda_ (4 + int(3*log(N))) when population_size is None.
    Raises ValueError on missing `layer_sizes` or centroid shape mismatch."""

def make_cma_ind_init(
    genome_config: Dict[str, Any],
) -> Callable[[Iterable[float]], creator.CMAIndividual]:
    """Closure for `strategy.generate(ind_init)` that materialises a
    CMAIndividual carrying its weights AND a matching FixedTopologyGenome
    attached as `.genome`."""

def register_cma_toolbox(
    strategy: deap.cma.Strategy,
    ind_init: Callable[[Iterable[float]], Any],
    evaluator: BaseEvaluator,
) -> deap.base.Toolbox:
    """Toolbox with `generate` / `update` / `evaluate` pre-registered
    for the CMA-ES generate-update loop."""

def make_cma_stats() -> tools.Statistics:
    """`tools.Statistics` with `avg` / `std` / `min` / `max` registered —
    same keys as EATrainer's fitness_history records."""
```

Typical loop (see `examples/`):

```python
strategy, dim = setup_cma_strategy(genome_config, sigma0, population_size)
ind_init = make_cma_ind_init(genome_config)
toolbox  = register_cma_toolbox(strategy, ind_init, evaluator)

for _ in range(num_generations):
    population = toolbox.generate()
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)
    toolbox.update(population)
```

---

### PSO Integration

Module: `deap_pettingzoo.integration.pso_integration`. Mirrors the CMA-ES
integration but drives a particle swarm instead.

```python
def setup_pso_swarm(
    genome_config, population_size=50, phi1=2.0, phi2=2.0,
    weight=0.7298, smin=None, smax=None, seed=None,
) -> Tuple[Dict[str, Any], int]

def make_pso_ind_init(genome_config, pso_state) -> Callable[[], Any]

def register_pso_toolbox(
    pso_state, ind_init, evaluator,
) -> deap.base.Toolbox

def make_pso_stats() -> tools.Statistics
```

Typical loop:

```python
pso_state, dim = setup_pso_swarm(genome_config)
ind_init = make_pso_ind_init(genome_config, pso_state)
toolbox = register_pso_toolbox(pso_state, ind_init, evaluator)

population = toolbox.generate()
for _ in range(num_generations):
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)
    toolbox.update(population)
```

### Multi-objective fitness

Module: `deap_pettingzoo.integration.objectives`.

```python
@dataclass
class ObjectiveSpec:
    name: str
    weight: float = 1.0
    source: str = "reward"  # "reward" | "parsimony" | "reward_component"
    component_key: Optional[str] = None

@dataclass
class FitnessSpec:
    objectives: Optional[List[ObjectiveSpec]] = None
    lexicase_mode: Optional[str] = None  # None | "per_episode" | "per_agent"
    reward_components_fn: Optional[Callable] = None

def compute_parsimony(genome) -> float
def build_fitness_vector(aggregated_reward, ...) -> Tuple[float, ...]
```

---

## Baselines

Module: `deap_pettingzoo.baselines`. Unified API for running EA and RL
algorithms under identical budgets and evaluating the resulting
policies with a shared evaluator. Public symbols re-exported from
`deap_pettingzoo.baselines.__init__`.

### Data models — `baselines.base`

```python
class BudgetMode(Enum):
    ENV_STEPS      = "env_steps"
    EVAL_EPISODES  = "eval_episodes"
    WALL_CLOCK     = "wall_clock"
```

```python
@dataclass
class TrainingConfig:
    env_name: str
    budget_mode: BudgetMode
    budget_value: int             # must be > 0
    num_seeds: int = 1            # must be > 0
    seed: Optional[int] = None                    # first-class; read by EA/CMA/PSO/AgileRL
    max_steps_per_episode: Optional[int] = None   # first-class; None -> trainer's own default
    coevolution_config: Optional[CoEvolutionConfig] = None  # typed entry point for CoEvolutionTrainer
    algorithm_params: Dict[str, Any] = {}         # genuinely algorithm-specific knobs only
    env_config: Dict[str, Any] = {}
    continuous: bool = False      # shortcut; merges {"continuous_actions": True}
    output_dir: Optional[Path] = None
    render: bool = False
    recording: RecordingConfig = RecordingConfig()
    probe: ProbeConfig = ProbeConfig()
    budget_on_consumed: bool = False  # opt-in real-tick budgeting (ENV_STEPS only)


@dataclass
class ProbeConfig:
    """Opt-in periodic held-out champion evaluation ("probe"). Every
    `every_n_steps` cumulative env steps (plus a baseline probe at the first
    progress event and a guaranteed end-of-run probe) the frozen current
    champion plays `n_episodes` episodes on a separate env instance — never
    charged to the training budget — and one row lands in
    `<run_dir>/probe.csv`. Driven by the shared `ProbeController`
    (`deap_pettingzoo/baselines/probing.py`); see
    `docs/api/logging_and_analysis.md` -> "Held-out champion probe"."""
    enabled: bool = False
    every_n_steps: Optional[int] = None   # required when enabled
    n_episodes: int = 16
    env_name: Optional[str] = None        # held-out env override (default: training env)
    env_config: Optional[Dict[str, Any]] = None
    info_keys: tuple = ()                 # env-info task metrics, episode-max aggregation
    seed: Optional[int] = None            # fixed per run -> CRN across checkpoints


@dataclass
class RecordingConfig:
    """Opt-in champion-rollout video recording. Cadence is algorithm-agnostic,
    selected by `mode` and driven by the shared `RecordingController`. See
    `docs/api/logging_and_analysis.md#framerecorder`."""
    enabled: bool = False
    mode: RecordingMode = RecordingMode.COUNT  # cadence policy (same for every algorithm)
    every_n_steps: Optional[int] = None        # INTERVAL mode: record every N env steps
    max_recordings: Optional[int] = None       # COUNT mode: N captures evenly spread (default 10)
    fps: int = 20
    fmt: str = "mp4"
    max_frames_per_episode: Optional[int] = None
    max_dim: Optional[int] = None              # downscale largest axis to this before buffering
    episode_prefix: str = ""                   # "" -> trainer uses "train_"


class RecordingMode(str, Enum):
    COUNT = "count"        # N (= max_recordings) videos spread evenly over the run (default)
    EVERY = "every"        # one per generation / iteration, no cap
    INTERVAL = "interval"  # one every every_n_steps env steps
    FINAL = "final"        # only the end-of-run champion rollout
```

The cadence is expressed once on the normalized progress axis shared by every
trainer (`ProgressEvent.fraction` and `ProgressEvent.env_steps`), so a "generation"
(EA/CMA/PSO/NEAT/coevolution) and an "iteration" (AgileRL env steps) behave
identically. The `RecordingController` (`deap_pettingzoo/baselines/recording.py`)
owns the recorder, the `UnifiedEvaluator`, the schedule, episode-offset bookkeeping,
sidecar metadata, and best-effort error handling; each trainer reports progress
through `BaselineTrainer._on_progress(callback, event, champion)`.

The `ProbeController` (`deap_pettingzoo/baselines/probing.py`) rides the same
`_on_progress` champion hook: any trainer that supports mid-run recording also
supports the held-out probe. QDax MAP-Elites probes mid-loop with buffered rows
(its logger only exists at persistence time); co-evolution does not use the
controller (a mixed-side rollout has no single champion score) and instead
passes `{"logger", "env_steps_consumed"}` context to three-argument
`generation_callback`s so callers can write the same `probe.csv` schema
per side.

**`continuous` shortcut.** Setting `continuous=True` merges
`{"continuous_actions": True}` into `env_config` at construction time
(MPE2 reads this kwarg to switch between the `Discrete` and `Box`
action variants). If the caller also sets
`env_config["continuous_actions"]` explicitly, the two must agree —
`continuous=True` with `env_config["continuous_actions"]=False` raises
`ValueError`. Setting `env_config["continuous_actions"]=True` without
the flag keeps the env_config authoritative and back-syncs
`continuous=True` so downstream introspection stays consistent.

```python
@dataclass
class TrainingResult:
    algorithm: str
    policy: PolicyProtocol
    metrics: Dict[str, Any]
    metadata: Dict[str, Any] = {}
    seed: Optional[int] = None
    training_time: Optional[float] = None
```

```python
@dataclass
class EvalResult:
    mean_reward: float
    std_reward:  float
    min_reward:  float
    max_reward:  float
    num_episodes: int
    episode_rewards: List[float] = []
    episode_lengths: List[int]   = []

    def to_dict(self) -> Dict[str, Any]: ...
```

```python
@dataclass
class ComparisonConfig:
    algorithms:   List[str]       # non-empty
    env_name:     str
    budget_mode:  Union[BudgetMode, str]
    budget_value: int             # > 0
    num_seeds:    int = 3         # > 0
    eval_episodes: int = 100      # > 0
    algorithm_configs: Dict[str, Dict[str, Any]] = {}
    output_dir: Optional[Path] = None
```

All four dataclasses raise `ValueError` from `__post_init__` on any
out-of-range numeric field.

### Protocols / ABCs — `baselines.base`

#### `PolicyProtocol`

```python
class PolicyProtocol(Protocol):
    def get_action(self, observation: np.ndarray) -> Any: ...
    def reset(self) -> None: ...
```

Structural type; implementations need not inherit. Satisfied by
`FeedforwardNeuralPolicy`, `APFEvaderPolicy`,
`baselines.ea_trainer.GenomePolicyWrapper`, and
`AgileRLPolicyWrapper`.

#### `BaselineTrainer` (ABC)

```python
class BaselineTrainer(ABC):
    @abstractmethod
    def train(self, config: TrainingConfig) -> TrainingResult: ...
    @abstractmethod
    def supports_budget_mode(self, mode: BudgetMode) -> bool: ...
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]: ...
```

`get_capabilities()` returns a dict with boolean keys
`discrete_action`, `continuous_action`, `needs_global_state`,
`on_policy`.

#### `BaselineEvaluator` (ABC)

```python
class BaselineEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        policy: PolicyProtocol,
        env_name: str,
        num_episodes: int,
        seed: Optional[int] = None,
        render: bool = False,
    ) -> EvalResult: ...
```

### EA trainer — `baselines.ea_trainer`

```python
class GenomePolicyWrapper:  # PolicyProtocol
    def __init__(self, genome: FixedTopologyGenome, env_name: str) -> None: ...
    def get_action(self, observation: np.ndarray) -> Any: ...
    def reset(self) -> None: ...        # no-op
    @property
    def genome(self) -> FixedTopologyGenome: ...
```

```python
class EATrainer(BaselineTrainer):
    def __init__(
        self,
        hidden_size: int = 64,
        population_size: int = 50,
        tournament_size: int = 3,
        crossover_alpha: float = 0.5,
        mutation_sigma: float = 0.2,
        mutation_indpb: float = 0.1,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.3,
        strategy: str = "simple",              # "simple" | "mu_plus_lambda" | "mu_comma_lambda"
        mu: Optional[int] = None,              # defaults to population_size
        lambda_: Optional[int] = None,         # defaults to population_size
        selection_method: str = "tournament",  # see create_toolbox
        crossover_method: str = "blend",       # see create_toolbox
        mutation_method: str = "gaussian",     # see create_toolbox
        selection_params: Optional[Dict[str, Any]] = None,
        crossover_params: Optional[Dict[str, Any]] = None,
        mutation_params: Optional[Dict[str, Any]] = None,
        num_workers: int = 1,                  # parallel evaluation
        parallel_backend: str = "process",     # "process" | "dask"
        scheduler_address: Optional[str] = None,  # remote Dask scheduler
    ) -> None: ...
```

`strategy="simple"` preserves the in-house generational loop;
`"mu_plus_lambda"` and `"mu_comma_lambda"` delegate to
`deap.algorithms.eaMuPlusLambda` / `eaMuCommaLambda` respectively with
`cxpb=crossover_rate`, `mutpb=mutation_rate`, and the supplied
`mu`/`lambda_`. `"mu_comma_lambda"` requires `lambda_ >= mu`.
`selection_method` / `crossover_method` / `mutation_method` forward
into `create_toolbox` and accept the operator keys documented above.

**Budget support:** `ENV_STEPS`, `EVAL_EPISODES`.
**Capabilities:** `discrete_action=True, continuous_action=True,
needs_global_state=False, on_policy=False`.

**Budget translation:** `num_generations =
budget_value // (population_size · max_steps_per_episode)` for
`ENV_STEPS`; `num_generations = budget_value // population_size` for
`EVAL_EPISODES`; minimum one generation. `max_steps_per_episode` is
read from the first-class `config.max_steps_per_episode` (default
`200` when unset).

**Opt-in real-tick budgeting (`budget_on_consumed`):** set
`TrainingConfig.budget_on_consumed=True` under `BudgetMode.ENV_STEPS` to
run until the *real* consumed env-step count reaches `budget_value` instead
of stopping after the pre-computed nominal generation count. Supported by
EA `simple`, CMA-ES, PSO, co-evolution (shared pool),
NEAT/HyperNEAT/ES-HyperNEAT (`n=1` population stepping), and the QDax
trainers on the numpy/PettingZoo scoring path; rejected by EA
`mu_plus_lambda` / `mu_comma_lambda` and the QDax JAX-native scoring path.
The realized
`num_generations` and `env_steps_consumed` are then determined at runtime.
Result metrics include `env_steps_nominal` (the pre-computed ceiling) and
`env_steps_consumed` (real ticks, always ≤ `env_steps_nominal` in nominal
mode, ≥ `budget_value` after a consumed-budget run).

### CMA-ES trainer — `baselines.cma_trainer`

Drives an (μ/μ_w, λ)-CMA-ES search over the same `FixedTopologyGenome`
flat weight vector as `EATrainer`, via `deap.cma.Strategy`. Registered
as `"cma-es"` in the trainer registry. Reuses
`baselines.ea_trainer.GenomePolicyWrapper` to expose the best-found
genome as a `PolicyProtocol`.

```python
class CMAESTrainer(BaselineTrainer):
    def __init__(
        self,
        hidden_size: int = 64,
        population_size: int = 50,
        sigma0: float = 0.3,
        activation: str = "tanh",
        weight_range: tuple = (-1.0, 1.0),
        variant: str = "mu_mu_lambda",   # or "one_plus_lambda"
        num_workers: int = 1,            # parallel evaluation
        parallel_backend: str = "process",
        scheduler_address: Optional[str] = None,  # remote Dask scheduler
    ) -> None: ...
```

`variant="mu_mu_lambda"` uses `deap.cma.Strategy` (default rank-μ
update). `variant="one_plus_lambda"` uses
`deap.cma.StrategyOnePlusLambda` — the elitist single-parent variant,
wired through the
`integration.cma_integration.setup_cma_one_plus_lambda_strategy`
helper.

**Budget support:** `ENV_STEPS`, `EVAL_EPISODES` — raises `ValueError`
from `_translate_budget` on any other mode.
**Capabilities:** `discrete_action=True, continuous_action=True,
needs_global_state=False, on_policy=False`.

**Budget translation:** identical to `EATrainer` so equivalent
`TrainingConfig`s yield the same `num_generations` for both trainers.

**Seeding:** reads the first-class `config.seed` and passes it to
`numpy.random.seed` prior to strategy construction. `deap.cma.Strategy`
samples exclusively via `numpy.random.standard_normal`, so this is the
only RNG knob.

**Result metrics:** `fitness_history` (per-generation
`{generation, mean, std, min, max}`), `final_fitness`,
`num_generations`, `population_size`, `dimension`.

### `class PSOTrainer(BaselineTrainer)`

Particle Swarm Optimisation trainer. Shares the `FixedTopologyGenome`
representation with `EATrainer` and `CMAESTrainer`.

```python
class PSOTrainer(BaselineTrainer):
    def __init__(
        self,
        hidden_size: int = 64,
        population_size: int = 50,
        phi1: float = 2.0,           # cognitive coefficient
        phi2: float = 2.0,           # social coefficient
        weight: float = 0.7298,      # inertia weight
        smin: Optional[float] = None,  # min velocity
        smax: Optional[float] = None,  # max velocity
        activation: str = "tanh",
        weight_range: tuple = (-1.0, 1.0),
        num_workers: int = 1,
        parallel_backend: str = "process",
        scheduler_address: Optional[str] = None,  # remote Dask scheduler
    ) -> None: ...
```

**Budget support:** `ENV_STEPS`, `EVAL_EPISODES`.
**Capabilities:** `discrete_action=True, continuous_action=True,
needs_global_state=False, on_policy=False, supports_turn_based=True`.

### Parallel evaluation — `utils.parallel`

`EATrainer`, `CMAESTrainer`, and `PSOTrainer` support **per-individual
parallel fitness evaluation** via a backend-agnostic executor. The worker
count, backend, and (for Dask) a remote scheduler are **first-class
`TrainingConfig` fields** (`num_workers` / `parallel_backend` /
`scheduler_address`), so they are selectable per run:

```python
from deap_pettingzoo.baselines.base import BudgetMode, TrainingConfig
from deap_pettingzoo.baselines.ea_trainer import EATrainer

config = TrainingConfig(
    env_name="simple_spread_v3",
    budget_mode=BudgetMode.ENV_STEPS,
    budget_value=50_000,
    num_workers=8,            # per-run worker count
    parallel_backend="process",  # "process" | "dask"
)
trainer = EATrainer(hidden_size=32, population_size=64)
result = trainer.train(config)
print(result.metadata["num_workers"], result.metadata["parallel_backend"])
```

The trainer constructors also accept `num_workers` / `parallel_backend` /
`scheduler_address`; those act as the **fallback** used only when the
corresponding `TrainingConfig` field is left at its default (`1` /
`"process"` / `None`). There is no
`config.algorithm_params["num_workers"|"parallel_backend"]` override — set
the `TrainingConfig` fields instead. Co-evolution reads the same fields from
the `Parallelism` settings into `CoEvolutionConfig`.

Supported backends (selected via `parallel_backend=`):

| Backend     | Requires extra | Use it when… |
| ----------- | -------------- | ------------ |
| `"process"` | none (stdlib)  | Single host, CPU-bound rollouts. The default — `concurrent.futures.ProcessPoolExecutor`, one fresh process per worker. |
| `"dask"`    | `[dask]`       | Single-machine (auto-built local cluster, 1 thread/worker) **or** a **distributed / remote** cluster via `scheduler_address` (or `DEAP_DASK_SCHEDULER`) — see below. |

**Custom environments.** Envs registered at runtime via `register_env(...)`
are propagated to workers on **every** backend: the registry snapshot is
shipped on each picklable worker spec and replayed lazily in the worker
(`replay_registry_snapshot_once`), so Dask workers — which have no
per-worker `initializer` hook — re-register them before the env is built. The
custom env's factory must be a **module-level / picklable** callable (lambdas
and closures will not pickle under `spawn`). `get_registry_snapshot()` emits a
`RuntimeWarning` naming any env it has to drop because its factory is a
non-module unpicklable (lambda/closure/local class), so a custom env silently
missing under `num_workers>1` surfaces at snapshot time rather than as a
cryptic worker-side "unregistered env".

**Determinism.** When `config.seed` is set, the trainer
allocates **per-individual evaluation seeds** (`base_seed + counter`)
so that the fitness array on the parallel path is bit-identical to
the serial path for the same population. Differences in the final
champion across `num_workers` settings come exclusively from DEAP's
selection / mutation / crossover (which use Python's unseeded
`random` module), not from evaluation noise.

**Worker semantics.** Each worker reconstructs the
`PettingZooWrapper` and `SingleEpisodeEvaluator` once on first call
via a module-global cache (see
`deap_pettingzoo.integration.deap_integration._WORKER_CACHE`). Live
`Env` / `Evaluator` objects are never pickled across the boundary; we
ship a small `_GenomePayload` dataclass (weights + topology) plus a
single `_EvalWorkerSpec` (env name, env config, max steps,
aggregation, optional team mapping, and a registry snapshot for custom
envs). This makes the wire format robust under both `spawn` (Windows)
and `fork` (Linux/macOS) start methods.

The worker caches assume **one task at a time per process**. The process
backend satisfies this naturally; the auto-built Dask client is
therefore pinned to one thread per worker (process-parallel only). Inject
your own `Client` if you need a different layout.

**`num_workers` normalisation.**
`deap_pettingzoo.utils.parallel.resolve_num_workers` collapses
`None` / `0` / `1` to `1` (serial), caps at `os.cpu_count()` with a
`UserWarning`, and rejects negative values with `ValueError`.

**Trainer metadata.** `EATrainer`, `CMAESTrainer`, and `PSOTrainer` populate
`result.metadata["num_workers"]`, `result.metadata["parallel_backend"]`, and
`result.metadata["scheduler_address"]` with the **effective** values used
(after normalisation). On the Dask backend, `result.metadata["dask_workers"]`
additionally lists the observed worker hosts (`None` on the `process`
backend or when no Dask cluster was built).

#### Dask backend

Install with `pip install 'deap-pettingzoo[dask]'`. A local process
cluster (one thread per worker) is built automatically when
`num_workers > 1` and no remote target is given. To reach a **remote**
`dask.distributed` scheduler, set `scheduler_address` — a first-class
`TrainingConfig` field (and matching trainer-constructor kwarg) — instead of
hand-building a `Client`:

```python
config = TrainingConfig(
    ..., parallel_backend="dask", num_workers=8,
    scheduler_address="tcp://scheduler:8786",   # or a scheduler-file path
)
result = EATrainer(...).train(config)
print(result.metadata["scheduler_address"], result.metadata["dask_workers"])
```

`make_executor` resolves the Dask connection with this precedence: an
injected `dask_client` (power users — lifetime stays theirs) > the
`scheduler_address` argument > the `DEAP_DASK_SCHEDULER` environment
variable (the Slurm ergonomic — an sbatch script exports it and an
unmodified driver script picks up the cluster automatically) > an
auto-built local `LocalCluster` (unchanged default). A value containing
`://` connects directly (`Client(address)`); otherwise it's treated as a
scheduler-file path (`Client(scheduler_file=...)`, the canonical
dask-on-Slurm pattern over a shared filesystem). When a remote
client/address/env var is in play, the `num_workers<=1` serial fast-path is
skipped and `num_workers` is not capped to the driver's `os.cpu_count()` —
an explicit remote target is never silently downgraded or shrunk.
Locally-owned clients are closed in `__exit__`; injected clients are not.

Power users can still bypass `scheduler_address` and pass a pre-built
`Client` directly:

```python
from dask.distributed import Client
from deap_pettingzoo.utils.parallel import make_executor
client = Client("tcp://scheduler:8786")
with make_executor(8, backend="dask", dask_client=client) as ex:
    ...
```

#### Payload pickle guarantee

Dask uses strict `pickle` (no guaranteed `cloudpickle`
fallback). All worker payloads in this project — `_GenomePayload`,
`_EvalWorkerSpec`, `_RolloutSpec`, `_NeatWorkerSpec`, the co-evolution
payloads, and caller-supplied descriptor classes (e.g. those in
`examples/qdax/descriptors.py`) — are top-level classes or plain
dataclasses and round-trip cleanly under `pickle`, including the
registry snapshot each carries for custom-env propagation.
Regression tests guard against reintroduction of unpicklable closures
and verify cross-process custom-env support.

### AgileRL trainers — `baselines.agilerl_trainer`

Abstract orchestration class. Imports the AgileRL algorithm **lazily**
inside `train()` so `deap_pettingzoo.baselines` can be imported without
the `[rl-baselines]` extra installed.

```python
class AgileRLTrainer(BaselineTrainer):
    ALGORITHM_NAME: str     = "agilerl"
    ALGORITHM_IMPORT: Tuple[str, str] = ("agilerl", "")
    ON_POLICY: bool         = False

    DEFAULT_PARAMS = {
        "batch_size":      64,
        "memory_size":     10_000,
        "learning_delay":  500,
        "rollout_steps":   256,
        "device":          "cpu",
    }

    def __init__(self, device: str = "cpu", verbose: bool = False) -> None: ...
```

**Budget support:** `ENV_STEPS`, `EVAL_EPISODES`.

#### Concrete subclasses

| Class | `ALGORITHM_NAME` | Import | `ON_POLICY` |
| --- | --- | --- | --- |
| `MADDPGTrainer` | `"maddpg"` | `agilerl.algorithms.maddpg.MADDPG` | `False` |
| `MATD3Trainer`  | `"matd3"`  | `agilerl.algorithms.matd3.MATD3`   | `False` |
| `IPPOTrainer`   | `"ippo"`   | `agilerl.algorithms.ippo.IPPO`     | `True`  |
| `GroupedIPPOTrainer` | `"ippo-grouped"` | inherited from `IPPOTrainer` | `True` |

`GroupedIPPOTrainer` uses AgileRL's prefix-based agent-grouping
(`agent_ids` split on the final `_`) so that agents with shared
observation/action spaces share a single actor/critic. It **fails
loudly** (`ValueError`) when (a) every agent ends up as a singleton
group — user probably wanted plain `IPPOTrainer` — or (b) agents in the
same group have mismatched observation or action spaces.

#### Vectorized rollouts

AgileRL trainers accept an optional `algorithm_params["num_envs"]`
(default `1`) that controls how many PettingZoo parallel environments
run in lock-step during rollout collection. When `num_envs > 1` the
trainer builds an
`agilerl.vector.pz_async_vec_env.AsyncPettingZooVecEnv` (one
sub-process per sub-env) via
`baselines.agilerl_adapter.build_vec_parallel_env`, and
the training loop switches to a vectorized branch that:

> **Subprocess registry isolation.**
> `AsyncPettingZooVecEnv` spawns fresh worker processes that do not
> inherit the main-process `_REGISTRY`.  `build_vec_parallel_env`
> therefore resolves the `EnvSpec` **once in the main process** via
> `get_env_spec(env_name)`, then each worker factory calls
> `build_env_from_spec(spec, ...)` directly — no registry lookup
> needed in the worker.  Any registered env whose factory is a
> picklable top-level callable (the requirement for Windows `spawn`
> workers) works transparently, including custom aquarium / scripted
> opponent envs registered via `register_scripted_aquarium`.

On-policy trainers (`IPPOTrainer`, `GroupedIPPOTrainer`) use the vec
env to collect batched rollout chunks before each `learn()` call. In
that path the trainer:

* passes the agent's `(num_envs, *)` actions straight through
    `env.step` (no per-step batch-dim squeezing);
* relies on the vec env's worker-side auto-reset on episode
    termination (no explicit `env.reset()` call);
* tracks one episode-reward accumulator per sub-env, emitting an
    entry to `metrics["episode_rewards"]` whenever any sub-env
    terminates.

Off-policy trainers (`MADDPGTrainer`, `MATD3Trainer`) use the same vec
env plumbing, but save batched transitions through AgileRL's
vectorized replay-buffer path. In that branch the trainer:

* preserves the leading `num_envs` axis on observations, raw actions,
    rewards, next observations, and done arrays when calling
    `MultiAgentReplayBuffer.save_to_memory(..., is_vectorised=True)`;
* interprets `learning_delay` and `learn_step` in total env-step
    units, so a single vec step may trigger multiple `learn()` calls if
    it crosses multiple thresholds;
* uses the same per-sub-env all-agent completion rule as the on-policy
    path when emitting completed-episode rewards.

Set `num_envs` to a power of two between `4` and `32` when rollout
cost dominates update cost. Larger values increase parallel sample
collection throughput but also increase replay growth rate and may
change the effective cadence of logging / recording.

Capability flags:

| Trainer | `discrete_action` | `continuous_action` | `needs_global_state` | `on_policy` |
| --- | --- | --- | --- | --- |
| `MADDPGTrainer` | `True` | `True` | `False` | `False` |
| `MATD3Trainer`  | `True` | `True` | `False` | `False` |
| `IPPOTrainer`   | `True` | `True` | `False` | `True`  |

All three trainers above support `algorithm_params["num_envs"] > 1`.
`GroupedIPPOTrainer` follows the same vectorized rollout contract as
`IPPOTrainer`.

### Policy wrapper — `baselines.agilerl_policy`

```python
class AgileRLPolicyWrapper:  # PolicyProtocol
    def __init__(
        self,
        agent: Any,                          # trained AgileRL algorithm
        agent_ids: Iterable[str],            # must be non-empty
        algorithm_name: Optional[str] = None,
    ) -> None: ...

    def get_action(self, observation: np.ndarray, **_: Any) -> Any: ...
    def reset(self) -> None: ...
```

**Behaviour:** maintains a rotation counter through `agent_ids` that
resets at each episode (relies on the caller iterating
`env.possible_agents` in a stable order). Caches per-agent
observations, constructs the full `{agent_id: obs}` dict AgileRL needs
(zero-filled placeholders for unseen agents), invokes the underlying
`get_action(obs_dict, training=False)` (falls back to no-kwargs on
`TypeError` for on-policy variants), then strips leading size-1
vectorisation dims and returns a scalar for 0-d arrays.

### Adapter helpers — `baselines.agilerl_adapter`

```python
SUPPORTED_PARALLEL_ENVS = {
    "simple_tag_v3":       mpe2.simple_tag_v3,
    "simple_adversary_v3": mpe2.simple_adversary_v3,
    "simple_spread_v3":    mpe2.simple_spread_v3,
}

def build_parallel_env(
    env_name: str,
    env_config: Optional[Dict[str, Any]] = None,
    render_mode: Optional[str] = None,
): ...

def extract_space_metadata(
    env,
) -> Tuple[List[gym.Space], List[gym.Space], List[str]]: ...

def translate_budget_to_env_steps(
    config: TrainingConfig,
    default_episode_len: int = 25,
) -> int: ...
```

Raises `ValueError` on unsupported env names or a `WALL_CLOCK` budget
(trainer must short-circuit on a wall-clock timer instead of converting
to steps).

### Unified evaluator — `baselines.unified_evaluator`

```python
class UnifiedEvaluator(BaselineEvaluator):
    def __init__(self, max_steps: int = 500) -> None: ...

    def evaluate(
        self,
        policy: PolicyProtocol,
        env_name: str,
        num_episodes: int,
        seed: Optional[int] = None,
        render: bool = False,
    ) -> EvalResult: ...

    def close(self) -> None: ...     # closes cached envs
```

Caches one `PettingZooWrapper` per env name. Per-episode seeds are
drawn from a `numpy.random.Generator` seeded by `seed` (all-`None` when
`seed is None`). Returns `EvalResult` with reward sums across **all**
agents per episode.

### Comparison orchestration — `baselines.comparison`

```python
def register_trainer(name: str, trainer_class: type) -> None: ...
def get_trainer(name: str, **kwargs) -> BaselineTrainer: ...

def compare_baselines(
    config: ComparisonConfig,
    validate: bool = True,
) -> Dict[str, Any]: ...
```

On import, the registry is auto-populated with `"ea"`, `"cma-es"`,
`"maddpg"`, `"matd3"`, `"ippo"`, `"ippo-grouped"`. `compare_baselines` trains
each algorithm for `num_seeds` seeds under the same `TrainingConfig`,
evaluates every trained policy with a `UnifiedEvaluator`, aggregates,
and optionally dumps JSON to `config.output_dir`.

Return dict shape:

```python
{
    "config": { "env_name", "budget_mode", "budget_value", "num_seeds", "eval_episodes" },
    "<algo>": {
        "seeds": [ { "seed", "training_time", "training_metrics", "eval_result" }, … ],
        "mean_reward", "std_reward", "min_reward", "max_reward",
        "best_seed", "capabilities",
    },
    # or { "<algo>": { "skipped": True, "issues": [...] } } when incompatible.
}
```

### Compatibility checks — `baselines.compatibility`

```python
@dataclass
class EnvironmentInfo:
    env_name: str
    num_agents: int
    agent_ids: List[str]
    has_continuous_actions: bool
    has_discrete_actions: bool
    has_global_state: bool
    obs_dim: Optional[int]        # None if heterogeneous
    act_dim: Optional[int]        # None if heterogeneous
    action_low:  Optional[np.ndarray]
    action_high: Optional[np.ndarray]
    extra: Dict[str, Any] = {}

    def to_dict(self) -> Dict[str, Any]: ...
```

```python
@dataclass
class CompatibilityResult:
    compatible: bool
    algorithm:  str
    env_name:   str
    issues:   List[str] = []   # fatal
    warnings: List[str] = []   # non-fatal

    def raise_if_incompatible(self) -> None: ...   # raises ValueError
    def to_dict(self) -> Dict[str, Any]: ...
```

```python
def inspect_environment(env_name: str) -> EnvironmentInfo: ...

def check_compatibility(
    trainer: BaselineTrainer,
    env_info: EnvironmentInfo,
    algorithm_name: str = "unknown",
) -> CompatibilityResult: ...

def validate_compatibility(
    trainer: BaselineTrainer,
    env_name: str,
    algorithm_name: str = "unknown",
) -> CompatibilityResult: ...

def build_compatibility_matrix(
    algorithms: List[str],
    env_names:  List[str],
) -> Dict[str, Dict[str, CompatibilityResult]]: ...

def get_compatible_algorithms(
    env_name: str,
    registered_only: bool = True,
) -> List[str]: ...

def format_compatibility_matrix(
    matrix: Dict[str, Dict[str, CompatibilityResult]],
    show_warnings: bool = False,
) -> str: ...
```

**Compatibility rules used by `check_compatibility`:**

Fatal (go into `issues`):

- Env has continuous (Box) actions but `caps["continuous_action"]` is
  `False`.
- Env has discrete actions but `caps["discrete_action"]` is `False`.
- `caps["needs_global_state"]` is `True` but env does not expose a
  `state()` method.

Non-fatal (go into `warnings`):

- `env_info.num_agents > 8` — scalability notice.
- `env_info.obs_dim is None` — heterogeneous observation spaces.
- `env_info.act_dim is None` — heterogeneous action spaces.

`get_compatible_algorithms(..., registered_only=False)` raises
`NotImplementedError` (registry-only discovery for now).

---

## Co-evolution (`baselines.coevolution`)

Multi-population / multi-side evolution is implemented as a single
configurable trainer. The package is intentionally source-agnostic on
the opponent-sampling side so future back-ends (e.g. QDax repertoires)
plug in by satisfying the `OpponentPool` Protocol without touching
trainer code.

Public surface (re-exported from `deap_pettingzoo.baselines`):

* `CoEvolutionTrainer`, `CoEvolutionConfig`
* `Side`, `SideState`, `EASideState`, `CMAESSideState`, `FrozenSideState`,
  `RLSideState` (`SUPPORTED_RL_ALGORITHMS`)
* `make_side_state(side)`, `register_side_state(token, loader_or_class,
  *, overwrite=False)`
* `OpponentPool` Protocol
* Concrete pools: `PopulationPool`, `ArchivePool`, `FrozenPolicyPool`,
  `CompositePool`, `RLPolicyPool`
* Archives: `HallOfFame` (ABC), `RingHallOfFame`, `ElitistHallOfFame`
* Samplers: `uniform_sampler`, `latest_sampler`,
  `fitness_proportional_sampler`, `tournament_sampler(k)`

### Design contract

```
OpponentPool.sample_opponent(rng) -> PolicyProtocol
OpponentPool.sample_opponents(rng, k) -> List[PolicyProtocol]
OpponentPool.update(population_like) -> None
len(pool) -> int
```

`sample_opponent` returns **one policy per side per episode**; the
policy is then shared across every env agent belonging to that side.
This mirrors canonical self-play ("one champion drives the team"),
dramatically reduces opponent-sampling variance in evaluation, and
works naturally with stateless `FeedforwardNeuralPolicy`. For
heterogeneous-within-side action spaces, use `FrozenPolicyPool` with
per-agent policy dispatchers.

### `@dataclass CoEvolutionConfig`

```python
CoEvolutionConfig(
    mode: str,                      # 'competitive_2pop' | 'self_play_hof' | 'cooperative'
    sides: List[Side],
    schedule: str = 'synchronous',  # or 'alternating'
    alternating_generations: int = 1,
    k_opponents: int = 3,
    aggregation: str = 'mean',      # 'mean' | 'min' | 'max' | 'sum'
    snapshot_every: int = 1,
    reporting_side: Optional[str] = None,
    max_steps_per_episode: int = 200,
    seed: Optional[int] = None,
    num_workers: int = 1,                 # episode-loop parallelism
    parallel_backend: str = 'process',    # 'process' | 'dask'
    scheduler_address: Optional[str] = None,  # remote Dask scheduler
)
```

Validation (`__post_init__`): disjoint side names, mode-specific
constraints (`competitive_2pop` / `cooperative` require ≥ 2 sides;
`self_play_hof` requires ≥ 2 sides **each with its own** `HallOfFame`
and forces `use_archive_in_opponent_pool=True` on every side).

### `@dataclass Side`

```python
Side(
    name: str,
    role_prefixes: Tuple[str, ...],       # env-agent prefixes (disjoint across sides)
    optimizer: str = 'ea',                # 'ea' | 'cma' | 'frozen' | 'rl' | 'qdax_map_elites' | 'qdax_cma_me'
    optimizer_config: Dict[str, Any] = {},
    hof: Optional[HallOfFame] = None,
    use_archive_in_opponent_pool: bool = False,
    archive_mix: float = 0.5,             # weight on HoF in CompositePool
    opponent_sampler: Sampler = uniform_sampler,
    archive_sampler: Sampler = uniform_sampler,
    frozen_policies: Optional[Sequence[PolicyProtocol]] = None,
)
```

Accepted `optimizer_config` keys per optimizer:

* **ea** — `hidden_size`, `population_size`, `crossover_rate`,
  `mutation_rate`, `selection_method` / `crossover_method` /
  `mutation_method` (same dispatch as `EATrainer`),
  `tournament_size`, `crossover_alpha`, `mutation_sigma`,
  `mutation_indpb`, `weight_range`, `activation`, `masked`, `seed`.
* **cma** — `hidden_size`, `population_size`, `sigma0`, `activation`,
  `weight_range`, `variant` (`'mu_mu_lambda'` | `'one_plus_lambda'`),
  `masked`, `seed`.
* **frozen** — none. Requires `frozen_policies` to be non-empty.
* **rl** — `algorithm` (**required**, `'maddpg'` | `'matd3'`;
  `'ippo'` / `'grouped_ippo'` are accepted by name but
  raise `NotImplementedError` from `initialise()`),
  `rollout_episodes_per_gen` (default `4`),
  `learning_steps_per_gen` (default `1`),
  `batch_size` (default `64`), `memory_size` (default `10_000`),
  `learning_delay` (default `0`), `device` (default: `'cuda'` if
  `torch.cuda.is_available()` else `'cpu'`),
  `agent_ctor_kwargs` (forwarded verbatim to the AgileRL algorithm
  constructor), `seed`.
* **qdax_map_elites** / **qdax_cma_me** — see
  `deap_pettingzoo.contrib.qdax.coevolution`.

### Bidirectional EA-RL co-evolution

`Side(optimizer="rl", ...)` plugs an AgileRL multi-agent algorithm
into the same generation loop as EA / CMA / QDax sides. The RL
side owns one algorithm object (no "population"), trains its
networks against opponents drawn from the *current* state of the
other sides every generation, and exposes its current weights back
to those sides as a degenerate size-1 `RLPolicyPool`. Supported
pairings: EA×EA, EA×RL, RL×RL, plus any combination with frozen,
CMA, or QDax sides.

```python
class RLSideState(SideState):
    """Used implicitly when Side.optimizer == 'rl'.

    initialise() lazy-imports AgileRL (raises a friendly ImportError
    pointing at the 'rl-baselines' extra if missing), constructs one
    algorithm object covering all side-controlled env agents, and
    builds a MultiAgentReplayBuffer (off-policy only). step_generation
    runs `rollout_episodes_per_gen` rollouts (sampling one opponent
    per side per episode), pushes transitions, calls
    `agent.learn(experiences)` `learning_steps_per_gen` times, and
    refreshes its `RLPolicyPool` so other sides next gen face the
    new policy.
    """
    algorithm: str            # 'maddpg' | 'matd3'
    is_on_policy: bool
    resolved_device: str      # 'cuda' / 'cpu' after initialise()
    def current_pool(self) -> RLPolicyPool: ...
    def best_policy(self, env) -> AgileRLPolicyWrapper: ...
```

```python
class RLPolicyPool(OpponentPool):
    """Degenerate size-1 pool wrapping the RL side's current
    AgileRLPolicyWrapper. Refreshed once per generation by the
    owning RLSideState."""
    def update(self, policy: PolicyProtocol) -> None: ...
    def sample_opponent(self, rng) -> PolicyProtocol: ...   # cached
    def snapshot_sync(self) -> RLPolicyPool: ...            # deep-copy for sync schedule
    def snapshot_opponents(self, rng, k):                   # raises
        raise NotImplementedError(...)
```

**Caveats:**

* `k_opponents` is auto-treated as 1 for RL rollouts (the RL side
  samples one opponent per rollout episode internally).
* Parallel co-evolution (`CoEvolutionConfig.num_workers > 1` or a
  remote Dask scheduler) works with RL sides: EA/CMA generations
  dispatch population-wide with the live RL opponent shipped as a
  per-generation `FrozenRLPolicyWrapper` CPU actor snapshot, and
  off-policy RL rollout collection uses the same executor. RL
  *learning* stays on the driver, and workers must have torch +
  AgileRL installed to unpickle the snapshots. An informational
  `RuntimeWarning` notes the arrangement.
* On-policy algorithms (IPPO / grouped IPPO) are supported, but
  collect their rollouts serially on the driver — the live
  actor+critic must produce the log_prob/value that an actor-only
  snapshot cannot.
* GPU contention: when an RL side resolves a CUDA device *and* JAX
  has GPU access (typical sign of a co-located QDax side), a
  `RuntimeWarning` is emitted from
  `deap_pettingzoo.utils.device_check.warn_on_gpu_contention` so
  the user can pick which library gets the GPU.

### `class CoEvolutionTrainer(BaselineTrainer)`

Registered in the baseline registry as `'coevolution'`. Consumes the typed
`TrainingConfig.coevolution_config` entry point (must be a
`CoEvolutionConfig` instance — a `TypeError` at `TrainingConfig` construction
time for any other type, or a `ValueError` from `train()` if left unset).

* `supports_budget_mode`: `ENV_STEPS`, `EVAL_EPISODES`.
* `get_capabilities()`: `discrete_action=True`, `continuous_action=True`,
  `supports_turn_based=True`.

`TrainingResult.policy` is the best-ever individual of
`reporting_side` (defaults to `sides[0].name`) wrapped as a
`FeedforwardNeuralPolicy`.

`TrainingResult.metrics`:

* `per_side_history[side_name]` — list of per-gen dicts
  (`generation`, `mean`, `std`, `min`, `max`, `best_fitness`).
* `final_best[side_name]` — best fitness per side; `None` for frozen
  sides.
* `hof_sizes[side_name]` — final archive length (0 if no HoF).
* `num_generations` — total generations (or side-steps for alternating)
  actually run; equals nominal derivation under `budget_on_consumed=False`,
  may differ when consumed-budget stop triggers early.
* `env_steps_nominal` — nominal ceiling: `ep_per_gen × max_steps × num_generations`.
* `env_steps_consumed` — real env ticks summed across all sides and
  all generations; ≤ `env_steps_nominal` in nominal mode (early episode
  truncation), ≥ `budget_value` after a consumed-budget run.

**Consumed-step budgeting (`budget_on_consumed=True`):** with
`BudgetMode.ENV_STEPS`, the trainer sums the real per-side
`GenerationStats.env_steps` after each completed generation (shared pool
across all sides) and stops once the aggregate reaches `budget_value`.
All side types report real ticks: EA/CMA/PSO via their evaluator counters;
RL and RL-HPO sides via per-step counters; NEAT/HyperNEAT/ES-HyperNEAT
via their fitness-loop `_steps` accumulator; frozen sides contribute 0.
Overshoot is bounded by one generation's pooled cost.

### `OpponentPool` Protocol

```python
@runtime_checkable
class OpponentPool(Protocol):
    def sample_opponent(self, rng: np.random.Generator) -> PolicyProtocol: ...
    def sample_opponents(self, rng, k: int) -> List[PolicyProtocol]: ...
    def snapshot_opponents(
        self, rng: np.random.Generator, k: int,
    ) -> List["OpponentSnapshot"]: ...   # picklable rebuild contract
    def update(self, population_like: Any) -> None: ...
    def __len__(self) -> int: ...
```

Any object satisfying the protocol can be plugged into the trainer
via `Side.optimizer='frozen'` (with `frozen_policies`) or a custom
`SideState` subclass. This is the extension point for QDax
repertoires, PSRO-style meta-solvers, or cross-framework opponent
sources.

### Parallel co-evolution

`CoEvolutionConfig.num_workers > 1` parallelises the `k_opponents`
episode loop **inside** one `CoEvolutionEvaluator.evaluate(...)`
call. The trainer opens **one** executor (`make_executor(num_workers,
backend=parallel_backend)`) for the whole `train()` call and reuses it
across every generation — the pool is not torn down between sides.

**Picklable snapshot contract.** Workers cannot share live torch /
JAX / DEAP state, so each pool implements `snapshot_opponents(rng, k)`
which returns a list of `OpponentSnapshot(kind, payload)` dataclasses.
The driver builds `k` payloads (one per episode) and ships them to
the executor; on the worker, `rebuild_opponent_policy(snap,
action_space=...)` reconstructs the policy locally. Supported kinds:

* `feedforward` — `FixedTopologyGenome` + `masked` flag (the default
  for `PopulationPool` and `ArchivePool`).
* `frozen` — pickle-by-value; works for any picklable
  `PolicyProtocol` (e.g. lambdas-free pure-numpy policies).
* `frozen_registered` — registry-name lookup for non-picklable
  policies (torch / AgileRL / closures). Register policies on every
  process via `register_frozen_policy(name, factory)` at module
  import.
* `qdax_flax` — flat numpy params + `layer_sizes` + `activation`,
  rebuilt into a Flax MLP by `_resolve_qdax_module`. Available via
  `QDaxRepertoirePool` (requires passing `layer_sizes=` to its
  constructor).

**Scope.** Only EA / CMA learner sides receive the executor — QDax
learner sides keep their existing inner numpy-pool lifecycle (use
`optimizer_config['num_workers']` there) and frozen sides never
evaluate. `k_opponents == 1` short-circuits to the serial path
regardless of `num_workers` (a `RuntimeWarning` is emitted in this
case).

**Determinism.** Opponent sampling and per-episode seed derivation
happen on the driver before dispatch, so a fixed
`CoEvolutionConfig.seed` produces identical fitness sequences across
`num_workers ∈ {1, 2, 4, …}`.

**Metadata.** `TrainingResult.metadata` records
`coevolution_num_workers` (the resolved worker count) and
`coevolution_parallel_backend`.

### Hall-of-Fame archives

```python
class HallOfFame(ABC):
    def add(snapshot: Any, fitness: float) -> None: ...
    @property items: List[Any]
    @property fitnesses: np.ndarray
    @property is_empty: bool
    def __len__() -> int
    def extend(snapshots, fitnesses) -> None
    def clear() -> None

class RingHallOfFame(HallOfFame):   # FIFO
    def __init__(capacity: int)

class ElitistHallOfFame(HallOfFame):  # top-k by fitness, ties do not displace
    def __init__(capacity: int)
```

### Samplers

```python
Sampler = Callable[[int, Optional[np.ndarray], np.random.Generator], int]

uniform_sampler
latest_sampler                          # always returns size - 1
fitness_proportional_sampler            # requires fitnesses; shifts negatives
tournament_sampler(k: int = 3)          # factory; caps k at pool size
```

---

## Logging & Analysis

Full reference for the experiment-logging foundation (`deap_pettingzoo.utils.logging`) and the offline analysis library (`deap_pettingzoo.analysis`) lives in [logging_and_analysis.md](logging_and_analysis.md). Highlights:

- `ExperimentLogger.from_config(config, *, algorithm)` writes `manifest.json` + `metrics.csv` + `summary.json` (and optional `tb/` / Parquet / replays) under `<output_dir>/<run_id>/`. Returns a `NoOpLogger` when `config.output_dir is None` so the backend has no functional dependency on disk artefacts.
- `deap_pettingzoo.analysis.load_experiment(root)` scans for manifests and returns an `Experiment` of `RunRecord`s. Analysis surface: `iqm`, `bootstrap_ci`, `learning_curve_with_ci`, `mann_whitney` / `welch_t` / `cohens_d` / `cliffs_delta`, `pairwise_table`, plus an `rliable` bridge and matplotlib plots / `build_report`. Multi-environment experiments add `collect_score_matrices` / `final_score_table`, `NormalizationSpec` (with `from_dict` / `from_minmax` / `from_reference_algorithm` factories), `normalize_score_matrices`, and `aggregate_metrics_forest`; `build_report` and the Streamlit Compare page auto-detect ≥ 2 envs and emit a "Multi-task aggregate" section.
- The Streamlit UI consumes exactly these public functions.

---

## Type & protocol reference

### `PolicyProtocol`

Structural contract (`typing.Protocol`). Minimum:

```python
def get_action(self, observation: np.ndarray) -> Any: ...
def reset(self) -> None: ...
```

Built-in implementations: `FeedforwardNeuralPolicy`, `APFEvaderPolicy`,
`AgileRLPolicyWrapper`, `GenomePolicyWrapper`.

### DEAP `Individual`

Created by `setup_deap_creators()`: a plain Python object whose
`.fitness` is a `creator.FitnessMax` (weights `(1.0,)`) and whose
`.genome` slot holds a `FixedTopologyGenome` (attached by
`create_deap_individual` or by user code). Evaluators require this
`.genome` attribute.

### Supported environments

Registered in both `PettingZooWrapper.SUPPORTED_ENVS` and
`SUPPORTED_PARALLEL_ENVS`:

- `simple_tag_v3` (MPE2)
- `simple_adversary_v3` (MPE2)
- `simple_spread_v3` (MPE2)

All other environment names raise `ValueError` at wrapper or adapter
construction time.

---

## Error handling conventions

No bespoke exception hierarchy is defined. The codebase uses
standard-library errors only:

| Exception | Raised by |
| --- | --- |
| `ValueError` | Invalid config (`layer_sizes`, `activation`, aggregation name, `env_name`, `budget_value`, missing scripted policies, empty algorithm list, …), action-not-in-space, shape mismatches. |
| `TypeError` | `FeedforwardNeuralPolicy(genome)` called with a non-`FixedTopologyGenome`. |
| `RuntimeError` | Wraps arbitrary rollout failures inside `SingleEpisodeEvaluator.evaluate`; `PursuitEvasionEnv._get_world` when the MPE world cannot be located. |
| `ImportError` | `AgileRLTrainer.train` when the `[rl-baselines]` extra is missing. |
| `NotImplementedError` | `get_compatible_algorithms(..., registered_only=False)`. |
| `UserWarning` | `PursuitEvasionEnv.__init__` when `curriculum=True` and `equal_ability=True`. |

`CompatibilityResult.raise_if_incompatible()` raises `ValueError` with
a multi-line message listing every collected issue — the only bespoke
error-formatting helper in the project.
