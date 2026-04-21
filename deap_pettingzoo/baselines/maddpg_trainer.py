"""
MADDPG trainer via cleanMARL subprocess.

Wraps the cleanMARL MADDPG implementation to conform to the unified
BaselineTrainer interface. Launches cleanMARL as an external subprocess,
monitors training progress via stdout parsing, and loads the trained
actor networks for evaluation via the PolicyProtocol interface.

Requirements (optional - only needed when actually training):
    pip install torch
    pip install cleanrl  # or clone https://github.com/vwxyzjn/cleanrl
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from deap_pettingzoo.baselines.base import (
    BaselineTrainer,
    BudgetMode,
    PolicyProtocol,
    TrainingConfig,
    TrainingResult,
)
from deap_pettingzoo.environments import PettingZooWrapper


class MADDPGPolicyWrapper:
    """Wraps a loaded cleanMARL MADDPG actor network as PolicyProtocol.

    Loads actor checkpoint(s) saved by cleanMARL from disk and provides
    a get_action() interface compatible with UnifiedEvaluator. For
    homogeneous environments all agents share the first actor found.

    Supports both full-model saves (torch.save(actor, path)) and
    state-dict saves (torch.save(actor.state_dict(), path)).
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        obs_dim: int,
        act_dim: int,
        action_low: float = -1.0,
        action_high: float = 1.0,
        hidden_dim: int = 256,
        device: str = "cpu",
    ):
        """Load MADDPG actor from checkpoint directory.

        Args:
            checkpoint_dir: Directory containing actor checkpoint file(s).
            obs_dim: Observation space dimensionality.
            act_dim: Action space dimensionality.
            action_low: Lower bound for action rescaling.
            action_high: Upper bound for action rescaling.
            hidden_dim: Hidden layer size (must match saved checkpoint).
            device: PyTorch device string ('cpu', 'cuda').

        Raises:
            ImportError: If PyTorch is not installed.
            FileNotFoundError: If no checkpoint is found in checkpoint_dir.
        """
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required to load MADDPG checkpoints. "
                "Install with: pip install torch"
            ) from exc

        self._torch = torch
        self._device = torch.device(device)
        self._obs_dim = obs_dim
        self._act_dim = act_dim

        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0

        self._actor = self._load_actor(
            checkpoint_dir, obs_dim, act_dim, hidden_dim, action_scale, action_bias
        )
        self._actor.eval()

    # ------------------------------------------------------------------
    # PolicyProtocol implementation
    # ------------------------------------------------------------------

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Run observation through actor network and return action.

        Args:
            observation: Flat observation vector.

        Returns:
            Action vector in the actor's output range.
        """
        import torch

        with torch.no_grad():
            obs_t = torch.tensor(
                observation, dtype=torch.float32, device=self._device
            ).unsqueeze(0)
            action = self._actor(obs_t)
            return action.squeeze(0).cpu().numpy()

    def reset(self) -> None:
        """No-op: MADDPG actor is a stateless feedforward network."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_actor(
        self,
        checkpoint_dir: Path,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int,
        action_scale: float,
        action_bias: float,
    ):
        """Load actor from checkpoint, trying multiple strategies.

        Strategy 1: Load full model saved with torch.save(actor, path).
        Strategy 2: Load state dict into a freshly built matching network.
        """
        import torch

        path = self._find_checkpoint(checkpoint_dir)

        # Try full-model load first (no architecture rebuild needed)
        try:
            actor = torch.load(path, map_location=self._device, weights_only=False)
            if callable(getattr(actor, "forward", None)):
                return actor
        except Exception:
            pass

        # Fall back to state-dict load into a rebuilt network
        state_dict = torch.load(path, map_location=self._device, weights_only=True)
        actor = _build_cleanmarl_actor(
            obs_dim, act_dim, hidden_dim, action_scale, action_bias
        ).to(self._device)
        actor.load_state_dict(state_dict)
        return actor

    def _find_checkpoint(self, checkpoint_dir: Path) -> Path:
        """Locate actor checkpoint file in the directory.

        Searches a prioritised list of common cleanMARL naming patterns.
        """
        candidates = [
            checkpoint_dir / "actor.pt",
            checkpoint_dir / "actor_0.pt",
            checkpoint_dir / "actor.pth",
            checkpoint_dir / "actor_0.pth",
            checkpoint_dir / "model.pt",
        ]
        # Also catch any actor*.pt / actor*.pth globs
        candidates += sorted(checkpoint_dir.glob("actor*.pt"))
        candidates += sorted(checkpoint_dir.glob("actor*.pth"))
        candidates += sorted(checkpoint_dir.glob("*.cleanrl_model"))

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"No actor checkpoint found in '{checkpoint_dir}'. "
            f"Searched: {[str(p) for p in candidates[:5]]}..."
        )


class MADDPGTrainer(BaselineTrainer):
    """MADDPG trainer via cleanMARL subprocess.

    Launches cleanMARL's MADDPG script as a child process using the
    project's Python interpreter, waits for completion (or timeout),
    then loads the saved actor checkpoint for evaluation.

    Args:
        cleanmarl_script: Explicit path to cleanMARL MADDPG script.
            If None, auto-discovers from installed cleanrl package or
            common relative paths. Can also be overridden per-run via
            config.algorithm_params['cleanmarl_script'].
        hidden_dim: Hidden layer width for actor/critic networks.
        learning_rate: Actor and critic learning rate.
        batch_size: Mini-batch size sampled from the replay buffer.
        buffer_size: Total replay buffer capacity (number of transitions).
        tau: Polyak averaging coefficient for target network updates.
        gamma: Discount factor.
        exploration_noise: Std of Gaussian noise added during training.
        device: PyTorch device string passed to cleanMARL ('cpu'/'cuda').
        timeout: Maximum wall-clock seconds before killing the subprocess.
            None means no limit.
    """

    def __init__(
        self,
        cleanmarl_script: Optional[str] = None,
        hidden_dim: int = 256,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        buffer_size: int = int(1e6),
        tau: float = 0.005,
        gamma: float = 0.99,
        exploration_noise: float = 0.1,
        device: str = "cpu",
        timeout: Optional[int] = None,
    ):
        self.cleanmarl_script = cleanmarl_script
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.tau = tau
        self.gamma = gamma
        self.exploration_noise = exploration_noise
        self.device = device
        self.timeout = timeout

    # ------------------------------------------------------------------
    # BaselineTrainer implementation
    # ------------------------------------------------------------------

    def train(self, config: TrainingConfig) -> TrainingResult:
        """Train MADDPG via cleanMARL subprocess.

        1. Inspects the environment to determine obs/action dimensions.
        2. Resolves the cleanMARL script path.
        3. Builds and launches the subprocess command.
        4. Streams stdout to parse training progress.
        5. Loads the saved actor checkpoint and wraps it as PolicyProtocol.

        Args:
            config: Training configuration.

        Returns:
            TrainingResult with actor wrapped as MADDPGPolicyWrapper.

        Raises:
            ImportError: If cleanMARL script cannot be located.
            RuntimeError: If cleanMARL subprocess exits with non-zero code.
        """
        start_time = time.time()

        # Prepare output directory
        run_dir = self._prepare_run_dir(config)

        # Inspect environment for obs/action dimensions
        env = PettingZooWrapper(config.env_name)
        obs = env.reset(seed=0)
        agent_id = env.possible_agents[0]
        obs_dim = len(obs[agent_id])
        action_space = env.action_space(agent_id)
        act_dim = (
            int(action_space.n)
            if hasattr(action_space, "n")
            else int(action_space.shape[0])
        )
        action_low = float(
            getattr(action_space, "low", np.full(act_dim, -1.0)).flat[0]
        )
        action_high = float(
            getattr(action_space, "high", np.full(act_dim, 1.0)).flat[0]
        )
        env.close()

        # Translate budget and resolve script
        total_timesteps = self._translate_budget(config)
        script = self._resolve_script(config)

        # Build command and run
        cmd = self._build_command(script, config, total_timesteps, run_dir)
        training_metrics = self._run_subprocess(cmd)
        training_time = time.time() - start_time

        # Load checkpoint
        policy = MADDPGPolicyWrapper(
            checkpoint_dir=run_dir,
            obs_dim=obs_dim,
            act_dim=act_dim,
            action_low=action_low,
            action_high=action_high,
            hidden_dim=self.hidden_dim,
            device=self.device,
        )

        return TrainingResult(
            algorithm="maddpg",
            policy=policy,
            metrics=training_metrics,
            metadata={
                "run_dir": str(run_dir),
                "total_timesteps": total_timesteps,
                "maddpg_config": {
                    "hidden_dim": self.hidden_dim,
                    "learning_rate": self.learning_rate,
                    "batch_size": self.batch_size,
                    "buffer_size": self.buffer_size,
                    "tau": self.tau,
                    "gamma": self.gamma,
                    "exploration_noise": self.exploration_noise,
                },
            },
            seed=config.algorithm_params.get("seed"),
            training_time=training_time,
        )

    def supports_budget_mode(self, mode: BudgetMode) -> bool:
        """MADDPG supports ENV_STEPS (total_timesteps) and WALL_CLOCK (timeout)."""
        return mode in {BudgetMode.ENV_STEPS, BudgetMode.WALL_CLOCK}

    def get_capabilities(self) -> Dict[str, Any]:
        """Return MADDPG algorithm capabilities."""
        return {
            "discrete_action": False,  # Standard MADDPG targets continuous actions
            "continuous_action": True,
            "needs_global_state": False,  # Decentralised execution
            "on_policy": False,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _translate_budget(self, config: TrainingConfig) -> int:
        """Convert budget config to total_timesteps for cleanMARL.

        Args:
            config: Training configuration.

        Returns:
            Total environment timesteps to pass to cleanMARL.
        """
        if config.budget_mode == BudgetMode.ENV_STEPS:
            return config.budget_value
        if config.budget_mode == BudgetMode.WALL_CLOCK:
            # Use a large default; self.timeout caps wall-clock time
            return config.algorithm_params.get("default_timesteps", 1_000_000)
        raise ValueError(
            f"MADDPG does not support budget mode: {config.budget_mode}"
        )

    def _prepare_run_dir(self, config: TrainingConfig) -> Path:
        """Create and return the output directory for this training run."""
        if config.output_dir is not None:
            run_dir = Path(config.output_dir)
        else:
            seed = config.algorithm_params.get("seed", 0)
            run_dir = Path("runs") / "maddpg" / config.env_name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _resolve_script(self, config: TrainingConfig) -> str:
        """Find the cleanMARL MADDPG script path.

        Priority:
        1. Explicit path set on the trainer instance.
        2. Path provided in config.algorithm_params['cleanmarl_script'].
        3. Auto-discovery from installed cleanrl package.
        4. Common relative paths (for development installs).

        Raises:
            ImportError: If the script cannot be found.
        """
        if self.cleanmarl_script:
            return self.cleanmarl_script

        override = config.algorithm_params.get("cleanmarl_script")
        if override:
            return str(override)

        # Try installed cleanrl package
        try:
            import cleanrl  # type: ignore

            cleanrl_dir = Path(cleanrl.__file__).parent
            for name in ("maddpg_continuous_action.py", "maddpg.py"):
                candidate = cleanrl_dir / name
                if candidate.exists():
                    return str(candidate)
        except ImportError:
            pass

        # Common relative/absolute paths for development setups
        for candidate in (
            Path("cleanrl") / "maddpg_continuous_action.py",
            Path("cleanmarl") / "maddpg.py",
        ):
            if candidate.exists():
                return str(candidate)

        raise ImportError(
            "cleanMARL MADDPG script not found. Options:\n"
            "  1. pip install cleanrl\n"
            "  2. MADDPGTrainer(cleanmarl_script='/path/to/maddpg.py')\n"
            "  3. config.algorithm_params['cleanmarl_script'] = '/path/to/maddpg.py'"
        )

    def _build_command(
        self,
        script: str,
        config: TrainingConfig,
        total_timesteps: int,
        run_dir: Path,
    ) -> List[str]:
        """Assemble the cleanMARL command-line invocation.

        Args:
            script: Path to cleanMARL MADDPG script.
            config: Training configuration.
            total_timesteps: Budget in environment steps.
            run_dir: Directory for model output.

        Returns:
            Argument list suitable for subprocess.Popen.
        """
        seed = config.algorithm_params.get("seed", 1)

        cmd = [
            sys.executable, script,
            "--env-id", config.env_name,
            "--total-timesteps", str(total_timesteps),
            "--seed", str(seed),
            "--save-model",
            "--learning-rate", str(self.learning_rate),
            "--batch-size", str(self.batch_size),
            "--buffer-size", str(self.buffer_size),
            "--tau", str(self.tau),
            "--gamma", str(self.gamma),
            "--exploration-noise", str(self.exploration_noise),
            "--output-dir", str(run_dir),
        ]

        # Forward any extra CLI arguments from config
        for key, value in config.algorithm_params.get("extra_args", {}).items():
            cmd.extend([f"--{key}", str(value)])

        return cmd

    def _run_subprocess(self, cmd: List[str]) -> Dict[str, Any]:
        """Launch cleanMARL subprocess and collect training metrics.

        Streams stdout line-by-line, parses metrics, and waits for
        the process to finish (or kills it on timeout).

        Args:
            cmd: Command list passed to subprocess.Popen.

        Returns:
            Dictionary with parsed metrics and return code.

        Raises:
            RuntimeError: If the process exits with a non-zero code.
        """
        metrics: Dict[str, Any] = {
            "episode_rewards": [],
            "global_steps": [],
            "stdout_tail": [],
            "timeout": False,
        }

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                stripped = line.rstrip()
                metrics["stdout_tail"].append(stripped)
                if len(metrics["stdout_tail"]) > 200:
                    metrics["stdout_tail"] = metrics["stdout_tail"][-100:]
                self._parse_progress_line(stripped, metrics)

            proc.wait(timeout=self.timeout)

        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            metrics["timeout"] = True

        metrics["return_code"] = proc.returncode

        if proc.returncode not in (0, None) and not metrics["timeout"]:
            raise RuntimeError(
                f"cleanMARL exited with code {proc.returncode}. "
                f"Last output: {metrics['stdout_tail'][-5:]}"
            )

        return metrics

    def _parse_progress_line(self, line: str, metrics: Dict[str, Any]) -> None:
        """Parse a single stdout line from cleanMARL for training metrics.

        Handles both JSON format (``{"global_step": 1000, "episodic_return": -5.2}``)
        and key=value format (``global_step=1000, episodic_return=-5.2``).

        Args:
            line: Single stdout line (stripped of newline).
            metrics: Metrics dict to update in-place.
        """
        if "episodic_return" not in line and "episode_reward" not in line:
            return

        try:
            if line.strip().startswith("{"):
                data = json.loads(line.strip())
                if "episodic_return" in data:
                    metrics["episode_rewards"].append(float(data["episodic_return"]))
                if "global_step" in data:
                    metrics["global_steps"].append(int(data["global_step"]))
            else:
                for part in line.split(","):
                    if "=" not in part:
                        continue
                    key, _, val = part.strip().partition("=")
                    key = key.strip()
                    val = val.strip()
                    if key == "episodic_return":
                        metrics["episode_rewards"].append(float(val))
                    elif key == "global_step":
                        metrics["global_steps"].append(int(val))
        except (ValueError, json.JSONDecodeError):
            pass


# ---------------------------------------------------------------------------
# Module-level helper: build cleanMARL-compatible actor architecture
# ---------------------------------------------------------------------------


def _build_cleanmarl_actor(
    obs_dim: int,
    act_dim: int,
    hidden_dim: int = 256,
    action_scale: float = 1.0,
    action_bias: float = 0.0,
):
    """Build a feedforward actor matching cleanMARL's MADDPG actor architecture.

    The architecture mirrors ``cleanrl/maddpg_continuous_action.py``::

        fc1 -> ReLU -> fc2 -> ReLU -> fc_mu -> tanh -> rescale

    Args:
        obs_dim: Input (observation) dimension.
        act_dim: Output (action) dimension.
        hidden_dim: Width of both hidden layers.
        action_scale: Multiplicative rescaling for tanh output.
        action_bias: Additive bias after rescaling.

    Returns:
        Instantiated nn.Module actor network.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to build MADDPG actor network. "
            "Install with: pip install torch"
        ) from exc

    class _Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(obs_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc_mu = nn.Linear(hidden_dim, act_dim)
            self.register_buffer(
                "action_scale",
                torch.tensor(action_scale, dtype=torch.float32),
            )
            self.register_buffer(
                "action_bias",
                torch.tensor(action_bias, dtype=torch.float32),
            )

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.tanh(self.fc_mu(x))
            return x * self.action_scale + self.action_bias

    return _Actor()
