"""
Tests for the unified baseline API.

Tests the core abstractions, EA trainer, and comparison workflow.
"""

import sys
import pytest
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

from deap_pettingzoo.baselines import (
    BudgetMode,
    TrainingConfig,
    EvalResult,
    ComparisonConfig,
    EATrainer,
    MADDPGTrainer,
    MADDPGPolicyWrapper,
    UnifiedEvaluator,
    compare_baselines,
    register_trainer,
    get_trainer,
)


class TestDataModels:
    """Test data model validation and serialization."""
    
    def test_training_config_validation(self):
        """Test TrainingConfig validation."""
        # Valid config
        config = TrainingConfig(
            env_name='simple_spread_v3',
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=1000,
            num_seeds=1,
        )
        assert config.budget_value == 1000
        assert config.budget_mode == BudgetMode.ENV_STEPS
        
        # Invalid budget value
        with pytest.raises(ValueError):
            TrainingConfig(
                env_name='simple_spread_v3',
                budget_mode=BudgetMode.ENV_STEPS,
                budget_value=-100,  # Negative
            )
        
        # String budget mode conversion
        config = TrainingConfig(
            env_name='simple_spread_v3',
            budget_mode='env_steps',  # String
            budget_value=1000,
        )
        assert config.budget_mode == BudgetMode.ENV_STEPS
    
    def test_eval_result_serialization(self):
        """Test EvalResult to_dict conversion."""
        result = EvalResult(
            mean_reward=10.5,
            std_reward=2.3,
            min_reward=5.0,
            max_reward=15.0,
            num_episodes=10,
            episode_rewards=[10.0, 11.0, 9.0],
            episode_lengths=[50, 52, 48],
        )
        
        data = result.to_dict()
        assert data['mean_reward'] == 10.5
        assert data['num_episodes'] == 10
        assert len(data['episode_rewards']) == 3
    
    def test_comparison_config_validation(self):
        """Test ComparisonConfig validation."""
        # Valid config
        config = ComparisonConfig(
            algorithms=['ea'],
            env_name='simple_spread_v3',
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=1000,
        )
        assert len(config.algorithms) == 1
        
        # Empty algorithms list
        with pytest.raises(ValueError):
            ComparisonConfig(
                algorithms=[],  # Empty
                env_name='simple_spread_v3',
                budget_mode=BudgetMode.ENV_STEPS,
                budget_value=1000,
            )


class TestEATrainer:
    """Test EA trainer implementation."""
    
    def test_ea_trainer_initialization(self):
        """Test EA trainer can be initialized."""
        trainer = EATrainer(
            hidden_size=32,
            population_size=10,
        )
        assert trainer.hidden_size == 32
        assert trainer.population_size == 10
    
    def test_ea_trainer_capabilities(self):
        """Test EA trainer capabilities reporting."""
        trainer = EATrainer()
        caps = trainer.get_capabilities()
        
        assert caps['discrete_action'] is True
        assert caps['continuous_action'] is True
        assert caps['needs_global_state'] is False
    
    def test_ea_trainer_budget_support(self):
        """Test EA trainer budget mode support."""
        trainer = EATrainer()
        
        assert trainer.supports_budget_mode(BudgetMode.ENV_STEPS) is True
        assert trainer.supports_budget_mode(BudgetMode.EVAL_EPISODES) is True
        assert trainer.supports_budget_mode(BudgetMode.WALL_CLOCK) is False
    
    def test_ea_trainer_budget_translation(self):
        """Test budget translation to EA parameters."""
        trainer = EATrainer(population_size=20)
        
        # ENV_STEPS mode
        config = TrainingConfig(
            env_name='simple_spread_v3',
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=10000,  # 10k steps
            algorithm_params={'max_steps_per_episode': 200},
        )
        params = trainer._translate_budget(config)
        # 10000 / (20 * 200) = 2.5 -> 2 generations
        assert params['num_generations'] == 2
        
        # EVAL_EPISODES mode
        config = TrainingConfig(
            env_name='simple_spread_v3',
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=100,  # 100 episodes
        )
        params = trainer._translate_budget(config)
        # 100 / 20 = 5 generations
        assert params['num_generations'] == 5
    
    @pytest.mark.slow
    def test_ea_trainer_training_smoke(self):
        """Smoke test: EA trainer can complete a tiny training run."""
        trainer = EATrainer(
            hidden_size=16,
            population_size=5,
        )
        
        config = TrainingConfig(
            env_name='simple_spread_v3',
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=5000,  # Very small budget
            algorithm_params={'max_steps_per_episode': 100},
        )
        
        result = trainer.train(config)
        
        assert result.algorithm == 'ea'
        assert result.policy is not None
        assert 'fitness_history' in result.metrics
        assert result.training_time is not None
        assert result.training_time > 0


class TestUnifiedEvaluator:
    """Test unified evaluator."""
    
    def test_evaluator_initialization(self):
        """Test evaluator can be initialized."""
        evaluator = UnifiedEvaluator(max_steps=200)
        assert evaluator.max_steps == 200
    
    @pytest.mark.slow
    def test_evaluator_smoke(self):
        """Smoke test: evaluate a trained EA policy."""
        # Train a tiny EA policy first
        trainer = EATrainer(hidden_size=16, population_size=5)
        config = TrainingConfig(
            env_name='simple_spread_v3',
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=5000,
            algorithm_params={'max_steps_per_episode': 100},
        )
        training_result = trainer.train(config)
        
        # Evaluate it
        evaluator = UnifiedEvaluator(max_steps=100)
        eval_result = evaluator.evaluate(
            policy=training_result.policy,
            env_name='simple_spread_v3',
            num_episodes=3,
            seed=42,
        )
        
        assert eval_result.num_episodes == 3
        assert len(eval_result.episode_rewards) == 3
        assert len(eval_result.episode_lengths) == 3
        assert eval_result.mean_reward is not None
        
        evaluator.close()


class TestTrainerRegistry:
    """Test trainer registration system."""
    
    def test_get_builtin_trainer(self):
        """Test getting built-in EA trainer."""
        trainer = get_trainer('ea')
        assert isinstance(trainer, EATrainer)
    
    def test_get_builtin_maddpg_trainer(self):
        """Test getting built-in MADDPG trainer."""
        trainer = get_trainer('maddpg')
        assert isinstance(trainer, MADDPGTrainer)
    
    def test_get_unknown_trainer(self):
        """Test getting unknown trainer raises error."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            get_trainer('nonexistent_algorithm')
    
    def test_register_custom_trainer(self):
        """Test registering a custom trainer."""
        from deap_pettingzoo.baselines import BaselineTrainer
        
        class CustomTrainer(BaselineTrainer):
            def train(self, config):
                pass
            def supports_budget_mode(self, mode):
                return True
            def get_capabilities(self):
                return {}
        
        register_trainer('custom', CustomTrainer)
        trainer = get_trainer('custom')
        assert isinstance(trainer, CustomTrainer)


class TestComparisonWorkflow:
    """Test end-to-end comparison workflow."""
    
    @pytest.mark.slow
    def test_compare_baselines_smoke(self):
        """Smoke test: compare_baselines with EA on tiny budget."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ComparisonConfig(
                algorithms=['ea'],
                env_name='simple_spread_v3',
                budget_mode=BudgetMode.ENV_STEPS,
                budget_value=5000,  # Very small
                num_seeds=1,  # Single seed for speed
                eval_episodes=3,  # Few evaluation episodes
                algorithm_configs={
                    'ea': {
                        'trainer_params': {
                            'hidden_size': 16,
                            'population_size': 5,
                        },
                        'algorithm_params': {
                            'max_steps_per_episode': 100,
                        },
                    },
                },
                output_dir=Path(tmpdir) / 'results',
            )
            
            results = compare_baselines(config)
            
            # Check result structure
            assert 'ea' in results
            assert 'config' in results
            
            # Check EA results
            ea_results = results['ea']
            assert 'seeds' in ea_results
            assert len(ea_results['seeds']) == 1
            assert 'mean_reward' in ea_results
            assert 'std_reward' in ea_results
            assert 'best_seed' in ea_results
            
            # Check results file was created
            results_file = Path(tmpdir) / 'results' / 'comparison_results.json'
            assert results_file.exists()


class TestMADDPGTrainer:
    """Tests for the MADDPG trainer (non-slow, no cleanMARL required)."""

    def test_maddpg_trainer_initialization(self):
        """Test MADDPGTrainer can be instantiated with defaults."""
        trainer = MADDPGTrainer()
        assert trainer.hidden_dim == 256
        assert trainer.learning_rate == 3e-4
        assert trainer.device == 'cpu'

    def test_maddpg_trainer_custom_params(self):
        """Test MADDPGTrainer accepts custom hyperparameters."""
        trainer = MADDPGTrainer(
            hidden_dim=128,
            learning_rate=1e-3,
            batch_size=64,
            tau=0.01,
            gamma=0.95,
        )
        assert trainer.hidden_dim == 128
        assert trainer.learning_rate == 1e-3
        assert trainer.batch_size == 64

    def test_maddpg_budget_support(self):
        """Test MADDPG supports ENV_STEPS and WALL_CLOCK."""
        trainer = MADDPGTrainer()
        assert trainer.supports_budget_mode(BudgetMode.ENV_STEPS) is True
        assert trainer.supports_budget_mode(BudgetMode.WALL_CLOCK) is True
        assert trainer.supports_budget_mode(BudgetMode.EVAL_EPISODES) is False

    def test_maddpg_capabilities(self):
        """Test MADDPG capabilities metadata."""
        trainer = MADDPGTrainer()
        caps = trainer.get_capabilities()
        assert caps['continuous_action'] is True
        assert caps['discrete_action'] is False
        assert caps['needs_global_state'] is False
        assert caps['on_policy'] is False

    def test_maddpg_budget_translation_env_steps(self):
        """Test ENV_STEPS budget maps directly to total_timesteps."""
        trainer = MADDPGTrainer()
        config = TrainingConfig(
            env_name='simple_spread_v3',
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=500_000,
        )
        assert trainer._translate_budget(config) == 500_000

    def test_maddpg_budget_translation_wall_clock(self):
        """Test WALL_CLOCK budget uses default_timesteps from params."""
        trainer = MADDPGTrainer()
        config = TrainingConfig(
            env_name='simple_spread_v3',
            budget_mode=BudgetMode.WALL_CLOCK,
            budget_value=3600,
            algorithm_params={'default_timesteps': 2_000_000},
        )
        assert trainer._translate_budget(config) == 2_000_000

    def test_maddpg_unsupported_budget_raises(self):
        """Test unsupported budget mode raises ValueError."""
        trainer = MADDPGTrainer()
        config = TrainingConfig(
            env_name='simple_spread_v3',
            budget_mode=BudgetMode.EVAL_EPISODES,
            budget_value=100,
        )
        with pytest.raises(ValueError, match="does not support budget mode"):
            trainer._translate_budget(config)

    def test_maddpg_no_script_raises_import_error(self):
        """Test that training without cleanMARL installed raises ImportError."""
        trainer = MADDPGTrainer(cleanmarl_script=None)
        config = TrainingConfig(
            env_name='simple_spread_v3',
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=10_000,
        )
        # Patch cleanrl import to simulate it not being installed
        with patch.dict('sys.modules', {'cleanrl': None}):
            with pytest.raises(ImportError, match="cleanMARL MADDPG script not found"):
                trainer._resolve_script(config)

    def test_maddpg_explicit_script_path_used(self):
        """Test that explicit cleanmarl_script path is returned as-is."""
        trainer = MADDPGTrainer(cleanmarl_script='/path/to/maddpg.py')
        config = TrainingConfig(
            env_name='simple_spread_v3',
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=10_000,
        )
        assert trainer._resolve_script(config) == '/path/to/maddpg.py'

    def test_maddpg_config_script_override(self):
        """Test algorithm_params cleanmarl_script overrides auto-discovery."""
        trainer = MADDPGTrainer()
        config = TrainingConfig(
            env_name='simple_spread_v3',
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=10_000,
            algorithm_params={'cleanmarl_script': '/custom/maddpg.py'},
        )
        assert trainer._resolve_script(config) == '/custom/maddpg.py'

    def test_maddpg_command_building(self, tmp_path):
        """Test subprocess command is built with correct arguments."""
        trainer = MADDPGTrainer(
            learning_rate=1e-3,
            batch_size=128,
            exploration_noise=0.2,
        )
        config = TrainingConfig(
            env_name='simple_spread_v3',
            budget_mode=BudgetMode.ENV_STEPS,
            budget_value=100_000,
            algorithm_params={'seed': 42},
        )
        cmd = trainer._build_command(
            script='/path/to/maddpg.py',
            config=config,
            total_timesteps=100_000,
            run_dir=tmp_path,
        )
        assert '/path/to/maddpg.py' in cmd
        assert '--env-id' in cmd
        assert 'simple_spread_v3' in cmd
        assert '--total-timesteps' in cmd
        assert '100000' in cmd
        assert '--seed' in cmd
        assert '42' in cmd
        assert '--save-model' in cmd
        assert '--learning-rate' in cmd
        assert '0.001' in cmd

    def test_maddpg_progress_parsing_json(self):
        """Test JSON-format stdout line parsing."""
        trainer = MADDPGTrainer()
        metrics: dict = {'episode_rewards': [], 'global_steps': []}
        trainer._parse_progress_line(
            '{"global_step": 1000, "episodic_return": -12.5}', metrics
        )
        assert metrics['episode_rewards'] == [-12.5]
        assert metrics['global_steps'] == [1000]

    def test_maddpg_progress_parsing_kv(self):
        """Test key=value format stdout line parsing."""
        trainer = MADDPGTrainer()
        metrics: dict = {'episode_rewards': [], 'global_steps': []}
        trainer._parse_progress_line(
            'global_step=2000, episodic_return=-8.3', metrics
        )
        assert metrics['episode_rewards'] == [-8.3]
        assert metrics['global_steps'] == [2000]

    def test_maddpg_progress_parsing_ignores_unrelated(self):
        """Test that unrelated lines leave metrics unchanged."""
        trainer = MADDPGTrainer()
        metrics: dict = {'episode_rewards': [], 'global_steps': []}
        trainer._parse_progress_line('Setting up environment...', metrics)
        assert metrics['episode_rewards'] == []

    def test_maddpg_subprocess_failure_raises(self, tmp_path):
        """Test that a non-zero subprocess exit raises RuntimeError."""
        trainer = MADDPGTrainer()
        cmd = [sys.executable, '-c', 'import sys; sys.exit(1)']
        with pytest.raises(RuntimeError, match="cleanMARL exited with code"):
            trainer._run_subprocess(cmd)

    def test_maddpg_policy_wrapper_missing_torch(self, tmp_path):
        """Test MADDPGPolicyWrapper raises ImportError when torch unavailable."""
        with patch.dict('sys.modules', {'torch': None}):
            with pytest.raises(ImportError, match="PyTorch is required"):
                MADDPGPolicyWrapper(
                    checkpoint_dir=tmp_path,
                    obs_dim=18,
                    act_dim=5,
                )

    def test_maddpg_policy_wrapper_missing_checkpoint(self, tmp_path):
        """Test MADDPGPolicyWrapper raises FileNotFoundError when no checkpoint."""
        pytest.importorskip('torch')
        with pytest.raises(FileNotFoundError, match="No actor checkpoint found"):
            MADDPGPolicyWrapper(
                checkpoint_dir=tmp_path,
                obs_dim=18,
                act_dim=5,
            )

    @pytest.mark.skipif(
        not __import__('importlib').util.find_spec('torch'),
        reason="torch not installed"
    )
    def test_maddpg_policy_wrapper_loads_and_acts(self, tmp_path):
        """Test MADDPGPolicyWrapper loads a saved checkpoint and returns actions."""
        import torch
        from deap_pettingzoo.baselines.maddpg_trainer import _build_cleanmarl_actor

        obs_dim, act_dim = 18, 5
        actor = _build_cleanmarl_actor(obs_dim, act_dim, hidden_dim=64)
        torch.save(actor.state_dict(), tmp_path / 'actor.pt')

        wrapper = MADDPGPolicyWrapper(
            checkpoint_dir=tmp_path,
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dim=64,
        )
        obs = np.zeros(obs_dim, dtype=np.float32)
        action = wrapper.get_action(obs)
        assert action.shape == (act_dim,)

        wrapper.reset()  # should be no-op, not raise

