"""Tests for NEAT config builder."""

import pytest

neat = pytest.importorskip("neat")

from deap_pettingzoo.contrib.neat.config import build_neat_config


class TestBuildNeatConfig:
    """Tests for build_neat_config()."""

    def test_minimal_config(self):
        cfg = build_neat_config(input_size=8, output_size=5)
        assert cfg.genome_config.num_inputs == 8
        assert cfg.genome_config.num_outputs == 5
        assert cfg.pop_size == 50  # default

    def test_custom_pop_size(self):
        cfg = build_neat_config(input_size=4, output_size=2, pop_size=30)
        assert cfg.pop_size == 30

    def test_with_seed(self):
        cfg = build_neat_config(input_size=4, output_size=2, seed=42)
        assert getattr(cfg, "seed", None) == 42

    def test_feed_forward_flag(self):
        cfg = build_neat_config(input_size=3, output_size=1)
        assert cfg.genome_config.feed_forward is True

    def test_overrides(self):
        overrides = {
            "NEAT": {"pop_size": 100},
            "DefaultGenome": {"activation_default": "relu"},
        }
        cfg = build_neat_config(
            input_size=6, output_size=3, overrides=overrides
        )
        assert cfg.pop_size == 100
        assert cfg.genome_config.activation_default == "relu"

    def test_config_is_usable(self):
        """The built config must be accepted by neat.Population."""
        cfg = build_neat_config(input_size=4, output_size=2, pop_size=10)
        pop = neat.Population(cfg)
        assert pop is not None
        assert len(pop.population) == 10
