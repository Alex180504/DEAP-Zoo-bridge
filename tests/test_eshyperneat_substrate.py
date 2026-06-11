"""Tests for HyperNEAT / ES-HyperNEAT substrate builders."""

import pytest

neat = pytest.importorskip("neat")

from deap_pettingzoo.contrib.neat._pureples.substrate import Substrate
from deap_pettingzoo.contrib.neat.substrate import (
    make_uniform_substrate,
    make_2d_grid_substrate,
    substrate_for_env,
)


class TestSubstrate:
    def test_smoke_attributes(self):
        sub = make_uniform_substrate(4, 3)
        assert sub.num_inputs == 4
        assert sub.num_outputs == 3
        assert sub.num_hidden_layers == 0

    def test_with_hidden_layers(self):
        sub = make_uniform_substrate(4, 2, hidden_layers=[3, 3])
        assert sub.num_hidden_layers == 2
        assert len(sub.hidden_coordinates[0]) == 3
        assert len(sub.hidden_coordinates[1]) == 3

    def test_coordinates_in_range(self):
        sub = make_uniform_substrate(6, 4, hidden_layers=[3])
        for x, y in sub.input_coordinates:
            assert -1.0 <= x <= 1.0
            assert y == -1.0
        for x, y in sub.output_coordinates:
            assert -1.0 <= x <= 1.0
            assert y == 1.0
        for layer in sub.hidden_coordinates:
            for x, y in layer:
                assert -1.0 <= x <= 1.0
                assert -1.0 < y < 1.0

    def test_2d_grid_shape(self):
        sub = make_2d_grid_substrate((3, 4), 2)
        assert sub.num_inputs == 3 * 4
        assert sub.num_outputs == 2

    def test_substrate_for_env_uniform(self):
        sub = substrate_for_env(10, 5, layout="uniform")
        assert sub.num_inputs == 10
        assert sub.num_outputs == 5

    def test_substrate_for_env_2d_grid(self):
        sub = substrate_for_env(9, 3, layout="2d_grid")
        # 9 -> ceil(sqrt(9)) = 3, so 3×3 = 9 inputs
        assert sub.num_inputs == 9

    def test_empty_substrate(self):
        sub = make_uniform_substrate(0, 0)
        assert sub.num_inputs == 0
        assert sub.num_outputs == 0
