from examples.pursuit_evasion.aquarium_ea_cma_mappo import APF_CONFIG, ENV_CONFIG


def test_prey_speed_advantage_config() -> None:
    # Prey are intentionally faster than predators on a torus to prevent
    # diagonal-drift exploits — see the ENV_CONFIG comment.
    assert ENV_CONFIG["prey_max_velocity"] > ENV_CONFIG["predator_max_velocity"]
    # Other parameters remain symmetric so only the speed cap differs.
    assert ENV_CONFIG["predator_max_acceleration"] == ENV_CONFIG["prey_max_acceleration"]
    assert ENV_CONFIG["predator_max_steer_force"] == ENV_CONFIG["prey_max_steer_force"]
    assert ENV_CONFIG["predator_view_distance"] == ENV_CONFIG["prey_view_distance"]
    assert ENV_CONFIG["predator_observe_count"] == ENV_CONFIG["prey_observe_count"]
    # FOV is intentionally asymmetric: predators have limited FOV (120°) so
    # they must actively steer toward prey; prey retain 360° awareness.
    assert ENV_CONFIG["fov_enabled"] is True
    assert ENV_CONFIG["predator_fov"] < ENV_CONFIG["prey_fov"]


def test_aquarium_experiment_uses_hardened_prey_apf_config() -> None:
    assert APF_CONFIG.k_predator == 5.0
    assert APF_CONFIG.k_tangential == 3.0
    assert APF_CONFIG.k_prey_separation == 0.5