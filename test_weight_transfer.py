"""
Quick verification that weight transfer works for heterogeneous agents.
Tests the fix for the "seed 42 bug" where heterogeneous agents ignored evolved weights.
"""

import numpy as np
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.environments import PettingZooWrapper
from deap_pettingzoo.evaluators import SingleEpisodeEvaluator

def test_heterogeneous_weight_transfer():
    """Verify that evolved weights transfer to heterogeneous agents."""
    
    # Create a genome for simple_tag_v3 (pursuer obs=16, evader obs=14)
    genome = FixedTopologyGenome(
        layer_sizes=[16, 32, 5],  # For pursuer
        activation='tanh',
        random_seed=42
    )
    
    # Get the evolved weights
    evolved_weights = genome.get_weights()
    print(f"\nEvolved genome has {len(evolved_weights)} weights")
    print(f"First 10 weights: {evolved_weights[:10]}")
    
    # Create evaluator with simple_tag_v3 (heterogeneous environment)
    env_wrapper = PettingZooWrapper(env_name='simple_tag_v3')
    evaluator = SingleEpisodeEvaluator(
        env_wrapper=env_wrapper,
        fitness_aggregation='sum'
    )
    
    # Reset environment to get observations
    observations = env_wrapper.reset(seed=42)
    
    print(f"\nAgent observation sizes:")
    for agent, obs in observations.items():
        print(f"  {agent}: {len(obs)} dimensions")
    
    # Create policies using the evaluator's method
    policies = evaluator._create_policies(genome, observations)
    
    print(f"\nCreated {len(policies)} policies")
    
    # Check that agent-specific weights were transferred
    for agent, policy in policies.items():
        agent_weights = policy.genome.get_weights()
        print(f"\n{agent}:")
        print(f"  Genome has {len(agent_weights)} weights")
        print(f"  First 10 weights: {agent_weights[:10]}")
        
        # For heterogeneous agents, check that some weights match evolved weights
        # The hidden layer and parts of output layer should match
        # Input layer will differ due to different observation sizes
        
        # Get layer info
        source_layers = evaluator._unflatten_weights(genome, evolved_weights)
        target_layers = evaluator._unflatten_weights(policy.genome, agent_weights)
        
        print(f"  Source hidden→hidden weights shape: {source_layers[1]['weights'].shape}")
        print(f"  Target hidden→hidden weights shape: {target_layers[1]['weights'].shape}")
        
        # Check if hidden layer weights match (they should for compatible architectures)
        if source_layers[1]['weights'].shape == target_layers[1]['weights'].shape:
            weights_match = np.allclose(
                source_layers[1]['weights'],
                target_layers[1]['weights']
            )
            print(f"  ✓ Hidden layer weights transferred: {weights_match}")
            assert weights_match, "Hidden layer weights should match for transferred weights"
        else:
            print(f"  ⚠ Hidden layer shapes don't match - expected for this test")
    
    print("\n✓ Weight transfer test passed!")

if __name__ == "__main__":
    test_heterogeneous_weight_transfer()
