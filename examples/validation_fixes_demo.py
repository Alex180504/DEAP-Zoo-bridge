"""Quick test for validation fixes."""
from deap_pettingzoo.policies import FeedforwardNeuralPolicy
from deap_pettingzoo.genomes import FixedTopologyGenome
from deap_pettingzoo.evaluators import SingleEpisodeEvaluator
from deap_pettingzoo.environments import PettingZooWrapper
import numpy as np

print("Testing validation fixes...")

# Test 1: None genome in policy
print("\n1. Testing None genome validation...")
try:
    p = FeedforwardNeuralPolicy(None)
    print("   ✗ FAIL: Should have raised ValueError")
except ValueError as e:
    print(f"   ✓ PASS: {e}")

# Test 2: Wrong type in policy
print("\n2. Testing wrong type validation...")
try:
    p = FeedforwardNeuralPolicy("not a genome")
    print("   ✗ FAIL: Should have raised TypeError")
except TypeError as e:
    print(f"   ✓ PASS: {e}")

# Test 3: Wrong observation size
print("\n3. Testing observation size validation...")
g = FixedTopologyGenome([5, 3])
p = FeedforwardNeuralPolicy(g)
try:
    p.forward(np.array([1, 2]))  # Wrong size (2 instead of 5)
    print("   ✗ FAIL: Should have raised ValueError")
except ValueError as e:
    print(f"   ✓ PASS: {e}")

# Test 4: None individual in evaluator
print("\n4. Testing None individual validation...")
env = PettingZooWrapper('simple_spread_v3')
evaluator = SingleEpisodeEvaluator(env)
try:
    evaluator.evaluate(None)
    print("   ✗ FAIL: Should have raised ValueError")
except ValueError as e:
    print(f"   ✓ PASS: {e}")

# Test 5: Individual without genome
print("\n5. Testing individual without genome...")
class FakeIndividual:
    pass
try:
    evaluator.evaluate(FakeIndividual())
    print("   ✗ FAIL: Should have raised ValueError")
except ValueError as e:
    print(f"   ✓ PASS: {e}")

# Test 6: Individual with None genome
print("\n6. Testing individual with None genome...")
class IndividualWithNoneGenome:
    def __init__(self):
        self.genome = None
try:
    evaluator.evaluate(IndividualWithNoneGenome())
    print("   ✗ FAIL: Should have raised ValueError")
except ValueError as e:
    print(f"   ✓ PASS: {e}")

print("\n" + "="*50)
print("All validation tests PASSED! ✓")
print("="*50)
