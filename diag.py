import numpy as np
from deap_pettingzoo.genomes import FixedTopologyGenome
import pettingzoo.mpe.simple_tag_v3 as e
for hidden in [64, 32, 16]:
    g = FixedTopologyGenome([16, hidden, 5])
    print(f"  hidden={hidden}: {g.get_weight_count()} params")
env = e.parallel_env(max_cycles=1)
obs, _ = env.reset(seed=42)
adv = next(k for k in obs if k.startswith("adversary"))
print(f"obs shape: {obs[adv].shape}")
print(f"obs values: {np.round(obs[adv], 3)}")
print(f"obs[12:14] = prey relative position: {np.round(obs[adv][12:14], 3)}")
env.close()
