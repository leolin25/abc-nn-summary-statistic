from simulator.sampler import (
    abc_smc,
    canonical_summary
)
from simulator.exact_sim import ou_exact
import numpy as np

# simulate an Ornstein-Uhlenbeck process to generate observed data
obs = ou_exact(h=0.01, T=10.0, mu=0.1, sigma=0.2, tau=0.5, x0=0.0, n_paths=1)[0].flatten()

results = abc_smc(
    obs=obs,
    prior_bounds=np.array([[0.0, 0.2], [0.1, 0.3], [0.4, 0.6]]),
    simulator=ou_exact,
    summarise=canonical_summary,
    n_part=1000,
    max_iter=5,
    init_size=100,
    q=0.1
)

print(results)