from simulator import abcsmc_forward, ou_exact
from simulator.sum_statistics import BaseSummary, CanonicalSummary, ModelSummary, NeuralSummary
from simulator.utilities import sample_prior
from functools import partial
import numpy as np

"""
This file is for experimenting the ABC-SMC loop with model-based summary statistics.
"""

# simulate an Ornstein-Uhlenbeck process to generate observed data
obs = ou_exact(h=0.01, T=10.0, mu=0.1, sigma=0.2, tau=0.5, x0=0.0, n_paths=1)[0].flatten()

def simulate_ou(particle):
    mu, sigma, tau = particle
    trajectory = ou_exact(
        h=0.01,
        T=10.0,
        mu=mu,
        sigma=sigma,
        tau=tau,
        x0=0.0,
        n_paths=1,
    )[0]
    return trajectory.flatten()

# sample 1000 particles from prior to fit summary statistic
prior_bounds = np.array([[0.0, 0.2], [0.1, 0.3], [0.4, 0.6]])
n_samples = 1000

# prior_bounds shape: (3, 2) where row 0 is mu, row 1 is sigma, row 2 is tau
low_bounds = prior_bounds[:, 0]
high_bounds = prior_bounds[:, 1]

prior_particles = np.random.uniform(
    low=low_bounds, 
    high=high_bounds, 
    size=(n_samples, len(prior_bounds))
)

# simulate trajectories for each sampled particle vector
calibration_paths = np.array([simulate_ou(particle) for particle in prior_particles])

# initialize canonical sum stats
summarise = CanonicalSummary()
summarise.fit(calibration_paths)

if isinstance(summarise, CanonicalSummary):
    print('Fitted summary statistic medians: ', summarise.medians)
    print('Fitted summary statistic MADs: ', summarise.mads)

results = abcsmc_forward(
    obs=obs,
    prior_bounds=prior_bounds,
    simulator=simulate_ou,
    summarise=summarise,
    n_part=1000,
    max_iter=5,
    q=0.1
)
"""
tuple: A tuple containing the following arrays up to the final completed round (t+1):
            - abc_particles (np.ndarray): The accepted parameter vectors forming the posterior.
            - abc_weights (np.ndarray): The calculated Sequential Monte Carlo importance weights.
            - abc_round_time (np.ndarray): The computation time (in seconds) spent simulating and sampling per round.
            - acc_rates (np.ndarray): The percentage of proposed simulations accepted in each round.
            - obs_summaries (np.ndarray): The neural network's evolving summary statistic representation of the observed data.
            - abc_thresholds (np.ndarray): The epsilon distance thresholds enforced at each round.
"""
print('Final ABC-SMC results:')
print('Particles: ', results[0])
print('Weights: ', results[1])
print('Round times: ', results[2])
print('Acceptance rates: ', results[3])
print('Observed summaries: ', results[4])
print('ABC thresholds: ', results[5])