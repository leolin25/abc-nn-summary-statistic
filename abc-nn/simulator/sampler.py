import time
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
from functools import partial
from .utilities import (
    sample_prior,
    gaussian_proposal,
    compute_param_ratio,
    logsumexp
)
from .sum_statistics import BaseSummary, CanonicalSummary, ModelSummary, NeuralSummary

def abcsmc_forward(obs: np.ndarray, prior_bounds: np.ndarray, simulator: callable, summarise: BaseSummary, n_part: int, max_iter: int, q: float):
    """
    Perform Approximate Bayesian Computation with Sequential Monte Carlo (ABC-SMC) loop.

    This function performs parameter inference by simulating data, dynamically training a neural network to extract summary statistics, and progressively tightening the acceptance threshold over multiple rounds.
    Args:
        obs (np.ndarray): observed data time-series data
        prior_bounds (np.ndarray): bounds for the prior distribution of the parameters
        simulator (function): a function that simulates a time-series
        summarise (Object): an instance of a summary statistic class
        n_part (int): number of particles to use in the ABC-SMC algorithm
        max_iter (int): maximum number of iterations for the ABC-SMC algorithm
        q (float): quantile for the ABC-SMC algorithm
    Returns:
        tuple: A tuple containing the following arrays up to the final completed round (t+1):
            - abc_particles (np.ndarray): The accepted parameter vectors forming the posterior.
            - abc_weights (np.ndarray): The calculated Sequential Monte Carlo importance weights.
            - abc_round_time (np.ndarray): The computation time (in seconds) spent simulating and sampling per round.
            - acc_rates (np.ndarray): The percentage of proposed simulations accepted in each round.
            - obs_summaries (np.ndarray): The neural network's evolving summary statistic representation of the observed data.
            - abc_thresholds (np.ndarray): The epsilon distance thresholds enforced at each round.
    """
    num_sum_stats = summarise.get_num_sum_stats()
    abc_round_time = np.zeros(max_iter)
    acc_rate = np.zeros(max_iter)

    # particle system, covariance matrix and thresholds
    param_dim = len(prior_bounds)
    abc_particles = np.zeros((max_iter, n_part, param_dim))
    abc_weights = np.zeros((max_iter, n_part))
    abc_cov = np.zeros((param_dim, param_dim))
    abc_thresholds = np.zeros(max_iter)

    obs_summaries = np.zeros((max_iter, num_sum_stats))

    # initialize abc-smc
    abc_thresholds[0] = np.inf
    abc_weights[0] = np.log(1 / n_part) # uniform weights for this round
    obs_summaries[0] = summarise(obs)

    # run abc-smc loop
    for t in range(max_iter):
        print(f'Round {t+1}')

        # instantiate prior density
        if t == 0:
            proposal = partial(sample_prior, prior_bounds=prior_bounds)
        else:
            proposal = partial(
                gaussian_proposal,
                abc_particles[t - 1],
                abc_weights[t - 1],
                np.linalg.cholesky(abc_cov),
                prior_bounds,
            )

        total_sim = 0
        round_start = time.time()
        distances = np.zeros(n_part)

        # iterate until we have n_part accepted particles
        for i in tqdm(range(n_part)):
            while True:
                # sample and simulate
                particle = proposal()
                path = simulator(particle)

                # reject numerical error
                if np.any(np.isnan(path)) or np.any(path >= 10e7):
                    continue

                total_sim += 1

                # summarise and calculate distance between observed ss and simulated ss
                sim_summary = summarise(path)
                dist = np.linalg.norm(obs_summaries[t] - sim_summary)

                if dist <= abc_thresholds[t]:
                    abc_particles[t, i] = particle
                    distances[i] = dist
                    break

        # compute weights
        if t > 0:
            compute_param_ratio(
                    t=t,
                    prior_bounds=prior_bounds,
                    particles=abc_particles,
                    weights=abc_weights,
                    cov_det=np.linalg.det(abc_cov),
                    cov_inv=np.linalg.inv(abc_cov),
                )

        # normalise weights and calculate covariance matrix for next round
        abc_weights[t] = np.exp(abc_weights[t] - logsumexp(abc_weights[t]))
        abc_cov = 2.0 * np.cov(abc_particles[t].T, aweights=abc_weights[t])

        # total round time
        abc_round_time[t] = time.time() - round_start

        # calculate acceptance rate and stop if below 1.5%
        acceptance_rate = (n_part / total_sim) * 100.0
        acc_rate[t] = acceptance_rate

        # last round
        if t == max_iter - 1:
            break

        # determine next round threshold
        new_threshold = np.quantile(distances, q)
        if new_threshold <= abc_thresholds[t]:
            abc_thresholds[t + 1] = new_threshold
        else:
            abc_thresholds[t + 1] = abc_thresholds[t] * 0.95

        print(f"The threshold for this round was {abc_thresholds[t]}")
        print(f"Acceptance rate is {round(acceptance_rate, 2)}%")

        if t > 2 and acceptance_rate < 1.5:
            print(f"Acceptance rate below 1.5%, terminating ABC-SMC loop")
            break

        print(f"Saving inference results up to round {t + 1}.")

    # return final round results
    return (
        abc_particles[: t+1],
        abc_weights[: t+1],
        abc_round_time[: t+1],
        acc_rate[: t+1],
        obs_summaries[: t+1],
        abc_thresholds[: t+1],
    )


abc_smc = abcsmc_forward