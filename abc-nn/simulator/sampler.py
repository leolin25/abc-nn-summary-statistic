import numpy as np
import scipy.stats as stats
from functools import partial
from utilities import sample_prior

def abcsmc_forward(obs: np.ndarray, prior_bounds: np.ndarray, simulator: callable, n_part: int, max_iter: int, init_size: int, q: float):
    """
    Perform Approximate Bayesian Computation with Sequential Monte Carlo (ABC-SMC) loop.

    This function performs parameter inference by simulating data, dynamically training a neural network to extract summary statistics, and progressively tightening the acceptance threshold over multiple rounds.
    Args:
        obs (np.ndarray): observed data time-series data
        prior_bounds (np.ndarray): bounds for the prior distribution of the parameters
        simulator (function): a function that simulates a time-series
        n_part (int): number of particles to use in the ABC-SMC algorithm
        max_iter (int): maximum number of iterations for the ABC-SMC algorithm
        init_size (int): initial size of the neural network training dataset
        q (float): quantile for the ABC-SMC algorithm
    Returns:
        tuple: A tuple containing the following arrays up to the final completed round (t+1):
            - abc_particles (np.ndarray): The accepted parameter vectors forming the posterior.
            - abc_weights (np.ndarray): The calculated Sequential Monte Carlo importance weights.
            - abcround_time (np.ndarray): The computation time (in seconds) spent simulating and sampling per round.
            - training_time (np.ndarray): The computation time (in seconds) spent training the neural network.
            - acc_rates (np.ndarray): The percentage of proposed simulations accepted in each round.
            - obssummaries (torch.Tensor): The neural network's evolving summary statistic representation of the observed data.
            - abc_thresholds (np.ndarray): The epsilon distance thresholds enforced at each round.
    """

    def _summarise(x: np.ndarray) -> np.ndarray:
        """
        Function to summarise a given trajectory
        Args:
            x (np.ndarray): simulated trajectory data
        Returns:
            np.ndarray: The extracted summary statistics.
        """
        # for current implementation we will use canonical summary statistics
        mean = np.mean(x)
        var = np.var(x)
        
        return np.array([mean, var])

    num_sum_stats = 2  # number of summary statistics (mean and variance)

    # particle system, covariance matrix and thresholds
    param_dim = len(prior_bounds)
    abc_particles = np.zeros((max_iter, n_part, param_dim))
    abc_weights = np.zeros((max_iter, n_part))
    abc_cov = np.zeros((param_dim, param_dim))
    abc_thresholds = np.zeros(max_iter)

    obssummaries = np.zeros((max_iter, num_sum_stats))

    # initialize abc-smc
    abc_thresholds[0] = np.inf
    abc_weights[0] = np.log(1 / n_part) # uniform weights for this round
    obssummaries[0] = _summarise(obs)

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

    return None

