import numpy as np
import math

def random_choice(w: np.ndarray):
    """
    Random choice of an index based on the weights.

    Args:
        w (np.ndarray): Array of weights

    Returns:
        np.int: Sampled index.
    """
    return np.searchsorted(np.cumsum(w), np.random.random())

def sample_prior(prior_bounds: np.ndarray) -> np.ndarray: 
    """
    Sample from the prior distribution 
    Args:
        prior_bounds (np.ndarray): bounds for the prior distribution of the parameters
    Returns:
        np.ndarray: Parameter proposal
    """
    d = len(prior_bounds)
    theta = np.zeros(d)

    for i in range(d):
        theta[i] = np.random.uniform(prior_bounds[i][0], prior_bounds[i][1])

    return theta

def gaussian_proposal(particles: np.ndarray, weights:np.ndarray, chol:np.ndarray, prior_bounds: np.ndarray) -> np.ndarray:
    """
    Randomly pick a particle and perturb it by a Gaussian

    Args:
        particles (np.ndarray): Array of particles.
        weights (np.ndarray): Weights of particles.
        chol (np.ndarray): Cholesky decomposition of a covariance matrix.
        prior_bounds (np.ndarray): Array of prior bounds.
    Returns:
        np.ndarray: Parameter proposal.
    """
    d = len(prior_bounds)
    while True:
        # Resample and perturb.
        theta = chol @ np.random.randn(d) + particles[random_choice(weights)]
        # Check whether theta is within bounds.
        flag = True
        for i in range(d):
            if theta[i] < prior_bounds[i, 0] or theta[i] > prior_bounds[i, 1]:
                flag = False
                break
        if flag:
            return theta

def compute_param_ratio(t: int, prior_bounds: np.ndarray, particles: np.ndarray, weights: np.ndarray, cov_det: np.ndarray, cov_inv: np.ndarray) -> float:
    """
    Compute the ratio of the prior and proosal distribtion for all particles at once

    Args:
        t (int): Round id.
        prior_bounds (np.ndarray): Bounds of the uniform prior distribution.
        particles (np.ndarray): Parameter values.
        weights (np.ndarray): Parameter weights.
        cov_det (np.ndarray): Determinant of the ABC-SMC covariance.
        cov_inv (np.ndarray): Inverse of the ABC-SMC covariance.
    """
    M, d = particles[t].shape

    # Product of uniform priors
    prior_prob = 1
    for j in range(d):
        diff = abs(prior_bounds[j, 1] - prior_bounds[j, 0])
        prior_prob *= 1 / diff
    logprior_prob = np.log(prior_prob)

    # GM prob
    for i in range(M):
        proposal_prob = 0
        for j in range(M):
            kernel_weight = mvnorm_pdf(
                particles[t, i],
                particles[t - 1, j],
                cov_det,
                cov_inv,
            )
            proposal_prob += kernel_weight * weights[t - 1, j]
        weights[t, i] = logprior_prob - np.log(proposal_prob)

def logsumexp(x: np.ndarray) -> float:
    """
    Computes the log of the sum of exponentials of input elements.
        Args:
            x: Array of logs.
        Returns:
            Logarithm of the summed logs.
    """
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

def mvnorm_pdf(x: np.ndarray, mu: np.ndarray, cov_det: np.ndarray, cov_inv: np.ndarray):
    """Hardcoded pdf of the multivarite Gaussian.
    The determinant and the inverse of the covariance are
    precomputed for efficiency.

    Args:
        x (np.ndarray): Parameter value
        mu (np.ndarray): Mean
        cov_det (np.float): Determinant of the covariance
        cov_inv (np.ndarray): Inverse of covariance.

    Returns:
        np.float: PDF evaluated in x.
    """
    d = len(x)

    return math.exp(-(1 / 2) * (x - mu) @ cov_inv @ (x - mu)) / math.sqrt((2 * math.pi) ** d * cov_det)