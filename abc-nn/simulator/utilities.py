import numpy as np

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

def gaussian_proposal(particles: np.ndarray, cov: np.ndarray) -> np.ndarray:
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