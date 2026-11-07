from random import seed
import numpy as np
import matplotlib.pyplot as plt

def ou_exact(h: float, T: float, mu: float, sigma: float, tau: float, x0: float, n_paths: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates an Ornstein-Uhlenbeck process using the exact solution
    Args:
        h (float): step size
        T (float): total time
        mu (float): drift coefficient
        sigma (float): diffusion coefficient
        tau (float): mean reversion speed
        x0 (float): initial condition
        n_paths (int): number of trajectories to simulate
    Returns:
        tuple[np.ndarray, np.ndarray]: simulated sequences and time points
    """
    n_steps = int(T / h)
    t = np.linspace(0, T, n_steps + 1)

    # Initialize a 2D array: rows being time steps and columns being trajectories
    X = np.zeros((n_steps + 1, n_paths))
    X[0, :] = x0 # initial condition for all paths

    theta = 1.0 / tau # mean reversion rate

    # calculate constants
    exp_term = np.exp(- theta * h)
    var = (sigma**2 / (2 * theta)) * (1 - exp_term ** 2)

    # generate all random standard normal variables for all paths at once
    Z = np.random.normal(0, 1, (n_steps, n_paths))

    for i in range(n_steps):
        X[i+1, :] = X[i, :] * exp_term + mu * (1 - exp_term) + np.sqrt(var) * Z[i, :]
    
    return X, t
