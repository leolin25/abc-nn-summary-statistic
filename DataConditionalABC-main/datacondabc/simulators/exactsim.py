import numpy as np
import numba as nb
from math import exp, sqrt
from random import normalvariate


@nb.njit
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


@nb.njit
def cir(x0: float, n: int, dt: float, theta: np.ndarray):
    """Exact sample from the Cox-Ingersoll-Ross model.

    Args:
        x0 (float): Initial state.
        n (int): Number of observations (without x0).
        dt (float): Timestep
        theta (np.ndarray): Parameter.

    Returns:
        np.ndarray: Cox-Ingersoll-Ross trajectory.
    """
    # Extract parameters.
    a, b, s = theta

    # Allocate space for the forward trajectory.
    x = np.zeros(n + 1)
    x[0] = x0

    # Incrementally sample from the transition density.
    for i in range(1, n + 1):
        c = 2 * b / ((1 - exp(-b * dt)) * s**2)
        ncp = 2 * c * x[i - 1] * exp(-b * dt)
        df = 4 * a * b / s**2
        x[i] = np.random.noncentral_chisquare(df, nonc=ncp) / (2 * c)

    return x
