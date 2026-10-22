from random import seed
import numpy as np
import matplotlib.pyplot as plt

def sde_simulation(h: float, T: float, mu:float, sigma:float, x0: float) -> np.ndarray:
    """
    Simulates a stochastic differential equation (SDE) using Euler-Maruyama method
    """
    n_steps = int(T / h)
    t = np.linspace(0, T, n_steps + 1)
    x = np.zeros(n_steps + 1)
    x[0] = 0  # initial condition

    for i in range(n_steps):
        W = np.random.normal(0, np.sqrt(h))  # Brownian motion increment
        x[i + 1] = x[i] + mu * h + sigma * W

    return x

# set a seed for reproducibility
np.random.seed(42)

# example execution
h = 0.01 # step size
T = 1.0 # total time
mu, sigma = 0.1, 0.2 # drift and diffusion coefficients
x0 = 0.0 # initial condition

simulated_sequence = sde_simulation(h, T, mu, sigma, x0)
print(simulated_sequence)