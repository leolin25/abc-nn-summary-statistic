import numpy as np
import matplotlib.pyplot as plt

def ou_eu(h: float, T: float, mu: float, sigma: float, tau: float, x0: float, n_paths: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates an Ornstein-Uhlenbeck process using Euler-Maruyama method for multiple paths
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
    
    X = np.zeros((n_steps + 1, n_paths))
    X[0, :] = x0 

    Z = np.random.normal(0, 1, (n_steps, n_paths))

    for i in range(n_steps):
        # Calculate drift and diffusion for all paths simultaneously
        drift = (1 / tau) * (mu - X[i, :]) * h
        diffusion = sigma * np.sqrt(h) * Z[i, :]
        
        X[i+1, :] = X[i, :] + drift + diffusion

    return X, t

# set a seed for reproducibility
np.random.seed(42)

h = 0.01 # step size
T = 10.0 # total time
mu, sigma = 0.1, 0.2 # drift and diffusion coefficients
tau = 0.5 # mean reversion speed
x0 = 0.0 # initial condition
n_paths = 50 # number of trajectories

simulated_sequences, t = ou_eu(h, T, mu, sigma, tau, x0, n_paths)

# create a plot of the simulated Ornstein-Uhlenbeck process
plt.figure(figsize=(12, 7))

# Plot all paths with low opacity so we can see the density distribution
plt.plot(t, simulated_sequences, color='green', alpha=0.5, linewidth=0.5)

# Plot the theoretical mean as a dashed line for reference
plt.axhline(mu, color='black', linestyle='--', linewidth=2, label='Long-term Mean ($\mu$)')

plt.title(f'{n_paths} Euler-Maruyama Simulations of an Ornstein-Uhlenbeck Process')
plt.xlabel('Time')
plt.ylabel('X(t)')
plt.legend()
plt.show()