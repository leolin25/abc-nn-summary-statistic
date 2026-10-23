from random import seed
import numpy as np
import matplotlib.pyplot as plt

def ou_simulation(h: float, T: float, mu:float, sigma:float, tau:float, x0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulates an Ornstein-Uhlenbeck process using Euler-Maruyama method
    """
    n_steps = int(T / h)
    t = np.linspace(0, T, n_steps + 1)
    X = np.zeros(n_steps + 1)
    X[0] = x0 # initial condition

    for i in range(n_steps):
        drift = (1 / tau) * (mu - X[i]) * h
        diffusion = sigma * np.sqrt(h) * np.random.normal(0, 1)
        X[i+1] = X[i] + drift + diffusion

    return X, t

# set a seed for reproducibility
np.random.seed(42)

h = 0.01 # step size
T = 10.0 # total time
mu, sigma = 0.1, 0.2 # drift and diffusion coefficients
tau = 0.5 # mean reversion speed
x0 = 0.0 # initial condition

simulated_sequence, t = ou_simulation(h, T, mu, sigma, tau, x0)

# create a plot of the simulated Ornstein-Uhlenbeck process
plt.figure(figsize=(10, 7))
plt.plot(t, simulated_sequence)
plt.title('Ornstein-Uhlenbeck Process Simulation')
plt.xlabel('Time')
plt.ylabel('X(t)')
plt.show()