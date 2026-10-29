from random import seed
import numpy as np
import matplotlib.pyplot as plt

def ou():
    """
    Simulates an Ornstein-Uhlenbeck process using the exact solution
    Args:
            h (float): step size
            T (float): total time
            mu (float): drift coefficient
            sigma (float): diffusion coefficient
            tau (float): mean reversion speed
            x0 (float): initial condition
        Returns:
            tuple[np.ndarray, np.ndarray]: simulated sequence and time points
    """