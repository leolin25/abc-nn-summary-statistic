from random import seed

import numpy as np

def sde_simulation():
    """
    Simulates a stochastic differential equation (SDE) 
    """
    return 

def simulate_markov(transition_matrix, initial_state, num_steps):
    """
    Simulates a Markov chain given a transition matrix, an initial state and the number of steps to simulate
    params:
        transition_matrix: A square matrix representing the transition probabilities between states
        initial_state: The initial state of the Markov chain
        num_steps: The number of steps to simulate
    returns:
        np.ndarray: An array of states visited during the simulation
    """
    num_states = transition_matrix.shape[0]

    # allocate sequence array
    sequence = np.zeros(num_steps, dtype=int)
    sequence[0] = initial_state

    # states the markov chain can go
    state_choices = np.arange(num_states)
    current_state = initial_state

    for i in range(1, num_steps):
        probabilities = transition_matrix[current_state]

        next_state = np.random.choice(state_choices, p=probabilities)

        sequence[i] = next_state
        current_state = next_state

    return sequence

# set a seed for reproducibility
np.random.seed(42)

# example execution
P = np.array([[0.2, 0.7, 0.1], [0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
init_state = 0
num_steps = 10

simulated_sequence = simulate_markov(P, init_state, num_steps)
print(simulated_sequence)