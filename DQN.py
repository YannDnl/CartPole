import numpy as np
from param import DT

def preprocessing(state: tuple, memory: float) -> tuple:
    """
    Preprocess the state and reward to be used as input for the DQN.

    Args:
        state(tuple): A tuple representing the current state (x, old_x, theta, old_theta).
        reward(float): A float representing the reward.

    Returns:
        tuple: A tuple representing the preprocessed state and reward.
    """
    x, old_x, theta, old_theta = state
    theta = theta % (2 * np.pi)  # Normalize theta to be within [0, 2*pi]
    state = x, (x - old_x)/DT, theta, (theta - old_theta)/DT
    reward = np.sin(theta)  # Reward is based on the angle of the pole
    return state, reward  # Return only the relevant state variables for DQN