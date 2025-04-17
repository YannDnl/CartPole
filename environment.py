import numpy as np
from param import R, M, m, G, DT

def initialState() -> tuple:
    """
    Returns the initial state of the cart-pole system.

    Returns:
        tuple: A tuple representing the initial state (x, old_x, theta, old_theta).
    """
    state = (0, 0, 1, 1)
    return state, getReward(state)

def nextState(state: tuple, action: float) -> tuple:
    """
    Given a state and an action, return the next state.

    Args:
        state(tuple): A tuple representing the current state (x, old_x, theta, old_theta).
        action(float): A float representing the force exerted by the cart to move.

    Returns:
        tuple: A tuple representing the next state and the associated reward ((x, old_x, theta, old_theta), reward).
    """
    x, old_x, theta, old_theta = state
    new_theta = 2 * theta - old_theta + (action * np.sin(theta)/m - G * np.cos(theta)) * DT**2/R
    dthetadt = (new_theta - theta) / DT
    a = (action + np.cos(theta) * (M * R * dthetadt**2 - G * M * np.sin(theta)))/m
    new_x = x + (x - old_x) * DT + a * DT**2/2
    new_state = (new_x, x, new_theta, theta)
    return new_state, getReward(new_state)
    
def getReward(state: tuple) -> float:
    """
    Given a state, return the reward.

    Args:
        state(tuple): A tuple representing the current state (x, old_x, theta, old_theta).

    Returns:
        float: A float representing the reward.
    """
    x, _, theta, _ = state
    return np.cos(theta)