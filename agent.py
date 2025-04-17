import numpy as np
from param import PROPORTIONAL_CORRECTION, INTEGRAL_CORRECTION, DERIVATIVE_CORRECTION, DT

def getAction(state: float, memory: tuple) -> tuple:
    """
    Given a reward, return the action.
    
    Args:
        reward(float): A float representing the reward.
        memory(tuple): A tuple representing the memory (old_reward, reward_integral).
    
    Returns:
        A float representing the action.
    """
    return getActionControlPID(state, memory)

def getActionControlPID(state: float, memory: tuple) -> tuple:
    """
    Given a reward, return the action using PID control.
    
    Args:
        reward(float): A float representing the reward.
        memory(tuple): A tuple representing the memory (old_reward, reward_integral).
    
    Returns:
        A float representing the action.
    """
    reward = np.cos(state[2])
    old_reward, reward_integral = memory
    reward_integral += reward * DT
    reward_derivative = (reward - old_reward) / DT
    action = reward * PROPORTIONAL_CORRECTION + reward_integral * INTEGRAL_CORRECTION + reward_derivative * DERIVATIVE_CORRECTION
    memory = reward, reward_integral
    return action, memory