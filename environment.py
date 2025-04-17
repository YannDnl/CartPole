import numpy as np
import random as rd
from param import R, M, m, G, DT, DURATION

class CartPole:
    def __init__(self):
        angle = rd.random() * 2 * np.pi
        self.state: tuple = (0, 0, angle, angle)
        self.steps: int = 0
    
    def getState(self) -> tuple:
        return self.state
    
    def setState(self, state: tuple) -> None:
        self.state = state
    
    def getSteps(self) -> int:
        return self.steps
    
    def incrementSteps(self) -> None:
        self.steps += 1
    
    def getReward(self) -> float:
        _, _, theta, _ = self.getState()
        return np.sin(theta)

    def nextState(self, action: float) -> tuple:
        x, old_x, theta, old_theta = self.getState()

        new_theta = 2 * theta - old_theta + (action * np.sin(theta)/m - G * np.cos(theta)) * DT**2/R
        dthetadt = (new_theta - theta) / DT
        a = (action + np.cos(theta) * (M * R * dthetadt**2 - G * M * np.sin(theta)))/m
        new_x = x + (x - old_x) * DT + a * DT**2/2
        new_state = (new_x, x, new_theta, theta)

        self.setState(new_state)

        if self.getSteps() > DURATION:
            done = True
        else:
            done = False
        self.incrementSteps()

        return self.getState(), self.getReward(), done

    def reset(self) -> tuple:
        angle = rd.random() * 2 * np.pi
        self.setState((0, 0, angle, angle))
        self.steps = 0
        return self.getState()