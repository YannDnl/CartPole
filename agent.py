import numpy as np
import torch
from param import PROPORTIONAL_CORRECTION, INTEGRAL_CORRECTION, DERIVATIVE_CORRECTION, MODEL_PATH, DQN_ACTION_SCALING, ACTION_LIST, STATE_SIZE, ACTION_SIZE, DT

from dqn_model import DQN, preprocessing

class Agent:
    def __init__(self):
        pass

    def getAction(self, state: float) -> float:
        return None

class PIDAgent(Agent):
    def __init__(self):
        self.memory = (0, 0)

    def getMemory(self) -> tuple:
        return self.memory
    
    def setMemory(self, memory: tuple) -> None:
        self.memory = memory

    def getAction(self, state: float) -> float:
        reward = (np.pi/2 - state[2] + np.pi/2) % (2 * np.pi) - np.pi/2 #Maintaining error periodicity is crucial
        old_reward, reward_integral = self.getMemory()
        reward_integral += reward * DT
        reward_derivative = (reward - old_reward) / DT
        action = reward * PROPORTIONAL_CORRECTION + reward_integral * INTEGRAL_CORRECTION + reward_derivative * DERIVATIVE_CORRECTION
        self.setMemory((reward, reward_integral))
        return action

class RLAgent(Agent):
    def __init__(self):
        model = DQN(STATE_SIZE, ACTION_SIZE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        self.model = model

    def getModel(self) -> torch.nn.Module:
        return self.model

    def getAction(self, state: float) -> float:
        state = preprocessing(state)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.getModel()(state)
        action = ACTION_LIST[torch.argmax(q_values).item()] * DQN_ACTION_SCALING
        return action