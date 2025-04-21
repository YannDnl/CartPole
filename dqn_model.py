import torch
import torch.nn as nn
import numpy as np
import random
import tqdm

from param import DT

# Model inputs preprocessing
def preprocessing(state: tuple) -> tuple:
    x, old_x, theta, old_theta = state
    theta = theta % (2 * np.pi)  # Normalize theta to be within [0, 2*pi]
    state = x, (x - old_x)/DT, theta, (theta - old_theta)/DT
    return state

# Define the Neural network model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x