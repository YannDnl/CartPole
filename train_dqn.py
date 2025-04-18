import torch.optim as optim
import numpy as np
import random
from collections import deque

from environment import CartPole
from dqn_model import DQN, train_dqn
from param import STATE_SIZE, ACTION_SIZE, LEARNING_RATE, MEMORY_SIZE, EPISODES

# Initialize the model, optimizer, and replay buffer
model = DQN(STATE_SIZE, ACTION_SIZE).cuda()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_SIZE)

training_env = CartPole()
train_dqn(training_env, model, optimizer, memory, EPISODES)