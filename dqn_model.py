import torch
import torch.nn as nn
import numpy as np
import random
import tqdm

from param import DT, EPSILON, EPSILON_MIN, EPSILON_DECAY, GAMMA, BATCH_SIZE, ACTION_LIST, DQN_ACTION_SCALING

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
    
# Define the epsilon-greedy policy
def epsilon_greedy_policy(model, state, epsilon):
    if random.random() < epsilon:
        return random.choice([0, 1, 2])
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = model(state)
        return torch.argmax(q_values).item()

# Training loop
def train_dqn(env, model, optimizer, memory, episodes, batch_size = BATCH_SIZE, gamma = GAMMA, epsilon = EPSILON, epsilon_min = EPSILON_MIN, epsilon_decay = EPSILON_DECAY, action_list = ACTION_LIST, action_scale = DQN_ACTION_SCALING):
    for episode in tqdm.trange(episodes):
        state = env.reset()
        state = preprocessing(state)
        done = False
        while not done:
            action = epsilon_greedy_policy(model, state, epsilon)
            next_state, reward, done = env.nextState(action_list[action] * action_scale)
            next_state = preprocessing(next_state)
            memory.append((state, action, reward, next_state, done))
            state = next_state

            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                for s, a, r, s_next, d in batch:
                    s = torch.FloatTensor(s).unsqueeze(0)
                    s_next = torch.FloatTensor(s_next).unsqueeze(0)
                    r = torch.FloatTensor([r])
                    a = torch.LongTensor([a])
                    d = torch.FloatTensor([d])

                    q_values = model(s)
                    q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)

                    next_q_values = model(s_next)
                    next_q_value = next_q_values.max(1)[0]
                    expected_q_value = r + gamma * next_q_value

                    loss = nn.MSELoss()(q_value, expected_q_value.detach())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if done:
                epsilon = max(epsilon_min, epsilon * epsilon_decay)

    torch.save(model.state_dict(), 'dqn_model.pt')
    print("Model saved to dqn_model.pt")