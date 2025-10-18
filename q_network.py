import copy
import numpy as np
import torch
from torch import nn, optim
from replay_buffer import ReplayBufferDiscreteAction

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class DeepQNetwork:
    def __init__(self, agent_name, device, gamma, epsilon, update_frequency, batch_size,
                 state_dim, action_dim, hidden_dim, learning_rate):
        self.agent_name = agent_name
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_count = 0
        self.update_frequency = update_frequency
        self.batch_size = batch_size
        self.memory = ReplayBufferDiscreteAction(capacity=self.batch_size)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.policy_net = QNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def sample_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0)
            if np.random.uniform() > self.epsilon or deterministic:
                action = self.policy_net(state).argmax(dim=-1).item()
            else:
                action = np.random.randint(0, self.action_dim)
            return action