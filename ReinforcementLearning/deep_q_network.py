import numpy as np
import torch
from torch import optim
from ReinforcementLearning.q_network import QNetwork
from torch.nn import functional as f

class DeepQNetwork:
    def __init__(self, state_dim, hidden_dim, action_dim,
                 learning_rate, gamma, epsilon, target_update, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device
        self.q_network = QNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.count = 0

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_network(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_network(states).gather(1, actions)
        max_next_q_values = self.target_network(next_states).max(1)[0].view(-1, 1)
        q_target = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(f.mse_loss(q_values, q_target))

        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        self.count += 1