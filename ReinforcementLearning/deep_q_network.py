import numpy as np
import torch
from torch import optim
from ReinforcementLearning.q_network import QNetwork

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

