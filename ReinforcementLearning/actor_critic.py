import torch
from torch import nn
from torch.nn import functional as f

from ReinforcementLearning.policy_net import PolicyNet
from ReinforcementLearning.value_net import ValueNet


class ActorCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        super(ActorCritic, self).__init__()
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dist):
        state = torch.tensor(transition_dist['states'], dtype=torch.float).to(self.device)
        action = torch.tensor(transition_dist['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dist['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor(transition_dist['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dist['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        target = rewards + self.gamma * self.critic(next_state) * (1 - dones)
        delta = target - self.critic(state)
        log_probs = torch.log(self.actor(state).gather(1, action))

        actor_loss = torch.mean(-log_probs * delta.detach())
        critic_loss = torch.mean(f.mse_loss(self.critic(state), target.detach()))

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()