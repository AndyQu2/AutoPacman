import random

import gymnasium as gym
import torch
from matplotlib import pyplot as plt

from ReinforcementLearning import utils
from ReinforcementLearning.actor_critic import ActorCritic

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

actor_lr = 1e-3
critic_lr = 1e-3
num_episodes = 1000
num_epochs = 10
hidden_dim = 512
gamma = 0.95
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_name = 'CartPole-v1'
env = gym.make(env_name, render_mode='rgb_array')
random.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)

return_list = utils.train_on_policy_agent(env, agent, num_episodes, num_epochs, 4)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()

mv_return = utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()

env.close()
print("Training finished")
torch.save(agent.state_dict(), "output\\gym.pth")
print("Model is saved to output\\gym.pth")