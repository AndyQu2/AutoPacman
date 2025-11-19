import random
import gymnasium as gym
import numpy as np
import torch
from ReinforcementLearning.deep_q_network import DeepQNetwork
from ReinforcementLearning.replay_buffer import ReplayBuffer

num_epochs = 20
lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 10000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode='human')
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DeepQNetwork(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
agent.load_state_dict(torch.load("output\\gym.pth"))

state, info = env.reset()
done = False
episode_return = 0
while not done:
    env.render()
    action = agent.take_action(state)
    next_state, reward, done, _, _ = env.step(action)
    replay_buffer.add(state, action, reward, next_state, done)
    state = next_state
    episode_return += reward

env.close()