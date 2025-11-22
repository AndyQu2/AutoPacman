import gymnasium as gym
import torch

from ReinforcementLearning.actor_critic import ActorCritic

actor_lr = 1e-3
critic_lr = 1e-3
num_episodes = 2000
hidden_dim = 128
gamma = 0.98
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_name = 'CartPole-v1'
env = gym.make(env_name, render_mode='human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device)
agent.load_state_dict(torch.load("output\\gym.pth"))

state, info = env.reset()
done = False
while not done:
    env.render()
    action = agent.take_action(state)
    state, reward, done, _, _ = env.step(action)

env.close()