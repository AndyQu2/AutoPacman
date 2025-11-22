import numpy as np
from tqdm import tqdm


def collect_episode_data(env, agent, max_steps=1000):
    episode_data = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    state, info = env.reset()
    episode_return = 0
    steps = 0

    for step in range(max_steps):
        action = agent.take_action(state)
        next_state, reward, done, truncated, info = env.step(action)

        episode_data['states'].append(state)
        episode_data['actions'].append(action)
        episode_data['next_states'].append(next_state)
        episode_data['rewards'].append(reward)
        episode_data['dones'].append(done)

        state = next_state
        episode_return += reward
        steps += 1

        if done or truncated:
            break

    return episode_data, episode_return, steps


def train_on_policy_agent(env, agent, num_episodes, num_epoch, num_workers=4):
    return_list = []

    for epoch in range(num_epoch):
        epoch_episodes = int(num_episodes / num_epoch)

        with tqdm(total=epoch_episodes, desc=f'Epoch {epoch + 1}/{num_epoch}') as pbar:
            for episode in range(epoch_episodes):
                episode_data, episode_return, steps = collect_episode_data(env, agent)
                return_list.append(episode_return)

                if len(episode_data['states']) > 0:
                    episode_data = {
                        'states': np.array(episode_data['states'], dtype=np.float32),
                        'actions': np.array(episode_data['actions'], dtype=np.int32),
                        'next_states': np.array(episode_data['next_states'], dtype=np.float32),
                        'rewards': np.array(episode_data['rewards'], dtype=np.float32),
                        'dones': np.array(episode_data['dones'], dtype=np.bool_)
                    }
                    agent.update(episode_data)

                if (episode + 1) % 10 == 0:
                    recent_returns = return_list[-10:] if len(return_list) >= 10 else return_list
                    avg_return = np.mean(recent_returns)
                    pbar.set_postfix({
                        'return': f'{avg_return:.3f}',
                        'steps': steps
                    })

                pbar.update(1)

    return return_list


def moving_average(a, window_size):
    if len(a) < window_size:
        return np.convolve(a, np.ones(window_size) / window_size, mode='valid')

    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size

    if window_size % 2 == 0:
        r = np.arange(1, window_size, 2)
    else:
        r = np.arange(1, window_size - 1, 2)

    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]

    return np.concatenate((begin, middle, end))