import torch
import numpy as np
import gym
import pickle
import torch
#from scipy.stats import spearmanr, pearsonr
import torch.nn as nn
from mdp_policy import gridworld, gridworld_wall

def flat_to_one_hot(val, ndim):
    shape =np.array(val).shape
    v = np.zeros(shape + (ndim,))
    if len(shape) == 1:
        v[np.arange(shape[0]), val] = 1.0
    else:
        v[val] = 1.0
    return v

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

# def evaluate(agent, env, num_episodes):
#     total_timesteps = []
#     total_returns = []
#
#     for _ in range(num_episodes):
#         state = env.reset()
#         done = False
#         t = 0
#         r = 0
#
#         while not done and t != 200:
#             action = agent.choose_action(state)
#             next_state, reward, done, info, _ = env.step(action)
#             state = next_state
#             r += reward
#             t += 1
#
#         total_returns.append(r)
#         total_timesteps.append(t)
#
#     return total_returns, total_timesteps

def hard_update(source, target):
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(param.data)

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def get_concat_samples(policy_batch, expert_batch):
    online_batch_state, online_batch_next_state, online_batch_action, online_batch_reward, online_batch_done = policy_batch

    expert_batch_state, expert_batch_next_state, expert_batch_action, expert_batch_reward, expert_batch_done = expert_batch

    batch_state = torch.cat([online_batch_state, expert_batch_state], dim=0)
    batch_next_state = torch.cat(
        [online_batch_next_state, expert_batch_next_state], dim=0)
    batch_action = torch.cat([online_batch_action, expert_batch_action], dim=0)
    batch_reward = torch.cat([online_batch_reward, expert_batch_reward], dim=0)
    batch_done = torch.cat([online_batch_done, expert_batch_done], dim=0)
    is_expert = torch.cat([torch.zeros_like(online_batch_reward, dtype=torch.bool),
                           torch.ones_like(expert_batch_reward, dtype=torch.bool)], dim=0)

    return batch_state, batch_next_state, batch_action, batch_reward, batch_done, is_expert

def save_best(agent, memory, path):
    agent.save(f'{path}/best')
    memory.save(f'{path}/best')

def save_best_offline(agent, path):
    agent.save(f'{path}/best')

def save_run(agent, memory, path):
    agent.save(f'{path}/end')
    memory.save(f'{path}/end')

def save_run_offline(agent, path):
    agent.save(f'{path}/end')


def load_expert(expert_path):
    f = open(expert_path, 'rb')
    expert_memory_replay = pickle.load(f)
    f.close()
    return expert_memory_replay


def evaluate_start(agent,logger):
    env = gridworld()
    total_timesteps = []
    total_returns = []
    starts = [7,8,9,12,15,16,17,20,21,22]
    i=0
    for start in starts:
        state = start
        done = False
        t = 0
        r = 0

        while not done and t != 200:
            state_vec = flat_to_one_hot(state, env.nb_states)
            action = agent.choose_action(state_vec)
            reward, next_state, done = env.step(state, action)
            state = next_state
            r += reward
            t += 1

        total_returns.append(r)
        total_timesteps.append(t)
        logger.log('test/episode_reward', t, i)
        logger.dump(i, ty='test')
        i+=1

    return total_returns, total_timesteps

def evaluate_wall(agent, logger):
    total_timesteps = []
    total_returns = []

    env = gridworld_wall()
    i=0
    for _ in range(500):
        state = 4
        done = False
        t = 0
        r = 0

        while not done and t != 200:
            state_vec = flat_to_one_hot(state, env.nb_states)
            action = agent.choose_action(state_vec)
            reward, next_state, done = env.step(state, action)
            state = next_state
            r += reward
            t += 1

        total_returns.append(r)
        total_timesteps.append(t)
        logger.log('test/episode_reward', t, i)
        logger.dump(i, ty='test')
        i+=1

    return total_returns, total_timesteps

def  RL_evaluate_start(agent, env, logger):
    total_timesteps = []
    total_returns = []
    starts = [7,8,9,12,15,16,17,20,21,22]
    i=0
    for start in starts:
        x = start
        done = False
        t = 0
        r = 0

        while not done and t != 200:
            u, reward, y, done = agent.test(env, x)
            x = y
            r += reward
            t += 1

        total_returns.append(r)
        total_timesteps.append(t)
        logger.log('test/episode_reward', t, i)
        logger.dump(i, ty='test')
        i+=1

    return total_returns, total_timesteps

def  RL_evaluate_wall(agent, env, logger):
    total_timesteps = []
    total_returns = []
    i=0
    for _ in range(1000):
        x = 4
        done = False
        t = 0
        r = 0
        while not done and t != 200:
            u, reward, y, done = agent.test(env, x)
            x = y
            r += reward
            t += 1
        total_returns.append(r)
        total_timesteps.append(t)
        logger.log('test/episode_reward', t, i)
        logger.dump(i, ty='test')
        i+=1
    return total_returns, total_timesteps

def eps(rewards):
    return [sum(x) for x in rewards]

def part_eps(rewards):
    return [np.cumsum(x) for x in rewards]

# def measure_correlation(agent, env): ##need to add logger
#     env_rewards = []
#     learnt_rewards = []
#     env_r = []
#     learnt_r =[]
#
#     for ep in range(100):
#         part_env_rewards = []
#         part_learnt_rewards = []
#
#         state = env.reset()
#         done = False
#         t = 0
#         episode_reward = 0
#         episode_irl_reward = 0
#
#         while not done and t != 200:
#             action = agent.choose_action(state)
#             next_state, reward, done, info, _ = env.step(action)
#
#             #Get Inverse reward
#             with torch.no_grad():
#                 q = agent.infer_q(state, action)
#                 next_v = agent.infer_v(next_state)
#                 y = (1 - done) * agent.gamma * next_v
#                 irl_reward = (q - y)
#
#             episode_irl_reward += irl_reward.item()
#             episode_reward += reward
#             part_learnt_rewards.append(irl_reward.item())
#             part_env_rewards.append(reward)
#
#             state = next_state
#             t += 1
#
#         env_r.append(episode_reward)
#         learnt_r.append(episode_irl_reward)
#
#
#
#         #print('Ep {}\tEpisode env rewards: {:.2f}\t'.format(ep, episode_reward))
#         #print('Ep {}\tEpisode learnt rewards {:.2f}\t'.format(ep, episode_irl_reward))
#
#         learnt_rewards.append(part_learnt_rewards)
#         env_rewards.append(part_env_rewards)
#
#
#
#     mean_env_r = np.array(env_r).mean()
#     mean_learnt_r = np.array(learnt_r).mean()
#
#
#    # print(f'Spearman correlation {spearmanr(eps(learnt_rewards), eps(env_rewards))}')
#     #print(f'Pearson correlation: {pearsonr(eps(learnt_rewards), eps(env_rewards))}')
#
#     return pearsonr(eps(learnt_rewards), eps(env_rewards)), mean_env_r, mean_learnt_r
#
