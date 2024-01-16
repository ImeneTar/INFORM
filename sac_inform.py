"""
Copyright (c) 2024 ImeneTar

Code adapted from https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy

"""
import os
import random
import time
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import pickle
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Pusher-v4"
    """the environment id of the task"""
    total_timesteps: int = 200000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 1
    """timestep to start learning"""
    policy_lr: float = 5e-6
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    eval_interval: float = 5e3
    """The frequency of evaluation"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    init_temp: float=0.01
    alpha: float = 0.01
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

def get_concat_samples(data_rb, data_expert):
    batch_state = torch.cat([data_rb.observations, data_expert.observations], dim=0)
    batch_next_state = torch.cat(
        [data_rb.next_observations, data_expert.next_observations], dim=0)
    batch_action = torch.cat([data_rb.actions, data_expert.actions], dim=0)
    batch_reward = torch.cat([data_rb.rewards, data_expert.rewards], dim=0)
    batch_done = torch.cat([data_rb.dones, data_expert.dones], dim=0)
    is_expert = torch.cat([torch.zeros_like(data_rb.rewards, dtype=torch.bool),
                           torch.ones_like(data_expert.rewards, dtype=torch.bool)], dim=0)

    return batch_state, batch_next_state, batch_action, batch_reward, batch_done, is_expert



def iq_loss(current_Q, current_v, next_v, batch, actor, critic, alpha2):
    gamma=0.99
    alpha =0.5
    #done = batch.dones
    obs, next_obs, action, env_reward, done, is_expert = batch
    loss_dict = {}


    y = (1 - done) * gamma * next_v
    reward = (current_Q - y)[is_expert]

    loss = - reward.mean()
    loss_dict['softq_loss'] = loss.item()

    value_loss = ((current_v - y)).mean()
    loss += value_loss
    loss_dict['value_loss'] = value_loss.item()




    y = (1 - done) * gamma * next_v
    reward = (current_Q - y)
    chi2_loss = 1 / (4 * alpha) * (reward ** 2).mean()
    loss += chi2_loss
    loss_dict['regularize_loss'] = chi2_loss.item()

    loss_dict['total_loss'] = loss.item()

    return loss, loss_dict


def getV(obs, actor, critic, alpha):
    action, log_prob, _ = actor.get_action(obs)
    current_Q = critic(obs, action)
    current_V = current_Q - alpha * log_prob
    return current_V



def iq_update_critic(data, actor, critic, critic_target, critic_optimizer, alpha):
    obs, next_obs, action = data[0:3]

    current_V = getV(obs, actor, critic, alpha)
    with torch.no_grad():
        next_V = getV(next_obs, actor, critic_target, alpha)

    current_Q = critic(obs, action)
    critic_loss, loss_dict = iq_loss(current_Q, current_V, next_V, data, actor, critic, alpha)

    # Optimize the critic
    critic_optimizer.zero_grad()
    critic_loss.backward()
    # step critic
    critic_optimizer.step()

    return loss_dict



def iq_update(policy_buffer, expert_buffer, step, actor, actor_optimizer, critic, critic_optimiser,critic_target, alpha, args):
    data_rb = policy_buffer.sample(args.batch_size)
    data_expert = expert_buffer.sample(args.batch_size)

    batch = get_concat_samples(data_rb, data_expert)
    obs, next_obs, action = batch[0:3]

    losses = iq_update_critic(batch, actor, critic, critic_target, critic_optimiser, alpha)


    if step % args.policy_frequency == 0:  # TD 3 Delayed update support
        for _ in range(
                args.policy_frequency
        ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
            pi, log_pi, _ = actor.get_action(obs)
            qf_pi = critic(obs, pi)
            actor_loss = ((alpha * log_pi) - qf_pi).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

    if step % args.target_network_frequency == 0:
        for param, target_param in zip(critic.parameters(), critic_target.parameters()):
            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

    return alpha, losses

def load_expert(expert_path):
    f = open(expert_path, 'rb')
    expert_memory_replay = pickle.load(f)
    f.close()
    return expert_memory_replay

def evaluate(actor, args, num_episodes=10):
    total_returns = []
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])

    while len(total_returns) < num_episodes:
        obs, _ = envs.reset(seed=args.seed)
        done = False

        with torch.no_grad():
            while not done:
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.detach().cpu().numpy()

                next_obs, rewards, terminations, truncations, infos = envs.step(actions)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        done = True
                        total_returns.append(abs(info['reward_dist']))

                    obs, _ = envs.reset(seed=args.seed)

                    break

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook

                obs = next_obs

    return total_returns

if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )


    # TRY NOT TO MODIFY: seeding
    #seeds = [0, 1, 2]
    seeds = [8]
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    #
    for run in seeds:
        seed = run #seeds[run]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # writer
        run_name = f"output/pusher/INFORM/Train/{run}"
        writer = SummaryWriter(run_name)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )


        envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        best_eval_returns = +np.inf

        max_action = float(envs.single_action_space.high[0])

        actor = Actor(envs).to(device)
        qf = SoftQNetwork(envs).to(device)
        qf_target = SoftQNetwork(envs).to(device)
        qf_target.load_state_dict(qf.state_dict())

        q_optimizer = optim.Adam(list(qf.parameters()), lr=args.q_lr)
        actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

        # Automatic entropy tuning
        if args.autotune:
            log_alpha = torch.tensor(np.log(args.init_temp)).to(device)
            log_alpha.requires_grad = True
            # target_entropy = args.init_temp
            target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
            # log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha = log_alpha.exp().item()
            a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
        else:
            alpha = args.init_temp

        envs.single_observation_space.dtype = np.float32

        # Load memory
        memory_path = f"output/pusher/TAMER/Train/{run}"
        memory_path1 = os.path.join(memory_path, 'memory_exploration')
        policy_buffer = load_expert(memory_path1)

        memory_path2 = os.path.join(memory_path, 'memory_expert')
        expert_buffer = load_expert(memory_path2)

        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs, _ = envs.reset(seed=args.seed)
        for global_step in range(args.total_timesteps):
            # ALGO LOGIC: put action logic here
            if global_step < args.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.detach().cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            if global_step == 70000:
                torch.save(actor.state_dict(), run_name + '/actor_model_70.pth')
                torch.save(qf.state_dict(), run_name + '/qf_model_70.pth')

            if global_step == 100000:
                torch.save(actor.state_dict(), run_name + '/actor_model_100.pth')
                torch.save(qf.state_dict(), run_name + '/qf_model_100.pth')

            if global_step == 150000:
                torch.save(actor.state_dict(), run_name + '/actor_model_150.pth')
                torch.save(qf.state_dict(), run_name + '/qf_model_150.pth')

            if global_step == 180000:
                torch.save(actor.state_dict(), run_name + '/actor_model_180.pth')
                torch.save(qf.state_dict(), run_name + '/qf_model_180.pth')

            if global_step % args.eval_interval == 0:
                eval_returns = evaluate(actor, args)
                returns = np.mean(eval_returns)

                if returns < best_eval_returns:
                    # Store best eval returns
                    print("best")
                    best_eval_returns = returns
                    torch.save(actor.state_dict(), run_name + '/actor_model_best.pth')
                    torch.save(qf.state_dict(), run_name + '/qf_model_best.pth')

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                for info in infos["final_info"]:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}, distance={info['reward_dist']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

                    obs, _ = envs.reset(seed=args.seed)
                    break

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]


            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                alpha, loss = iq_update(policy_buffer,expert_buffer, global_step, actor,actor_optimizer, qf, q_optimizer, qf_target, alpha, args)

        # save model
        torch.save(actor.state_dict(), run_name + '/actor_model.pth')
        torch.save(qf.state_dict(), run_name + '/qf_model.pth')

        envs.close()
        writer.close()
