
import os
import random
import time
from dataclasses import dataclass
import pickle
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

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
    total_timesteps: int = 50000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
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

    seeds = [0, 1 ,5]
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    length_runs = []
    x_runs = []
    y_runs = []
    success_runs = []
    for run in seeds:
        seed = run #seeds[run]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # env setup
        envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        max_action = float(envs.single_action_space.high[0])

        #load model
        actor = Actor(envs).to(device)

        actor_path = f"output/pusher/TAMER/Train/{run}/actor_model.pth"
        actor.load_state_dict(torch.load(actor_path, map_location=device))


        alpha = args.alpha
        x_final = []
        y_final = []
        success_final = []
        total = 100


        envs.single_observation_space.dtype = np.float32
        t_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs, _ = envs.reset(seed=args.seed)
        done = False


        #get goal position
        x_goal = obs[0][20]
        y_goal = obs[0][21]

        global_step = 0
        episode_length = []
        for episode in range(total):
            x_cord = []
            y_cord = []
            success = []
            while not done:
                x_cord.append(obs[0][17])
                y_cord.append(obs[0][18])
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.detach().cpu().numpy()

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards, terminations, truncations, infos = envs.step(actions)

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                if "final_info" in infos:
                    for info in infos["final_info"]:
                        episode_length.append(abs(info['reward_dist']))
                        if abs(info['reward_dist']) < 0.08:
                            success.append(1)
                        else:
                            success.append(0)
                    obs, _ = envs.reset(seed=args.seed)

                    break

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook

                obs = next_obs
                global_step += 1

            success_final.append(success)
            x_final.append(x_cord)
            y_final.append(y_cord)

        success_runs.append(success_final)
        x_runs.append(x_final)
        y_runs.append(y_final)
        length_runs.append(episode_length)

    # save list
    path = f"output/pusher/TAMER/Test_original"
    if not os.path.exists(path):
        os.makedirs(path)

    f = open(path + "/distance", "wb")
    pickle.dump(length_runs, f)
    f.close()

    f = open(path + "/success", "wb")
    pickle.dump(success_runs, f)
    f.close()

    f = open(path + "/x_cord", "wb")
    pickle.dump(x_runs, f)
    f.close()

    f = open(path + "/y_cord", "wb")
    pickle.dump(y_runs, f)
    f.close()


    # fig, ax = plt.subplots()
    #
    # for i in range(total):
    # # Plot the trajectories
    #     ax.plot(x_final[i], y_final[i], color='orange')
    #
    # # Plot the goal position as a red point
    # #ax.plot(x_goal, y_goal, label='Goal Position', color='red', marker='o', markersize=10)
    # circle = Circle((x_goal, y_goal), 0.08, color='blue', fill=True, alpha=0.2)  # Adjust color and alpha as needed
    # plt.gca().add_patch(circle)
    #
    # # Adding labels and title
    # ax.set_xlabel('X Position')
    # ax.set_ylabel('Y Position')
    # ax.set_title('Object Trajectories and Goal Position in Pusher Environment')
    # ax.legend()
    #
    # # Show the plot
    # plt.show()
    # envs.close()
