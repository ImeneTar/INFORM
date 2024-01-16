
import os
import random
import time
from dataclasses import dataclass
import pickle
from Agents.Qlearning import QN
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from utils import flat_to_one_hot
from mdp_policy import gridworld, gridworld_wall



@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    test_setting: str = "wall"
    """testing environment (start or wall)"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "wall"
    """the id of the environment"""
    total_timesteps: int = 50000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3 #2e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 1
    """the frequency of training"""


def make_env(env_id):
    if env_id == "grid":
        env = gridworld()
    elif env_id == "wall":
        env = gridworld_wall()
    return env



# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear( np.array(env.nb_states).prod(), 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, env.nb_actions),
        )

    def forward(self, x):
        return self.network(x)



def evaluate_start(agent):
    # env setup
    envs = make_env("grid")

    episode_length = []
    starts = [7, 8, 9, 12, 15, 16, 17, 20, 21, 22]

    for start in starts:
        obs = start
        done = False
        t = 0

        while not done and t != 200:
            action, action_value = agent.select_action(obs)
            next_obs, reward, done, info = envs.step(obs, action, t)
            obs = next_obs
            t += 1
        episode_length.append(t)

    return episode_length

def evaluate_wall(agent):
    # env setup
    envs = make_env("wall")

    episode_length = []

    for i in range(100):
        obs = envs.reset()
        done = False
        t = 0

        while not done and t != 200:
            action, action_value = agent.select_action(obs)
            next_obs, reward, done, info = envs.step(obs, action, t)
            obs = next_obs
            t += 1
        episode_length.append(t)

    return episode_length




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

    seeds = [0, 1, 2, 4, 6]
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    length_runs = []

    for run in range(len(seeds)):
        print("RUN", run)
        seed = seeds[run]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # writer
        run_name = f"output/grid/QL/Train/Env/{run}"
        writer = SummaryWriter(run_name)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        # env setup
        envs = make_env(args.env_id)
        #assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

        Ql = QN(envs, envs.r, timeout=200, nEpisodes=300, alpha=0.2)
        global_step = 0

        #for global_step in range(args.total_timesteps):
        for epoch in range(2000):
            t = 0
            reward_sum = 0
            obs = 4
            if obs == envs.goal:
                done = True
            else:
                done = False
            while not done and t != Ql.timeout:
                action, action_value = Ql.select_action(obs)
                next_obs, reward, done, info = envs.step(obs, action, t)

                # if done:
                #     Ql.w[action][obs] += Ql.alpha * (reward - action_value)
                # else:
                _, new_action_value = Ql.select_action(next_obs)
                Ql.w[action][obs] += Ql.alpha * (reward + (1-done) * Ql.gamma * new_action_value - action_value)

                obs = next_obs
                t += 1
                global_step +=1



            print(f"global_step={global_step}, episodic_return={t}")
            writer.add_scalar("charts/episodic_return", t, global_step)

        #Evaluation
        # if args.test_setting == "start":
        #     episode_length = evaluate_start(Ql)
        # else:
        #     episode_length = evaluate_wall(Ql)
        #
        # length_runs.append(episode_length)

    # save list
    # path = f"output/grid/QL/Test/{args.test_setting}"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    # f = open(path + "/length", "wb")
    # pickle.dump(length_runs, f)
    # f.close()








