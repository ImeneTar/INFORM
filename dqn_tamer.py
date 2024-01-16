# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
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
    env_id: str = "grid"
    """the id of the environment"""
    total_timesteps: int = 200000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3 #2e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0
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
    exploration_fraction: float = 0.2
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 1000
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


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def load_oracle(path):
    f = open(path, 'rb')
    oracle = pickle.load(f)
    f.close()
    return oracle
def get_feedback(oracle, state, action, env, t):
    alpha = 0.999
    a_opt, _ =  oracle.select_action(state)
    if oracle.w[action][state] > alpha * oracle.w[a_opt][state]:
        feedback = 1
    else:
        feedback = 0
    return feedback

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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )


    # TRY NOT TO MODIFY: seeding
    seeds = [0, 1, 2, 4, 6]

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # env setup

    #assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    #load grid
    oracle = load_oracle("oracles/grid")

    for run in range(len(seeds)):
        seed = seeds[run]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        #writer
        run_name = f"output/grid/TAMER/Train/{run}"

        writer = SummaryWriter(run_name)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        envs = make_env(args.env_id)

        q_network = QNetwork(envs).to(device)
        optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
        target_network = QNetwork(envs).to(device)
        target_network.load_state_dict(q_network.state_dict())

        rb = ReplayBuffer(
            args.buffer_size,
            envs.obs_space,
            envs.action_space,
            device,
            handle_timeout_termination=False,
        )

        rb_success = ReplayBuffer(
            args.buffer_size,
            envs.obs_space,
            envs.action_space,
            device,
            handle_timeout_termination=False,
        )

        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs = envs.reset()
        dones = False
        step = 0
        traj = []
        for global_step in range(args.total_timesteps):

            obs_vector = flat_to_one_hot(obs, envs.nb_states)
            # ALGO LOGIC: put action logic here
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
            if random.random() < epsilon:
                actions = np.array(envs.get_random_action()) #np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                q_values = q_network(torch.Tensor(obs_vector).to(device))
                actions = torch.argmax(q_values, dim=-1).cpu().numpy()

            #print("Obs:", obs, "action", actions)
            # TRY NOT TO MODIFY: execute the game and log data.
            #next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            next_obs, rewards, dones, infos = envs.step(obs, actions, step)
            feedback = get_feedback(oracle, obs, actions, envs, global_step)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if dones:
                print(f"global_step={global_step}, episodic_return={step}")
                writer.add_scalar("charts/episodic_return", step, global_step)
                #reset env
                rb.add(flat_to_one_hot(obs, envs.nb_states), flat_to_one_hot(next_obs, envs.nb_states), actions, feedback, dones, infos)
                traj.append([flat_to_one_hot(obs, envs.nb_states), flat_to_one_hot(next_obs, envs.nb_states), actions, feedback, dones, infos])

                if step < 30:
                    for i in range(len(traj)):
                        rb_success.add(traj[i][0], traj[i][1], traj[i][2], traj[i][3], traj[i][4], traj[i][5])

                obs = envs.reset()
                dones = False
                step = 0
                traj = []


            else:
                # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
                real_next_obs = next_obs.copy()
                rb.add(flat_to_one_hot(obs, envs.nb_states), flat_to_one_hot(real_next_obs, envs.nb_states), actions, feedback, dones, infos)
                traj.append([flat_to_one_hot(obs, envs.nb_states), flat_to_one_hot(next_obs, envs.nb_states), actions, feedback, dones, infos])

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs
                step += 1

            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                if global_step % args.train_frequency == 0:
                    data = rb.sample(args.batch_size)
                    with torch.no_grad():
                        target_max, _ = q_network(data.next_observations).max(dim=1)
                        td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                    old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                    if global_step % 100 == 0:
                        writer.add_scalar("losses/td_loss", loss, global_step)
                        writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                        #print("SPS:", int(global_step / (time.time() - start_time)))
                        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                    # optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # update target network
                if global_step % args.target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                        target_network_param.data.copy_(
                            args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                        )

        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(q_network.state_dict(), model_path)
            print(f"model saved to {model_path}")


        #save replay buffer
        f = open(run_name + "/memory_expert", "wb")
        pickle.dump(rb_success, f)
        f.close()

        f = open(run_name + "/memory_exploration", "wb")
        pickle.dump(rb, f)
        f.close()

        #save agent
        model_path = os.path.join(run_name, 'q_model.pth')
        torch.save(q_network.state_dict(), model_path)

        #envs.close()
        writer.close()