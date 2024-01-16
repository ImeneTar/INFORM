import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
from model import SimpleQNetwork


class DeepTAMER(object):
    def __init__(self, num_inputs, action_dim, batch_size, gamma=0.99, entropy=0.01, lr=1e-4, freq=4, device_name='cpu'):
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = torch.device(device_name)
        self.h_net = SimpleQNetwork(num_inputs, action_dim, device=self.device).to(self.device)
        self.optimizer = optim.Adam(self.h_net.parameters(), lr=lr, betas=[0.9, 0.999])
        self.train()

    def train(self, training=True):
        self.training = training
        self.h_net.train(training)

    def getV(self, obs):
        h = self.h_net(obs)
        # v = torch.sum(h, dim=1)
        v = self.alpha * \
            torch.logsumexp(h / self.alpha, dim=1, keepdim=True)
        return v

    def choose_action(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            h = self.h_net(state)
            dist = F.softmax(h / self.alpha, dim=1)
            action = torch.argmax(dist, dim=1)
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer, step):
        obs, next_obs, action, feedback, done = replay_buffer.get_samples(self.batch_size, self.device)
        # get next_v

        with torch.no_grad():
            next_q = self.h_net(next_obs)
            y = feedback + (1 - done) * self.gamma * next_v

        # compute loss
        critic_loss = F.mse_loss(self.critic(obs, action), y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # print("loss", critic_loss)
        return critic_loss.item()

    def save(self, path):
        new_path = path + "_weight"
        print("Saving model to ", new_path)
        torch.save(self.h_net.state_dict(), new_path)


    def load(self, path):
        print("Loading from ", path)
        self.h_net.load_state_dict(torch.load(path))

    def infer_q(self, state, action):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = torch.FloatTensor([action]).unsqueeze(0)
        with torch.no_grad():
            q = self.critic(state, action)
        return q.squeeze(0).cpu().numpy()

    def infer_v(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            v = self.getV(state).squeeze()
        return v.cpu().numpy()

def make_tamer(env, batch_size, gamma=0.99, entropy=0.01, lr=1e-4, freq=4, device='cpu'):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DeepTAMER(obs_dim, action_dim, batch_size, gamma=gamma, entropy=entropy, lr=lr, freq=freq, device_name=device)
    return agent

def get_reward_tamer(agent, state, action, next_state, done, gamma=0.99):
    with torch.no_grad():
        q = agent.infer_q(state, action)
        next_v = agent.infer_v(next_state)
        y = (1 - done) * gamma * next_v
        irl_reward = (q - y)
    return irl_reward

