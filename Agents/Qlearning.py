import numpy as np
from utilis import argmax

class QN:
    def __init__(self, gridworld, reward, alpha=0.01, gamma=0.99, eps=0.05, nEpisodes=300,
                 timeout=400, starts=None):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.nEpisodes = nEpisodes
        self.timeout = timeout
        self.nb_features = gridworld.nb_states
        self.nb_actions = gridworld.nb_actions
        self.states = gridworld.states
        self.w = np.zeros((self.nb_actions, self.nb_features))
        self.map = np.zeros_like(gridworld.states)
        self.starts = starts
        self.time = []
        self.reward = reward
        self.trajectory = []

    def select_action(self, x):
        # epsilon greedy search
        action_values = []
        for a in range(self.nb_actions):
            action_values.append(self.w[a][x])
        if np.random.random() < self.eps:
            chosen_action = np.random.choice(self.nb_actions)
        else:
            chosen_action = argmax(action_values)

        return chosen_action, action_values[chosen_action]

    def compute(self, gridworld):
        # select action
        u, action_value = self.select_action(x)
        # take action
        r, y, done = gridworld.step(x, u)
        r = self.reward[x,u]
        # append trajectory
        #self.trajectory.append((x, u, y))

        # update w
        if done:
            self.w[u][x] += self.alpha * (r - action_value)
        else:
            _, new_action_value= self.select_action(y)
            self.w[u][x] += self.alpha * (r + self.gamma * new_action_value - action_value)
        return u, r, y, done

    def test(self, gridworld, x):
        # select action
        u, action_value = self.select_action(x)
        # take action
        r, y, done = gridworld.step(x, u)
        return u, r, y, done

    def update_heatmap(self, x):
        i,j = np.where(self.states == x)
        self.map[i,j] +=1

    def plot_reward(self):
        plt.plot(self.reward)

    def plot_time(self):
        plt.plot(self.time)