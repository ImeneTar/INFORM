import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.table import Table
from matplotlib import rc
from gymnasium.spaces import Box, Discrete
from utilis import discreteProb

class gridworld:

    def __init__(self, height=6, width=6, goal=5, walls=[7, 13, 15, 16, 19, 25],
                 h_walls=[[0, 1, [1, 2, 3, 4]], [1, 2, [3, 4]], [2, 3, [3, 4]], [4, 5, [1, 2, 3, 4]]],
                 v_walls=[[0, 1, [1, 2, 3, 4]], [1, 2, [1, 2, 3, 4]], [4, 5, [0]], [2, 3, [2]], [4, 5, [2]]],
                 nb_actions=4):

        self.height = height
        self.width = width
        self.walls = walls
        self.h_walls = h_walls
        self.v_walls = v_walls
        self.start = 0
        self.goal = goal
        self.nb_states, self.states = self.create_states()
        self.grid = self.create_grid()
        self.nb_actions = 4
        self.obs_space = Box(low=0, high=1, shape=(30,), dtype=np.float32)
        self.action_space = Discrete(nb_actions)
        self.actions = np.arange(nb_actions)
        self.transition = self.create_transition()
        self.r = self.create_reward()
        self.hf = self.create_reward_balanced()
        self.hf_positive = self.create_reward_positive()
        self.hf_negative = self.create_reward_negative()

    def reset(self, random=False):
        if random:
            s = np.random.randint(0, self.nb_states - 1)
        else:
            s = 4
        return s

    def get_state(self, i, j):
        s = self.width * i + j
        return s

    def create_states(self):
        states = np.zeros((self.height, self.width))
        s = 0
        for i in range(self.height):
            for j in range(self.width):
                if self.get_state(i, j) in self.walls:
                    states[i, j] = -1
                else:
                    states[i, j] = s
                    s += 1
        nb_states = s
        return nb_states, states

    def create_grid(self):

        grid = np.zeros((self.height, self.width))
        # put -2 in walls cases
        if self.walls:
            for s in self.walls:
                grid[np.where(self.states == s)] = -2

        # put 10 in goal
        grid[np.where(self.states == self.goal)] = 10
        return grid

    def create_transition(self):
        transition_matrix = np.zeros((self.nb_states, self.nb_actions, self.nb_states))

        # Intit the transion matrix
        # U:O, D=1, L=2, R=3
        transition_matrix[:, 0, :] = np.zeros((self.nb_states, self.nb_states))
        transition_matrix[:, 1, :] = np.zeros((self.nb_states, self.nb_states))
        transition_matrix[:, 2, :] = np.zeros((self.nb_states, self.nb_states))
        transition_matrix[:, 3, :] = np.zeros((self.nb_states, self.nb_states))

        for s in range(self.nb_states):
            if s == self.goal:
                transition_matrix[s, :, s] = 1

            else:
                i, j = np.where(self.states == s)

                # Up action
                if i == 0:
                    next_s = s
                else:
                    n_state = int(self.states[i - 1, j])
                    if n_state == -1:
                        next_s = s
                    else:
                        for n in range(len(self.h_walls)):
                            if i == self.h_walls[n][1] and j in self.h_walls[n][2]:
                                next_s = s
                                break
                            else:
                                next_s = int(self.states[i - 1, j])
                transition_matrix[s, 0, next_s] = 1

                # Down action
                if i == self.height - 1:
                    next_s = s
                else:
                    n_state = int(self.states[i + 1, j])
                    if n_state == -1:
                        next_s = s
                    else:
                        for n in range(len(self.h_walls)):
                            if i == self.h_walls[n][0] and j in self.h_walls[n][2]:
                                next_s = s
                                break
                            else:
                                next_s = int(self.states[i + 1, j])
                transition_matrix[s, 1, next_s] = 1

                # Left action
                if j == 0:
                    next_s = s
                else:
                    n_state = int(self.states[i, j - 1])
                    if n_state == -1:
                        next_s = s
                    else:
                        for n in range(len(self.v_walls)):
                            if j == self.v_walls[n][1] and i in self.v_walls[n][2]:
                                next_s = s
                                break
                            else:
                                next_s = int(self.states[i, j - 1])
                transition_matrix[s, 2, next_s] = 1

                # Right action
                if j == self.width - 1:
                    next_s = s
                else:
                    n_state = int(self.states[i, j + 1])
                    if n_state == -1:
                        next_s = s
                    else:
                        for n in range(len(self.v_walls)):
                            if j == self.v_walls[n][0] and i in self.v_walls[n][2]:
                                next_s = s
                                break
                            else:
                                next_s = int(self.states[i, j + 1])
                transition_matrix[s, 3, next_s] = 1

        return transition_matrix

    def create_reward(self):
        reward = np.zeros((self.nb_states, self.nb_actions))

        for s in range(self.nb_states):

            for a in range(self.nb_actions):
                next_s = np.where(self.transition[s, a, :] == 1)[0][0]
                if next_s == self.goal:
                    reward[s, a] = 1
        return reward

    def create_reward_balanced(self):
        reward = np.zeros((self.nb_states, self.nb_actions))

        for s in range(self.nb_states):

            for a in range(self.nb_actions):
                next_s = np.where(self.transition[s, a, :] == 1)[0][0]
                if next_s == self.goal:
                    reward[s, a] = 1
                elif next_s == s:
                    reward[s, a] = -1
                else:
                    i, j = np.where(self.states == s)
                    if (i == 0 and j != 0 and a == 2) or (j == 0 and i != self.height - 1 and a == 1) or (
                            i == self.height - 1 and j != self.width - 1 and a == 3) or (
                            j == self.width - 1 and a == 0) or (j == 2 and (i==2 or i==3 or i==4) and a == 0)\
                            or (i == 1 and (j==2 or j==3 or j==4) and a == 3):
                        reward[s, a] = 1

                    if (i == 0 and j != 0 and j != 4 and a == 3) or (j == 0 and i != self.height - 1 and a == 0) \
                            or (i == self.height - 1 and j != self.width - 1 and a == 2) \
                            or (j == self.width - 1 and a == 1) \
                            or (i == 1 and j in (j==3 or j==4 or j==5) and a == 2)\
                            or (j == 2 and (i==1 or i==2 or i==3) and a == 1):
                        reward[s, a] = - 1
        return reward

    def create_reward_positive(self):
        reward = np.zeros((self.nb_states, self.nb_actions))

        for s in range(self.nb_states):

            for a in range(self.nb_actions):
                next_s = np.where(self.transition[s, a, :] == 1)[0][0]
                if next_s == self.goal:
                    reward[s, a] = 1
                elif next_s == s:
                    reward[s, a] = -1

                else:
                    i, j = np.where(self.states == s)
                    if (i == 0 and j != 0 and a == 2) or (j == 0 and i != self.height - 1 and a == 1) or (
                            i == self.height - 1 and j != self.width - 1 and a == 3) or (
                            j == self.width - 1 and j != 0 and a == 0):
                        reward[s, a] = 1

        return reward

    def create_reward_negative(self):
        reward = np.zeros((self.nb_states, self.nb_actions))

        for s in range(self.nb_states):

            for a in range(self.nb_actions):
                next_s = np.where(self.transition[s, a, :] == 1)[0][0]
                if next_s == self.goal:
                    reward[s, a] = 1
                if next_s == s:
                    reward[s, a] = -1

                else:
                    i, j = np.where(self.states == s)
                    if (i == 0 and j != 0 and j != 4 and a == 3) or (j == 0 and i != self.height - 1 and a == 0) \
                            or (i == self.height - 1 and j != self.width - 1 and a == 2) \
                            or (j == self.width - 1 and a == 1) \
                            or (i == 1 and j in (j == 3 or j == 4 or j == 5) and a == 2) \
                            or (j == 2 and (i == 1 or i == 2 or i == 3) and a == 1):
                        reward[s, a] = - 1

        return reward

    def step(self, s, a, timestep):
        info = []
        next_s = np.where(self.transition[s, a, :] == 1)[0][0]
        reward = self.r[s, a]

        if next_s == self.goal or timestep == 200:
            done = True
        else:
            done = False

        return next_s, reward, done, info

    def get_random_action(self):
        action = np.random.randint(self.nb_actions)
        return action

    def sample(self, prob_list=None):
        # returns an action drawn according to the prob_list distribution,
        # if the param is not set, then it is drawn from a uniform distribution
        if prob_list is None:
            prob_list = np.ones(self.nb_actions) / self.nb_actions
        index = discreteProb(prob_list)
        return self.actions[index]



class gridworld_wall:

    def __init__(self, height=6, width=6, goal=5, walls=[7, 13, 15, 16, 19, 25],
                 h_walls=[[0, 1, [1, 2, 3, 4]], [1, 2, [3, 4, 5]], [2, 3, [3, 4, 5]], [4, 5, [1, 2, 3, 4]]],
                 v_walls=[[0, 1, [1, 2, 3, 4]], [1, 2, [1, 2, 3, 4]], [4, 5, [0]], [2, 3, [2]], [4, 5, [2]]],
                 nb_actions=4):

        self.height = height
        self.width = width
        self.walls = walls
        self.h_walls = h_walls
        self.v_walls = v_walls
        self.start = 0
        self.goal = goal
        self.nb_states, self.states = self.create_states()
        self.grid = self.create_grid()
        self.nb_actions = 4
        self.action_space = Discrete(nb_actions)
        self.actions = np.arange(nb_actions)
        self.actions = np.arange(nb_actions)
        self.transition = self.create_transition()
        self.r = self.create_reward()
        self.hf = self.create_reward_balanced()
        self.hf_positive = self.create_reward_positive()
        self.hf_negative = self.create_reward_negative()

    def reset(self, random=False):
        if random:
            s = np.random.randint(0, self.nb_states - 1)
        else:
            s = 4
        return s

    def get_state(self, i, j):
        s = self.width * i + j
        return s

    def create_states(self):
        states = np.zeros((self.height, self.width))
        s = 0
        for i in range(self.height):
            for j in range(self.width):
                if self.get_state(i, j) in self.walls:
                    states[i, j] = -1
                else:
                    states[i, j] = s
                    s += 1
        nb_states = s
        return nb_states, states

    def create_grid(self):

        grid = np.zeros((self.height, self.width))
        # put -2 in walls cases
        if self.walls:
            for s in self.walls:
                grid[np.where(self.states == s)] = -2

        # put 10 in goal
        grid[np.where(self.states == self.goal)] = 10
        return grid

    def create_transition(self):
        transition_matrix = np.zeros((self.nb_states, self.nb_actions, self.nb_states))

        # Intit the transion matrix
        # U:O, D=1, L=2, R=3
        transition_matrix[:, 0, :] = np.zeros((self.nb_states, self.nb_states))
        transition_matrix[:, 1, :] = np.zeros((self.nb_states, self.nb_states))
        transition_matrix[:, 2, :] = np.zeros((self.nb_states, self.nb_states))
        transition_matrix[:, 3, :] = np.zeros((self.nb_states, self.nb_states))

        for s in range(self.nb_states):
            if s == self.goal:
                transition_matrix[s, :, s] = 1

            else:
                i, j = np.where(self.states == s)

                # Up action
                if i == 0:
                    next_s = s
                else:
                    n_state = int(self.states[i - 1, j])
                    if n_state == -1:
                        next_s = s
                    else:
                        for n in range(len(self.h_walls)):
                            if i == self.h_walls[n][1] and j in self.h_walls[n][2]:
                                next_s = s
                                break
                            else:
                                next_s = int(self.states[i - 1, j])
                transition_matrix[s, 0, next_s] = 1

                # Down action
                if i == self.height - 1:
                    next_s = s
                else:
                    n_state = int(self.states[i + 1, j])
                    if n_state == -1:
                        next_s = s
                    else:
                        for n in range(len(self.h_walls)):
                            if i == self.h_walls[n][0] and j in self.h_walls[n][2]:
                                next_s = s
                                break
                            else:
                                next_s = int(self.states[i + 1, j])
                transition_matrix[s, 1, next_s] = 1

                # Left action
                if j == 0:
                    next_s = s
                else:
                    n_state = int(self.states[i, j - 1])
                    if n_state == -1:
                        next_s = s
                    else:
                        for n in range(len(self.v_walls)):
                            if j == self.v_walls[n][1] and i in self.v_walls[n][2]:
                                next_s = s
                                break
                            else:
                                next_s = int(self.states[i, j - 1])
                transition_matrix[s, 2, next_s] = 1

                # Right action
                if j == self.width - 1:
                    next_s = s
                else:
                    n_state = int(self.states[i, j + 1])
                    if n_state == -1:
                        next_s = s
                    else:
                        for n in range(len(self.v_walls)):
                            if j == self.v_walls[n][0] and i in self.v_walls[n][2]:
                                next_s = s
                                break
                            else:
                                next_s = int(self.states[i, j + 1])
                transition_matrix[s, 3, next_s] = 1

        return transition_matrix

    def create_reward(self):
        reward = np.zeros((self.nb_states, self.nb_actions))

        for s in range(self.nb_states):

            for a in range(self.nb_actions):
                next_s = np.where(self.transition[s, a, :] == 1)[0][0]
                if next_s == self.goal:
                    reward[s, a] = 1
        return reward

    def create_reward_balanced(self):
        reward = np.zeros((self.nb_states, self.nb_actions))

        for s in range(self.nb_states):

            for a in range(self.nb_actions):
                next_s = np.where(self.transition[s, a, :] == 1)[0][0]
                if next_s == self.goal:
                    reward[s, a] = 1
                elif next_s == s:
                    reward[s, a] = -1

                else:
                    i, j = np.where(self.states == s)
                    if (i == 0 and j != 0 and a == 2) or (j == 0 and i != self.height - 1 and a == 1) or (
                            i == self.height - 1 and j != self.width - 1 and a == 3) or (
                            j == self.width - 1 and j != 0 and a == 0):
                        reward[s, a] = 1

                    # if (i == 0 and j != 0 and j != 4 and a == 3) or (j == 0 and i != self.height - 1 and a == 0) or (
                    #         i == self.height - 1 and j != self.width - 1 and a == 2) or (
                    #         j == self.width - 1 and j != 0 and a == 1) or (j == self.width - 1 and a == 2):
                    #     reward[s, a] = - 1
        return reward

    def create_reward_positive(self):
        reward = np.zeros((self.nb_states, self.nb_actions))

        for s in range(self.nb_states):

            for a in range(self.nb_actions):
                next_s = np.where(self.transition[s, a, :] == 1)[0][0]
                if next_s == self.goal:
                    reward[s, a] = 1
                # elif next_s == s:
                #   reward[s, a] = -1

                else:
                    i, j = np.where(self.states == s)
                    if (i == 0 and j != 0 and a == 2) or (j == 0 and i != self.height - 1 and a == 1) or (
                            i == self.height - 1 and j != self.width - 1 and a == 3) or (
                            j == self.width - 1 and j != 0 and a == 0):
                        reward[s, a] = 1

        return reward

    def create_reward_negative(self):
        reward = np.zeros((self.nb_states, self.nb_actions))

        for s in range(self.nb_states):

            for a in range(self.nb_actions):
                next_s = np.where(self.transition[s, a, :] == 1)[0][0]
                if next_s == self.goal:
                    reward[s, a] = 1
                if next_s == s:
                    reward[s, a] = -1
                """
                else:
                    i, j = np.where(self.states == s)
                    if (i == 0 and j != 0  and a == 3) or (j == 0 and i != self.height - 1 and a == 0) or (
                            i == self.height - 1 and j != self.width - 1 and a == 2) or (
                            j == self.width - 1 and i != 0 and a == 1) or (j == self.width - 1 and a == 2):
                        reward[s, a] = - 1"""

        return reward

    def get_random_action(self):
        action = np.random.randint(self.nb_actions)
        return action

    def step(self, s, a, timestep):
        info = []
        next_s = np.where(self.transition[s, a, :] == 1)[0][0]
        reward = self.r[s, a]

        if next_s == self.goal or timestep > 200:
            done = True
        else:
            done = False

        return next_s, reward, done, info

    def sample(self, prob_list=None):
        # returns an action drawn according to the prob_list distribution,
        # if the param is not set, then it is drawn from a uniform distribution
        if prob_list is None:
            prob_list = np.ones(self.nb_actions) / self.nb_actions

        index = discreteProb(prob_list)
        return self.actions[index]


class maze_plotter():
    def __init__(self, gridworld, goal, optimal=[], red=[], gray=[]):  # maze defined in the mdp notebook
        self.height = gridworld.height
        self.width = gridworld.width
        self.walls = gridworld.walls
        self.h_walls = gridworld.h_walls
        self.v_walls = gridworld.v_walls
        self.goal = goal
        self.states = gridworld.states
        self.nb_states = gridworld.nb_states
        plt.ion()
        self.figW = self.width
        self.figH = self.height
        self.figure_history = []
        self.axes_history = []
        self.table_history = []
        self.agent_patch_history = []
        self.optimal = optimal
        self.red = red
        self.gray = gray

    def init_table(self):  # the states of the mdp are drawn in a matplotlib table, this function creates this table

        width = 1  # 0.1
        height = 1  # 0.2
        self.axes_history[-1].set_xlim([0, self.width])
        self.axes_history[-1].set_ylim([0, self.height])

        edge_color = np.zeros(3)
        edge_color[0] = edge_color[1] = edge_color[2] = 0.75

        for i in range(self.height):
            for j in range(self.width):
                state = self.width * i + j
                color = np.zeros(3)
                if state in self.walls or state in self.gray:
                    color[0] = color[1] = color[2] = 0.75
                elif state == self.goal:
                    color[0] = 0.96
                    color[1] = 0.85
                    color[2] = 0.55
                elif state in self.optimal:
                    color[0] = 0.85
                    color[1] = 0.97
                    color[2] = 0.87

                elif state in self.red:
                    color[0] = 0.99
                    color[1] = 0.73
                    color[2] = 0.73

                else:
                    color[0] = color[1] = color[2] = 1
                self.table_history[-1].add_cell(i, j, width, height, facecolor=color, edgecolor=edge_color, text='',
                                                loc='center')
        self.axes_history[-1].add_table(self.table_history[-1])
        # self.axes_history[-1].vlines(x=4, ymin=1, ymax=3,color='b')

    def init_walls(self):
        for h in self.h_walls:
            raw = h[1]
            start = h[2][0]
            end = h[2][-1]
            self.axes_history[-1].hlines(y=self.height - raw, xmin=start, xmax=end + 1, color='k', linewidth=2)

        for v in self.v_walls:
            col = v[1]
            start = self.height - v[2][-1] - 1
            end = self.height - v[2][0]
            self.axes_history[-1].vlines(x=col, ymin=start, ymax=end, color='k', linewidth=2)

    def new_render(self):  # initializes the plot by creating its basic components (figure, axis, agent patch and table)
        # a trace of these components is stored so that the old outputs will last on the notebook
        # when a new rendering is performed
        self.figure_history.append(plt.figure(figsize=(self.figW, self.figH)))
        self.axes_history.append(
            self.figure_history[-1].add_subplot(111))  # 111 : number of rows, columns and index of subplot
        self.table_history.append(Table(self.axes_history[-1], bbox=[0, 0, 1, 1]))  # [left, bottom, width, height]
        self.agent_patch_history.append(mpatches.Ellipse((-1, -1), 0.06, 0.085, ec="none", fc="dodgerblue", alpha=0.6))
        self.axes_history[-1].add_patch(self.agent_patch_history[-1])
        self.init_table()
        self.init_walls()
        plt.xticks([])
        plt.yticks([])

    def simple_render(self, agent_state=-1, V=[], policy=[], render=True):  # updates the values of the table
        # and the agent position and current policy
        # some of these components may not show depending on the parameters given when calling this function
        if len(self.figure_history) == 0:  # new plot
            self.new_render()

        self.axes_history[-1].clear()
        self.axes_history[-1].set_xlim([0, self.width])
        self.axes_history[-1].set_ylim([0, self.height])
        self.axes_history[-1].add_table(self.table_history[-1])
        self.init_walls()

        #### Table values and policy update
        if len(V) > 0:  # working with state values
            self.Q_render(V, policy)

        # plt.subplots_adjust(left=0.2, bottom=0.2)
        plt.xticks([])
        plt.yticks([])

    def render(self, agent_state=-1, V=[], policy=[], render=True):  # updates the values of the table
        # and the agent position and current policy
        # some of these components may not show depending on the parameters given when calling this function
        if len(self.figure_history) == 0:  # new plot
            self.new_render()

        self.axes_history[-1].clear()
        self.axes_history[-1].set_xlim([0, self.width])
        self.axes_history[-1].set_ylim([0, self.height])
        self.axes_history[-1].add_table(self.table_history[-1])
        self.init_walls()

        #### Table values and policy update
        if len(V) > 0:  # working with state values
            self.Q_render(V, policy)

        if agent_state >= 0:
            x, y = self.coords(agent_state)

            self.agent_patch_history[-1].center = x, y
            self.axes_history[-1].add_patch(self.agent_patch_history[-1])
            # print(agent_state,i,x,j,y)

        # plt.subplots_adjust(left=0.2, bottom=0.2)
        plt.xticks([])
        plt.yticks([])
        """if render :
            self.figure_history[-1].canvas.draw()
            self.figure_history[-1].canvas.flush_events()"""

        return self.figure_history[-1]

    def Q_render(self, Q, policy):

        for state in range(self.nb_states):
            # print("state:", state)
            # print("x,y:", self.get_coord(state))

            if state != self.goal:  ##goal in state space not grid space
                """qmin = np.min(Q[:,state])
                if qmin < 0:
                    qmin *= -1
                pos_Q = Q[:,state] + qmin
                qmax = np.max(pos_Q)
                norm_Q = pos_Q / (np.sum(pos_Q)-(list(pos_Q).count(qmax)*qmax)+0.1)
                print(state)
                print(norm_Q)
                """

                if not all(Q[:, state] == 0):

                    norm = np.linalg.norm(Q[:, state])
                    norm_Q = np.absolute(Q[:, state] / norm)

                    for action in range(len(Q[:, state])):
                        x0, y0, x, y = self.qarrow_params(self.height,

                                                          self.width, state, action)

                        arw_color = "green"
                        alpha = norm_Q[action]

                        qmax = np.max(Q[:, state])
                        qmin = np.min(Q[:, state])
                        # print("action:",Q[:,state].argmax(axis=0))

                        if Q[action][state] > 0:
                            arw_color = "green"
                            if Q[action][state] == qmax:
                                alpha = 0.9
                            else:
                                alpha = norm_Q[action]

                        if Q[action][state] < 0:
                            arw_color = "red"
                            if Q[action][state] == qmin:
                                alpha = 0.9
                            else:
                                alpha = norm_Q[action]

                        """
                        if not Q[action][state]==qmax:
                            arw_color = "red"
                            alpha = norm_Q[action]

                        """

                        if x == 0 and y == 0:
                            circle = mpatches.Circle((x0, y0), 0.02, ec=arw_color, fc=arw_color, alpha=alpha)
                            self.axes_history[-1].add_patch(circle)
                        else:

                            # self.axes_history[-1].arrow(x0, y0, x, y)

                            self.axes_history[-1].arrow(x0, y0, x, y, alpha=alpha,
                                                        head_width=0.1, head_length=0.1, fc=arw_color, ec=arw_color)

    def agent_render(self, agent_state=-1, n_episode=0, step=0, render=True):  # updates the values of the table
        # and the agent position and current policy
        # some of these components may not show depending on the parameters given when calling this function
        if len(self.figure_history) == 0:  # new plot
            self.new_render()

        self.axes_history[-1].clear()
        self.axes_history[-1].set_xlim([0, self.width])
        self.axes_history[-1].set_ylim([0, self.height])
        self.axes_history[-1].add_table(self.table_history[-1])
        self.init_walls()

        if agent_state >= 0:
            x, y = self.coords(agent_state)

            self.agent_patch_history[-1].center = x, y
            self.agent_patch_history[-1].set_color('b')
            self.axes_history[-1].add_patch(self.agent_patch_history[-1])
            # print(agent_state,i,x,j,y)

        ## text update
        self.axes_history[-1].set_title('Episode: ' + str(n_episode) + "                  Timestep: " + str(step),
                                        loc='left', size=14, color='k')

        # plt.subplots_adjust(left=0.2, bottom=0.2)
        plt.xticks([])
        plt.yticks([])
        """if render :
            self.figure_history[-1].canvas.draw()
            self.figure_history[-1].canvas.flush_events()"""

        return self.figure_history[-1]

    def get_coord(self, state):
        i, j = np.where(self.states == state)
        x = j
        y = self.height - 1 - i
        return x[0], y[0]

    def coords(self, state):  # processes the starting position of the arrows
        # i = state%self.maze_attr.height
        # j = int(state/self.maze_attr.height)
        i, j = self.get_coord(state)
        """
        h = 1/self.figH
        ch = h
        w = 1/self.figW
        cw = w/2
        x,y = j*w + cw,1-(i*h + ch)
        """

        return i + 0.5, j + 0.5

    def qarrow_params(self, height, width, state, action):  # processes the starting position of the arrows
        x, y = self.coords(state)

        if action == 0:  # up
            return [x, y + 0.05, 0.0, 0.25]  # 1/(10*self.figH)]
        elif action == 1:  # down
            return [x, y - 0.05, 0.0, -0.25]
        elif action == 2:  # left
            return [x - 0.05, y, -0.25, 0.0]
        elif action == 3:  # right
            return [x + 0.05, y, 0.25, 0.0]
        else:
            return [x, y, 0.0, 0.0]

    def create_animation(self, state_list=[], Q_list=[], pol_list=[], nframes=0):
        new_Qlist = Q_list
        new_polist = pol_list
        """
        if nframes > 0 :
            new_Qlist, new_polist = self.resize_lists(Q_list, pol_list, nframes)
        """

        self.new_render()
        anim = animation.FuncAnimation(self.figure_history[-1], self.update, frames=len(new_Qlist),
                                       fargs=[state_list, new_Qlist, new_polist], blit=True)
        # plt.close()
        return anim

    def update(self, frame, state_list, V_list, pol_list):
        """
        if len(pol_list)>frame:
            return self.render(V=V_list[frame],policy=pol_list[frame], render=False)

        else:
            return self.render(V=V_list[frame], render=False)
        """
        print("hello")
        print("Frame:", frame)
        print("bye")

        return self.render(agent_state=state_list[frame], V=V_list[frame], render=False)

    def agent_animation(self, state_list=[], nframes=0, n_episode=0):
        new_list = state_list
        """
        if nframes > 0 :
            new_Qlist, new_polist = self.resize_lists(Q_list, pol_list, nframes)
        """

        self.new_render()
        anim = animation.FuncAnimation(self.figure_history[-1], self.update_agent, frames=len(new_list),
                                       fargs=[state_list, n_episode], blit=True)
        # plt.close()
        return anim

    def update_agent(self, frame, state_list, n_episode=0):
        """
        if len(pol_list)>frame:
            return self.render(V=V_list[frame],policy=pol_list[frame], render=False)

            return self.render(V=V_list[frame], render=False)
        """
        print(frame)
        return self.agent_render(agent_state=state_list[frame], step=frame, n_episode=n_episode, render=False)
