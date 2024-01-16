import numpy as np
from scipy.special import logsumexp as sp_lse
import time

def argmax(q_values):
    top = float("-inf")
    ties = []
    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = []
        if q_values[i] == top:
            ties.append(i)
    return np.random.choice(ties)

def discreteProb(p):
    # Draw a random number using probability table p (column vector)
    # Suppose probabilities p=[p(1) ... p(n)] for the values [1:n] are given, sum(p)=1
    # and the components p(j) are nonnegative.
    # To generate a random sample of size m from this distribution,
    # imagine that the interval (0,1) is divided into intervals with the lengths p(1),...,p(n).
    # Generate a uniform number rand, if this number falls in the jth interval given the discrete distribution,
    # return the value j. Repeat m times.
    r = np.random.random()
    cumprob = np.hstack((np.zeros(1), p.cumsum()))
    sample = -1
    for j in range(p.size):
        if (r > cumprob[j]) & (r <= cumprob[j + 1]):
            sample = j
            break
    return sample

def flat_to_one_hot(val, ndim):
    shape =np.array(val).shape
    v = np.zeros(shape + (ndim,))
    if len(shape) == 1:
        v[np.arange(shape[0]), val] = 1.0
    else:
        v[val] = 1.0
    return v

def get_policy(q_fn, ent_wt=1.0):
    """
    Return a policy by normalizing a Q-function
    """
    v_rew = logsumexp(q_fn, alpha=ent_wt)
    adv_rew = q_fn - np.expand_dims(v_rew, axis=1)
    pol_probs = np.exp((1.0/ent_wt)*adv_rew)
    assert np.all(np.isclose(np.sum(pol_probs, axis=1), 1.0)), str(pol_probs)
    return pol_probs

def get_trajectory(gridword, pol, timeout=30, start=4):
    t = 0
    s = np.random.randint(0, gridword.nb_states - 1)
    if s == gridword.goal:
        done = True
    else:
        done = False
    traj = []
    while not done and t!=timeout:
        t += 1
        a = int(np.argmax(pol[s,:]))
        traj.append([s,a])
        next_s = np.where(gridword.transition[s,a,:]==1)[0][0]
        if next_s == gridword.goal:
            done = True
            break
        else:
            s = next_s
    return traj

def generate_trajectories(gridword, pol, n):
    trajectories = []
    for i in range(n):
        trajectories.append(get_trajectory(gridword, pol))
    return trajectories

def get_prob(gridworld, trajectories):
    visitation_freq = np.zeros((gridworld.nb_states, gridworld.nb_actions))
    for t in trajectories:
        for i in range(len(t)):
            visitation_freq[t[i][0], t[i][1]] += 1
    prob = visitation_freq.copy()
    for row in prob:
        n = sum(row)
        if n > 0:
            row[:] = [f / sum(row) for f in row]
        else:
            row[:] = 1 / gridworld.nb_actions
    return prob


def logsumexp(q, alpha=1.0, axis=1):
    return alpha*sp_lse((1.0/alpha)*q, axis=axis)

def adam_optimizer(lr, beta1=0.9, beta2=0.999, eps=1e-8):
    itr = 0
    pm = None
    pv = None
    def update(x, grad):
        nonlocal itr, lr, pm, pv
        if pm is None:
            pm = np.zeros_like(grad)
            pv = np.zeros_like(grad)

        pm = beta1 * pm + (1-beta1)*grad
        pv = beta2 * pv + (1-beta2)*(grad*grad)
        mhat = pm/(1-beta1**(itr+1))
        vhat = pv/(1-beta2**(itr+1))
        update_vec = mhat / (np.sqrt(vhat)+eps)
        new_x = x - lr * update_vec
        itr += 1
        return new_x
    return update

class TrainingIterator(object):
    def __init__(self, itrs, heartbeat=float('inf')):
        self.itrs = itrs
        self.heartbeat_time = heartbeat
        self.__vals = {}

    def random_idx(self, N, size):
        return np.random.randint(0, N, size=size)

    @property
    def itr(self):
        return self.__itr

    @property
    def heartbeat(self):
        return self.__heartbeat

    @property
    def elapsed(self):
        assert self.heartbeat, 'elapsed is only valid when heartbeat=True'
        return self.__elapsed

    def itr_message(self):
        return '==> Itr %d/%d (elapsed:%.2f)' % (self.itr+1, self.itrs, self.elapsed)

    def record(self, key, value):
        if key in self.__vals:
            self.__vals[key].append(value)
        else:
            self.__vals[key] = [value]

    def pop(self, key):
        vals = self.__vals.get(key, [])
        del self.__vals[key]
        return vals

    def pop_mean(self, key):
        return np.mean(self.pop(key))

    def __iter__(self):
        prev_time = time.time()
        self.__heartbeat = False
        for i in range(self.itrs):
            self.__itr = i
            cur_time = time.time()
            if (cur_time-prev_time) > self.heartbeat_time or i==(self.itrs-1):
                self.__heartbeat = True
                self.__elapsed = cur_time-prev_time
                prev_time = cur_time
            yield self
            self.__heartbeat = False