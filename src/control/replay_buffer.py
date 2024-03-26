import random
import collections

import numpy as np

Transition = collections.namedtuple('Transition', ('state', 'action', 'next_state'))


class ReplayBuffer:

    def __init__(self, state_dim, action_dim, max_size=10000, normalize=False):
        self.data = collections.deque([], maxlen=max_size)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_mean, self.state_var = np.zeros(self.state_dim), np.ones(self.state_dim)
        self.action_mean, self.action_var = np.zeros(self.action_dim), np.ones(self.action_dim)
        self.normalize = normalize

    def push(self, state, action, next_state):
        """
        Push (s, a, s') tuple to replay memory where 'state', 'action', and 'next_state' are not yet normalized.
        """
        # Update means and variances
        if self.__len__() > 1 and self.normalize:
            self.state_mean = (self.__len__()*self.state_mean + state) / (self.__len__() + 1)
            self.action_mean = (self.__len__()*self.action_mean + action) / (self.__len__() + 1)
            self.state_var = ((self.__len__() - 1)*self.state_var +
                              np.square(state - self.state_mean))/(self.__len__() + 1 - 1)
            self.action_var = ((self.__len__() - 1)*self.action_var +
                               np.square(action - self.action_mean))/(self.__len__() + 1 - 1)

        # Push normalized data into replay buffer
        transition = Transition(state, action, next_state)
        self.data.append(transition)

    def sample(self, batch_size):
        """
        Sample a batch of experiences from replay. Note that the third element returned is
        'next_state - state', NOT 'next_state'.
        """
        transitions = random.sample(self.data, batch_size)
        batch = Transition(*zip(*transitions))

        state = np.array(batch.state)
        action = np.array(batch.action)
        next_state = np.array(batch.next_state)

        # Normalize data
        if self.normalize:
            n_state, n_action, d_n_state = self.normalize_tuple(state, action, next_state, batch_size)
            return n_state, n_action, d_n_state

        return state, action, next_state  # - state

    def normalize_tuple(self, state, action, next_state, batch_size):
        """
        Normalize state, action, and next_state - state. Assume covariance matrix of action
        and states have zero on the diagonal.
        """
        n_state = (state - self.state_mean) @ np.diagflat(np.reciprocal(np.sqrt(self.state_var)))
        n_action = (action - self.action_mean) @ np.diagflat(np.reciprocal(np.sqrt(self.action_var)))
        n_next_state = (next_state - self.state_mean) @ np.diagflat(np.reciprocal(np.sqrt(self.state_var)))
        d_n_state = n_next_state - n_state

        # Add zero-mean gaussian noise to tuple variables
        n_state += self.sample_state_gaussian(batch_size)
        n_action += self.sample_action_gaussian(batch_size)
        d_n_state += self.sample_state_gaussian(batch_size)

        return n_state, n_action, d_n_state

    def sample_state_gaussian(self, size):
        return np.random.multivariate_normal(mean=np.zeros(self.state_dim),
                                             cov=0.01*np.eye(self.state_dim),
                                             size=size)

    def sample_action_gaussian(self, size):
        return np.random.multivariate_normal(mean=np.zeros(self.action_dim),
                                             cov=0.01*np.eye(self.action_dim),
                                             size=size)

    def get_state_mean(self):
        return self.state_mean

    def get_state_var(self):
        return self.state_var

    def get_action_mean(self):
        return self.action_mean

    def get_action_var(self):
        return self.action_var

    def __len__(self):
        return len(self.data)
