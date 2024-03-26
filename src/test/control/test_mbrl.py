from unittest import TestCase
from control.mbrl import MBRLLearner
import gymnasium as gym
import math
import numpy as np


class TestMBRL(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_mbrl_inverted_pendulum(self):
        """
        Test to see if can run an example without crashing.
        """
        state_dim = 4
        action_dim = 1
        env = gym.make("InvertedPendulum-v4")
        num_episodes = 5000
        episode_len = 100
        batch_size = 256

        def reward(state, action):
            return 1

        def terminate(state, action, t):
            # If episode is at t>=1000, terminate episode
            if t >= 1000:
                return True
            # If absolute value of vertical angle between pole and cart is greater than 0.2,
            # terminate episode
            elif state[1] > 0.2 or state[1] < -0.2:
                return True
            else:
                return False

        learner = MBRLLearner(state_dim=state_dim, action_dim=action_dim, env=env,
                              num_episodes=num_episodes, episode_len=episode_len, reward=reward,
                              terminate=terminate, batch_size=batch_size)
        learner.train()

    # def test_mbrl_cartpole(self):
    #     """
    #     Test to see if can run an example without crashing.
    #     """
    #     state_dim = 4
    #     action_dim = 1
    #     env = gym.make("CartPole-v1")
    #     num_episodes = 2000
    #     episode_len = 200
    #     batch_size = 256
    #     train_buffer_len = num_episodes  # Right now have it set to only supervised learning
    #
    #     def reward(state, action):
    #         return 1
    #
    #     def terminate(state, action, t):
    #         if t >= 500:
    #             return True
    #         elif state[0] > 2.4 or state[0] < -2.4:
    #             return True
    #         elif state[2] > 0.2 or state[2] < -0.2:
    #             return True
    #         else:
    #             return False
    #
    #     learner = MBRLLearner(state_dim=state_dim, action_dim=action_dim, env=env,
    #                           num_episodes=num_episodes, episode_len=episode_len, reward=reward,
    #                           terminate=terminate, batch_size=batch_size, train_buffer_len=train_buffer_len,
    #                           save_name="demo1", normalize=False)
    #     learner.train()

    def test_mbrl_pendulum(self):
        """
        Test to see if can run an example without crashing.
        """
        state_dim = 2
        action_dim = 1
        env = gym.make("Pendulum-v1")
        num_episodes = 3000
        episode_len = 200
        batch_size = 256
        train_buffer_len = num_episodes  # Right now have it set to only supervised learning

        def angle_normalize(x):
            return ((x + np.pi) % (2 * np.pi)) - np.pi

        def reward(state, action):
            th = state[0]
            thdot = state[1]
            u = action
            return - (angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2))

        learner = MBRLLearner(state_dim=state_dim, action_dim=action_dim, env=env,
                              num_episodes=num_episodes, episode_len=episode_len, reward=reward,
                              terminate=None, batch_size=batch_size, train_buffer_len=train_buffer_len,
                              save_name="pend_demo", normalize=False)
        learner.train()

