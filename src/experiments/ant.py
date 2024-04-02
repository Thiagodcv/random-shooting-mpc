import numpy as np
import torch
import gymnasium as gym
from src.control.mpc import MPC
from src.control.dynamics import DynamicsModel
from src.control.mbrl import MBRLLearner
from src.constants import MODELS_PATH
import os
from multiprocessing.pool import ThreadPool
import math
import multiprocessing
import ray

import time
import cProfile


def reward(state, action):
    x_vel = state[13]
    return x_vel + 0.5 - 0.005 * np.linalg.norm(action/150) ** 2


def terminate(state, action, t):
    return state[0] < 0.2 or state[0] > 1.0


def ant():
    state_dim = 27
    action_dim = 8
    episode_len = 400
    env = gym.make("Ant-v4", render_mode="human")

    model = DynamicsModel(state_dim, action_dim, normalize=True)
    model.load_state_dict(torch.load(os.path.join(MODELS_PATH, "ant-4-2.pt")))

    num_traj = 4096  # Make sure it's divisible by num_workers
    gamma = 0.999
    horizon = 15

    # Ray stuff
    num_workers = multiprocessing.cpu_count()
    print("Number of workers: ", num_workers)
    ray.init(num_cpus=num_workers)

    mpc = MPC(model, num_traj, gamma, horizon, reward, True, terminate)

    start_time = time.time()
    MBRLLearner.static_eval_model(env, episode_len, mpc, gamma)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    # cProfile.run('pendulum()')
    ant()