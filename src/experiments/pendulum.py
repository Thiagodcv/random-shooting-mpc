import numpy as np
import torch
import gymnasium as gym
from src.control.mpc import MPC
from src.control.dynamics import DynamicsModel
from src.control.mbrl import MBRLLearner
from src.constants import MODELS_PATH
import os
import math


def pendulum():
    state_dim = 2
    action_dim = 1
    episode_len = 200
    env = gym.make("Pendulum-v1", render_mode="human")
    model = DynamicsModel(state_dim, action_dim, normalize=False)
    model.load_state_dict(torch.load(os.path.join(MODELS_PATH, "pend_demo.pt")))

    def angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def reward(state, action):
        th = state[0]
        thdot = state[1]
        u = action
        return - (angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2))

    num_traj = 20
    gamma = 0.999
    horizon = 10
    mpc = MPC(model, num_traj, gamma, horizon, reward)

    MBRLLearner.static_eval_model(env, episode_len, mpc, gamma)


if __name__ == "__main__":
    pendulum()
