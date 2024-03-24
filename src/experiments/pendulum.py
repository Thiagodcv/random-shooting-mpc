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
    state_dim = 3
    action_dim = 1
    episode_len = 200
    env = gym.make("Pendulum-v1", render_mode="human")
    model = DynamicsModel(state_dim, action_dim, normalize=False)
    model.load_state_dict(torch.load(os.path.join(MODELS_PATH, "pend_demo.pt")))

    def reward(state, action):
        x = state[0]
        y = state[1]
        rot_x = np.cos(-np.pi / 2) * x - np.sin(-np.pi / 2) * y
        rot_y = np.sin(-np.pi / 2) * x + np.cos(-np.pi / 2) * y
        theta = math.atan2(rot_y, rot_x)
        d_theta = state[2]

        torque = action
        return -(theta ** 2 + 0.1 * d_theta ** 2 + 0.001 * torque ** 2)

    num_traj = 200
    gamma = 0.999
    horizon = 20
    mpc = MPC(model, num_traj, gamma, horizon, reward)

    MBRLLearner.static_eval_model(env, episode_len, mpc, gamma)


if __name__ == "__main__":
    pendulum()
