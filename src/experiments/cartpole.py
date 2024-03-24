import numpy as np
import torch
import gymnasium as gym
from src.control.mpc import MPC
from src.control.dynamics import DynamicsModel
from src.control.mbrl import MBRLLearner
from src.constants import MODELS_PATH
import os


def cartpole():
    state_dim = 4
    action_dim = 1
    episode_len = 200
    env = gym.make("CartPole-v1", render_mode="human")
    model = DynamicsModel(state_dim, action_dim, normalize=True)
    model.load_state_dict(torch.load(os.path.join(MODELS_PATH, "demo.pt")))

    def reward(state, action):
        return 1

    def terminate(state, action, t):
        if t >= 500:
            return True
        elif state[0] > 2.4 or state[0] < -2.4:
            return True
        elif state[2] > 0.2 or state[2] < -0.2:
            return True
        else:
            return False

    num_traj = 30
    gamma = 0.999
    horizon = 20
    mpc = MPC(model, num_traj, gamma, horizon, reward, terminate)

    MBRLLearner.static_eval_model(env, episode_len, mpc, gamma)


if __name__ == "__main__":
    cartpole()
