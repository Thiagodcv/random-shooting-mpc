from control.mbrl import MBRLLearner
import gymnasium as gym
import numpy as np
import multiprocessing
import ray


def reward(state, action):
    x_vel = state[13]
    # return x_vel + 0.5 - 0.005 * np.linalg.norm(action/150)**2
    return x_vel + 0.5 - 0.5 / 8 * np.linalg.norm(action) ** 2


def terminate(state, action, t):
    return state[0] < 0.2 or state[0] > 1.0


def run_mbrl(num_times=1):
    """
    Parameters:
    -----------
    num_times: int
        number of times to train a new model
    """

    env_dict = {
        'state_dim': 27,
        'action_dim': 8,
        'env': gym.make('Ant-v4')
    }

    train_dict = {
        'num_episodes': 275,
        'num_rand_eps': 25,
        'episode_len': 200,
        'reward': reward,
        'terminate': terminate,
        'lr': 1e-3,
        'batch_size': 256,
        'rl_prop': 0.9,
        'epsilon': 0.05
    }

    mpc_dict = {
        'num_traj': 1024,
        'gamma': 0.99,
        'horizon': 15
    }

    misc_dict = {
        'normalize': True,
        'override_env_reward': True,
        'override_env_terminate': True,
        'save_name': 'demo_model',
        'save_every_n_episodes': 25,
        'print_every_n_episodes': 10
    }

    # Stuff for Multi-threading using Ray
    num_workers = multiprocessing.cpu_count()
    print("Number of workers: ", num_workers)
    ray.init(num_cpus=num_workers)

    if num_times > 1:
        save_name = misc_dict['save_name']
        for rep in range(num_times):
            misc_dict['save_name'] = save_name + '-run' + str(rep)
            learner = MBRLLearner(env_dict=env_dict, train_dict=train_dict, mpc_dict=mpc_dict, misc_dict=misc_dict)
            learner.train()
    else:
        learner = MBRLLearner(env_dict=env_dict, train_dict=train_dict, mpc_dict=mpc_dict, misc_dict=misc_dict)
        learner.train()


if __name__ == "__main__":
    run_mbrl(num_times=1)
