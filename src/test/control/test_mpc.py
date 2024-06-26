from unittest import TestCase
from src.control.dynamics import DynamicsModel
from src.constants import MODELS_PATH
from control.mpc import MPC
import gymnasium as gym
import numpy as np
import torch
import os
import time


class TestMPC(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # def test_nn_random_shooting(self):
    #     """
    #     Test to see if can run an example without crashing.
    #     """
    #     PATH = "C:/Users/thiag/Git/random-shooting-mpc/models/good_model.pt"
    #     state_dim = 4
    #     action_dim = 1
    #
    #     # input params
    #     num_traj = 1000
    #     gamma = 0.99
    #     horizon = 10
    #
    #     env = gym.make("InvertedPendulum-v4")  # ,render_mode="human")
    #     model = DynamicsModel(state_dim, action_dim)
    #     model.load_state_dict(torch.load(PATH))
    #
    #     def reward(state, action):
    #         return 1
    #
    #     def terminate(state, action, t):
    #         # If episode is at t>=1000, terminate episode
    #         if t >= 1000:
    #             return True
    #         # If absolute value of vertical angle between pole and cart is greater than 0.2,
    #         # terminate episode
    #         elif state[1] > 0.2 or state[1] < -0.2:
    #             return True
    #         else:
    #             return False
    #
    #     mpc = MPC(model, num_traj, gamma, horizon, reward, terminate)
    #
    #     state, _ = env.reset()
    #     for t in range(100):
    #         print(t)
    #         action = mpc.random_shooting(state)
    #         next_state, reward, terminated, truncated, _ = env.step(action)
    #         if terminated or truncated:
    #             break
    #         state = next_state

    def test_find_discrete_optimal_action(self):
        """
        Test to see if random shooting mpc can find optimal set of actions.
        Note that the action space = {0, 1}.
        """
        # input params
        num_traj = 100
        gamma = 1e-5
        horizon = 20

        # Defining the model just to get the code to run
        state_dim = 4
        action_dim = 1
        model = DynamicsModel(state_dim, action_dim)
        model.load_state_dict(torch.load(os.path.join(MODELS_PATH, "testv8.pt")))

        # Defining the reward
        optimal_action = 1

        def reward(state, action):
            return -(action.item() - optimal_action)**2

        mpc = MPC(model, num_traj, gamma, horizon, reward)

        state_dummy = np.zeros(state_dim)
        for i in range(100):
            print("action: ", mpc.random_shooting(state_dummy))

    def test_termination_function(self):
        """
        Test to see if random shooting mpc avoid outputting "1" if environment
        terminates when action=1 is taken.
        """
        # input params
        num_traj = 100
        gamma = 1e-5
        horizon = 20

        # Defining the model just to get the code to run
        state_dim = 4
        action_dim = 1
        model = DynamicsModel(state_dim, action_dim)
        model.load_state_dict(torch.load(os.path.join(MODELS_PATH, "testv8.pt")))

        def reward(state, action):
            return 1

        def terminate(state, action, t):
            if action == 1:
                return True
            else:
                return False

        mpc = MPC(model, num_traj, gamma, horizon, reward, terminate)

        state_dummy = np.zeros(state_dim)
        for i in range(100):
            print("action: ", mpc.random_shooting(state_dummy))

    def test_random_sampling_time(self):
        start_time = time.time()
        for i in range(200):
            # action_seqs = np.random.uniform(low=-10, high=10, size=(7000, 15, 1))
            action_seqs = np.random.standard_normal(size=(2000, 15, 6))
        print("--- %s seconds ---" % (time.time() - start_time))

    def test_call_times(self):
        start_time = time.time()

        a = torch.zeros(size=(3, 1))
        b = np.ones(3)
        mat1 = np.identity(3)
        mat2 = np.identity(3)
        for i in range(200):
            for seq in range(200):
                for t in range(15):
                    torch.from_numpy(b).float()
                    # np.matmul(mat1, mat2)
                    # np.reciprocal(b)
                    # mat1 @ mat2
                    # np.sqrt(b)
                    # a.detach().numpy()

        print("--- %s seconds ---" % (time.time() - start_time))

    def test_norm_time(self):
        action = np.ones(3)
        action_mean = 0.5 * np.ones(3)
        action_var = 0.2 * np.ones(3)

        start_time = time.time()
        for i in range(200):
            for seq in range(200):
                for t in range(15):
                    # (action - action_mean) @ np.diagflat(np.reciprocal(np.sqrt(action_var)))
                    sqrt_var = np.sqrt(action_var)
                    norm_action = action - action_mean
                    for j in range(norm_action.shape[0]):
                        norm_action[j] = norm_action[j] / sqrt_var[j]

        print("--- %s seconds ---" % (time.time() - start_time))

        sqrt_var = np.sqrt(action_var)
        norm_action = action - action_mean
        for j in range(norm_action.shape[0]):
            norm_action[j] = norm_action[j] / sqrt_var[j]

        norm_action_1 = (action - action_mean) @ np.diagflat(np.reciprocal(np.sqrt(action_var)))
        self.assertTrue(np.linalg.norm(norm_action - norm_action_1) < 1e-5)

    def test_denorm_time(self):
        state = np.ones(3)
        state_mean = 0.5 * np.ones(3)
        state_var = 0.2 * np.ones(3)

        start_time = time.time()
        for i in range(200):
            for seq in range(200):
                for t in range(15):
                    denorm_state = state
                    sqrt_state = np.sqrt(state_var)
                    for j in range(state.shape[0]):
                        denorm_state[j] = denorm_state[j] * sqrt_state[j]
                    denorm_state = denorm_state + state_mean
                    # state @ np.diagflat(np.sqrt(state_var)) + state_mean

        print("--- %s seconds ---" % (time.time() - start_time))

        denorm_state = state
        sqrt_state = np.sqrt(state_var)
        for j in range(state.shape[0]):
            denorm_state[j] = denorm_state[j] * sqrt_state[j]
        denorm_state = denorm_state + state_mean

        denorm_state_1 = state @ np.diagflat(np.sqrt(state_var)) + state_mean
        self.assertTrue(np.linalg.norm(denorm_state - denorm_state_1) < 1e-5)

    def test_model_forward(self):
        import scipy.linalg.blas as blas

        state_dim = 2
        action_dim = 1
        model = DynamicsModel(state_dim, action_dim, normalize=True)
        model.load_state_dict(torch.load(os.path.join(MODELS_PATH, "pend_demo_256.pt")))
        # x = torch.ones(3).float()

        start_time = time.time()
        x = np.ones(3)

        w1 = model.linear_relu_stack[0].weight.detach().numpy()
        b1 = model.linear_relu_stack[0].bias.detach().numpy()

        w2 = model.linear_relu_stack[2].weight.detach().numpy()
        b2 = model.linear_relu_stack[2].bias.detach().numpy()

        w3 = model.linear_relu_stack[4].weight.detach().numpy()
        b3 = model.linear_relu_stack[4].bias.detach().numpy()

        w1 = np.array(w1, order='F')
        w2 = np.array(w2, order='F')
        w3 = np.array(w3, order='F')

        for i in range(200):
            for seq in range(200):
                for t in range(15):
                    y = blas.sgemv(alpha=1., a=w1, x=x) + b1
                    y = y * (y > 0)
                    y = blas.sgemv(alpha=1., a=w2, x=y) + b2
                    y = y * (y > 0)
                    y = blas.sgemv(alpha=1., a=w3, x=y) + b3

                    # y = w1 @ x + b1
                    # y = y * (y > 0)
                    # y = w2 @ y + b2
                    # y = y * (y > 0)
                    # y = w3 @ y + b3

                    # with torch.no_grad():
                    #     model.linear_relu_stack(x)

        print("--- %s seconds ---" % (time.time() - start_time))

