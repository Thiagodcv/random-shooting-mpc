from unittest import TestCase
from src.control.dynamics import DynamicsModel
from src.constants import MODELS_PATH
from control.mpc import MPC
import gymnasium as gym
import numpy as np
import torch
import os

import do_mpc
import casadi
import onnx
import math


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

    def test_onnx_functionality(self):

        state_dim = 2
        action_dim = 1
        model = DynamicsModel(state_dim, action_dim)
        model.load_state_dict(torch.load(os.path.join(MODELS_PATH, "pend_demo.pt")))

        x = torch.ones(1, 2).float()
        u = torch.ones(1, 1).float()

        torch.onnx.export(model,
                          args=(x, u),
                          f="../../../models/onnx/model.onnx",
                          export_params=True,
                          input_names=["x", "u"],
                          output_names=["next_x"])

    def test_do_mpc_setup(self):

        # 1. Get dynamics model
        dynamics = onnx.load("../../../models/onnx/model.onnx")
        onnx.checker.check_model(dynamics)

        # Define model type
        model_type = "discrete"
        model = do_mpc.model.Model(model_type, 'SX')

        # Define states and inputs
        x = model.set_variable(var_type='_x', var_name='state', shape=(3, 1))
        u = model.set_variable(var_type='_u', var_name='action', shape=(1, 1))

        casadi_converter = do_mpc.sysid.ONNXConversion(dynamics)
        casadi_converter.convert(x=x.T, u=u.T)
        x_next = casadi_converter['next_x']

        model.set_rhs('state', x_next)
        model.setup()

        # 2. Get MPC
        mpc = do_mpc.controller.MPC(model)

        mpc.settings.n_horizon = 20
        mpc.settings.t_step = 1.0
        mpc.settings.supress_ipopt_output()

        def cost(x, u):
            x = x[0]
            y = x[1]
            rot_x = np.cos(-np.pi / 2) * x - np.sin(-np.pi / 2) * y
            rot_y = np.sin(-np.pi / 2) * x + np.cos(-np.pi / 2) * y
            theta = math.atan2(rot_y, rot_x)
            d_theta = x[2]

            torque = u
            return theta ** 2 + 0.1 * d_theta ** 2 + 0.001 * torque ** 2

        l = cost(model.x['state'], model.u['action'])
        mpc.set_objective(lterm=l)

        mpc.bounds['lower', '_u', 'input'] = -2
        mpc.bounds['upper', '_u', 'input'] = 2

        mpc.setup()








