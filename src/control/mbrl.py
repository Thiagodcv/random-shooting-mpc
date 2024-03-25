import numpy as np
import torch
import torch.nn as nn
from src.control.dynamics import DynamicsModel
from src.control.mpc import MPC
from src.control.replay_buffer import ReplayBuffer
import os
from datetime import datetime
from src.constants import MODELS_PATH


class MBRLLearner:
    """
    A class for training a model-based reinforcement learning agent.
    """

    def __init__(self, state_dim, action_dim, env, num_episodes, episode_len,
                 reward, terminate=None, lr=1e-3, batch_size=16, train_buffer_len=2000,
                 save_name=None, normalize=False):
        """
        Parameters
        ----------
        state_dim : int
            Dimension of the state space.
        action_dim : int
            Dimension of the action space.
        env : gym.Env
        num_episodes : int
            Number of episodes to train for.
        episode_len : int
            The length of each episode.
        reward : function
            The instantaneous reward function at each timestep.
        terminate : function
            For a given (s, a, t) tuple returns true if episode has ended.
        lr : float
            Learning rate for dynamics model.
        batch_size : int
            Batch size for dynamics model training.
        train_buffer_len : int
            Number of episodes in the beginning of training where MPC not utilized (random action taken).
        save_name : str
            The name of the trained dynamics model saved.
        normalize : boolean
            If true, normalizes the data dynamics mode is trained on.
        """
        # RL Training Parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env = env
        self.num_episodes = num_episodes
        self.episode_len = episode_len
        self.eval_num = 1
        self.train_buffer_len = train_buffer_len
        self.normalize = normalize

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, normalize=self.normalize)

        # Dynamics Model Trainings Parameters
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device("cpu")  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = DynamicsModel(self.state_dim, self.action_dim, self.normalize).to(self.device)
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.save_name = save_name

        # MPC Parameters
        self.num_traj = 25  # 50
        self.gamma = 0.99
        self.horizon = 10  # 15
        self.reward = reward
        self.terminate = terminate
        self.policy = MPC(self.model, self.num_traj, self.gamma, self.horizon, self.reward, self.terminate)

    def train(self):
        """
        Train the MBRL agent.
        """
        for ep in range(self.num_episodes):
            print("Episode {}".format(ep))

            if self.replay_buffer.__len__() > self.batch_size:
                self.update_model_statistics()
                self.update_dynamics()

            o, _ = self.env.reset()
            self.policy.empty_past_action_list()
            for t in range(self.episode_len):
                # Only start MPC once a full episode has passed
                if ep > self.train_buffer_len:
                    action = self.policy.random_shooting(o)
                else:
                    action = np.random.uniform(low=-20, high=20, size=(1,))  # Pendulum
                    # action = np.random.binomial(n=1, p=0.5, size=(1,))  # Cartpole

                next_o, reward, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    break
                self.replay_buffer.push(o, action, next_o)
                o = next_o
            self.env.close()

            if ep % self.eval_num == 0 and ep > self.train_buffer_len - 1:
                self.eval_model()

        # Save trained dynamics model
        if self.save_name is None:
            now = datetime.now()
            self.save_name = now.strftime("%Y%m%d-%H%M%S")
        torch.save(self.model.state_dict(), os.path.join(MODELS_PATH, self.save_name + ".pt"))

    def update_dynamics(self):
        """
        Update the dynamics model using sampled (s,a,s'-s) triplets stored in replay_buffer.
        """
        state, action, d_state = self.replay_buffer.sample(self.batch_size)
        # input = torch.from_numpy(np.concatenate((state, action), axis=1)).float().to(self.device)
        target = torch.from_numpy(d_state).float().to(self.device)
        self.optimizer.zero_grad()
        output = self.model(torch.from_numpy(state).float().to(self.device),
                            torch.from_numpy(action).float().to(self.device))
        loss = self.loss(output, target)
        loss.backward()
        self.optimizer.step()

        # TODO: if do-mpc is being used, need to refresh onnx model after model updates
        self.policy.regen_onnx_model()

    def eval_model(self):

        o, _ = self.env.reset()
        self.policy.empty_past_action_list()
        ret = 0
        for t in range(self.episode_len):
            action = self.policy.random_shooting(o)
            next_o, reward, terminated, truncated, _ = self.env.step(action)
            ret += self.gamma**t * reward
            if terminated or truncated:
                break
            o = next_o
        self.env.close()

        print("----------------------------------------")
        print("Model Evaluation: ret = {}".format(ret))
        print("----------------------------------------")

    @staticmethod
    def static_eval_model(env, episode_len, policy, gamma):
        """
        A static version of eval_model.
        """
        o, _ = env.reset()
        ret = 0
        for t in range(episode_len):
            action = policy.random_shooting(o)
            next_o, reward, terminated, truncated, _ = env.step(action)
            ret += gamma**t * reward
            if terminated or truncated:
                break
            o = next_o
        env.close()

        print("----------------------------------------")
        print("Model Evaluation: ret = {}".format(ret))
        print("----------------------------------------")

    def update_model_statistics(self):
        self.model.update_state_var(self.replay_buffer.get_state_var())
        self.model.update_state_mean(self.replay_buffer.get_state_mean())
        self.model.update_action_var(self.replay_buffer.get_action_var())
        self.model.update_action_mean(self.replay_buffer.get_action_mean())
