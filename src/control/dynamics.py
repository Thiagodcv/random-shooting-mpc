import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, normalize=False):
        """
        Parameters
        ----------
        state_dim : int
            Dimension of states.
        action_dim : int
            Dimension of actions.
        normalize : boolean
            Normalize data.
        """
        # super(DynamicsModel, self).__init__()
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_var = nn.Parameter(torch.ones(state_dim), requires_grad=False)
        self.state_mean = nn.Parameter(torch.zeros(state_dim), requires_grad=False)
        self.action_var = nn.Parameter(torch.ones(action_dim), requires_grad=False)
        self.action_mean = nn.Parameter(torch.zeros(action_dim), requires_grad=False)
        self.normalize = normalize

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )

    def forward(self, s, a):
        """
        Parameters
        ----------
        s: torch.Tensor
            Normalized state
        a: torch.Tensor
            Normalized action
        """
        x = torch.cat((s, a), dim=1)
        output = self.linear_relu_stack(x)
        return output

    def forward_np(self, state, action):
        if self.normalize:
            n_state, n_action = self.normalize_state_action(state, action)
            # x = np.concatenate((n_state, n_action))
            # x_torch = torch.from_numpy(x).float()
            s = torch.from_numpy(n_state).float()[None, :]
            a = torch.from_numpy(n_action).float()[None, :]
            n_next_state = self.forward(s, a).detach().numpy().flatten() + n_state
            output = self.denormalize_state(n_next_state)
        else:
            # TODO: Make non-normalized version compatible with new forward() implementation
            x = np.concatenate((state, action))
            x_torch = torch.from_numpy(x).float()
            output = self.forward(x_torch).detach().numpy() + state

        return output

    def update_state_var(self, state_var):
        self.state_var = nn.Parameter(torch.from_numpy(state_var))

    def update_state_mean(self, state_mean):
        self.state_mean = nn.Parameter(torch.from_numpy(state_mean))

    def update_action_var(self, action_var):
        self.action_var = nn.Parameter(torch.from_numpy(action_var))

    def update_action_mean(self, action_mean):
        self.action_mean = nn.Parameter(torch.from_numpy(action_mean))

    def normalize_state_action(self, state, action):
        state_mean = self.state_mean.detach().numpy()
        state_var = self.state_var.detach().numpy()
        action_mean = self.action_mean.detach().numpy()
        action_var = self.action_var.detach().numpy()
        n_state = (state - state_mean) @ np.diagflat(np.reciprocal(np.sqrt(state_var)))
        n_action = (action - action_mean) @ np.diagflat(np.reciprocal(np.sqrt(action_var)))
        return n_state, n_action

    def denormalize_state(self, state):
        state_mean = self.state_mean.detach().numpy()
        state_var = self.state_var.detach().numpy()
        output = state @ np.diagflat(np.sqrt(state_var)) + state_mean
        return output
