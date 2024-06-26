import numpy as np
import scipy.linalg.blas as blas
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
            nn.Linear(state_dim + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, state_dim)
        )

        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(state_dim + action_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, state_dim)
        # )

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output

    def forward_np(self, state, action):
        if self.normalize:
            n_state, n_action = self.normalize_state_action(state, action)
            x = np.concatenate((n_state, n_action))
            x_torch = torch.from_numpy(x).float()
            n_next_state = self.forward(x_torch).detach().numpy() + n_state
            output = self.denormalize_state(n_next_state)
        else:
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

    def create_nn_params(self):
        # Take layers from nn.sequential
        w1 = self.linear_relu_stack[0].weight.detach().numpy()
        b1 = self.linear_relu_stack[0].bias.detach().numpy()

        w2 = self.linear_relu_stack[2].weight.detach().numpy()
        b2 = self.linear_relu_stack[2].bias.detach().numpy()

        w3 = self.linear_relu_stack[4].weight.detach().numpy()
        b3 = self.linear_relu_stack[4].bias.detach().numpy()

        # Make matrices FORTRAN-contiguous
        w1 = np.array(w1, order='F')
        w2 = np.array(w2, order='F')
        w3 = np.array(w3, order='F')

        stack = {'w1': w1,
                 'w2': w2,
                 'w3': w3,
                 'b1': b1,
                 'b2': b2,
                 'b3': b3}

        nn_params = {'state_mean': self.state_mean.detach().numpy(),
                     'state_var': self.state_var.detach().numpy(),
                     'action_mean': self.action_mean.detach().numpy(),
                     'action_var': self.action_var.detach().numpy(),
                     'stack': stack}
        return nn_params


def normalize_state_action_static(state_mean, state_var,
                                  action_mean, action_var, state, action):
    sqrt_state_var = np.sqrt(state_var)
    n_state = state - state_mean
    for j in range(n_state.shape[0]):
        n_state[j] = n_state[j] / sqrt_state_var[j]

    sqrt_action_var = np.sqrt(action_var)
    n_action = action - action_mean
    for j in range(n_action.shape[0]):
        n_action[j] = n_action[j] / sqrt_action_var[j]

    return n_state, n_action


def denormalize_state_static(state_mean, state_var, state):
    output = state
    sqrt_state = np.sqrt(state_var)
    for j in range(state.shape[0]):
        output[j] = output[j] * sqrt_state[j]
    output = output + state_mean

    return output


def forward_static(stack, x):
    w1 = stack['w1']
    w2 = stack['w2']
    w3 = stack['w3']
    b1 = stack['b1']
    b2 = stack['b2']
    b3 = stack['b3']

    y = blas.sgemv(alpha=1., a=w1, x=x) + b1
    y = y * (y > 0)
    y = blas.sgemv(alpha=1., a=w2, x=y) + b2
    y = y * (y > 0)
    y = blas.sgemv(alpha=1., a=w3, x=y) + b3

    return y


def forward_np_static(nn_params, state, action):
    state_mean = nn_params['state_mean']
    state_var = nn_params['state_var']
    action_mean = nn_params['action_mean']
    action_var = nn_params['action_var']
    stack = nn_params['stack']

    n_state, n_action = normalize_state_action_static(state_mean, state_var,
                                                      action_mean, action_var,
                                                      state, action)
    x = np.concatenate((n_state, n_action))
    n_next_state = forward_static(stack, x) + n_state
    output = denormalize_state_static(state_mean, state_var, n_next_state)
    return output
