from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pybnn.bohamiann import Bohamiann
from pybnn.util.layers import AppendLayer

def pow3(x, a, b, c):
    return a/(b-x)**c


def logpow(x, a, b, c):
    return a*(b/x-1)**c


def logpow2(x, a, b, c):
    return -a*(b/x-1)**(-c)


def pow4(x, a, b, c, d):
    return a/(b-x)**c + d


def MMF(x, a, b, c, d):
    return (((b-a)/(b-x)-1)/c) ** d

# doesn't make sense
def exp3(x, a, b, c): # exp4 didn't converge
    return a*torch.log(b-x) + c


def Janoschek(x, a, b, c, d): # same as Weibull
    return a*(torch.log((b-c)/(b-x)))**d


def ilog2(x, a, b, c):
    return a*torch.log(b-x)+c

def rat(x, a, b): ## not in Domhan
    return (b-a)/(b-x)

def exp2(x, a, b, c, d): ## not in Domhan
    return a*torch.exp(b*x) + c*torch.exp(d*x)

#doesn't make sense
def pow2(x, a, b, c): ## not in Domhan
    return a * x**b + c

# a,d>0, c in [0,1], b fixed upper bound (2500)
def lin_decay(x, a, b, c, d): 
    return a/(b-c*x)**d

# a,d>0, c in [0,1], b fixed upper bound (2500)
def log_decay(x, a, b, c, d): 
    return a*(torch.log(d/(b-c*x)))

# a,d>0, c in [0,1], b fixed upper bound (2500)
def exp_decay(x, a, b, c, d): 
    return a*(torch.exp(-(b-c*x)**d))


def bf_layer(theta, t):

    # y_a = logpow(t, 0.04545, 1115.7914, -0.5220)

    # y_b = logpow(t, 0.04545, 1115.7914, -0.5220)

    # y_c = logpow(t, 0.04545, 1115.7914, -0.5220)

    # y_a = logpow(t, theta[:, 0], t+theta[:, 1], theta[:, 2])

    # y_b = logpow(t, theta[:, 3], t+theta[:, 4], theta[:, 5])

    # y_c = logpow(t, theta[:, 6], t+theta[:, 7], theta[:, 8])

    # y_a = logpow(t, theta[:, 0], t+theta[:, 6], theta[:, 1])
    y_a = lin_decay(t, theta[:, 0], 2500, theta[:, 6], theta[:, 1])

    y_b = log_decay(t, theta[:, 2], 2500, theta[:, 7], theta[:, 3])

    y_c = exp_decay(t, theta[:, 4], 2500, theta[:, 8], theta[:, 5])

    # y_a = rat2(t, 1, 2500, theta[:, 6])

    # y_b = rat2(t, 1, 2500, theta[:, 7])

    # y_c = rat2(t, 1, 2500, theta[:, 8])

    # print(torch.max(y_a).detach().numpy(), torch.min(y_a).detach().numpy())
    if (y_a != y_a).any():
        import pdb
        pdb.set_trace()

    if torch.max(y_a).detach().numpy() == float('inf'):
        import pdb
        pdb.set_trace()

    if (y_b != y_b).any():
        import pdb
        pdb.set_trace()

    if torch.max(y_b).detach().numpy() == float('inf'):
        import pdb
        pdb.set_trace()

    if (y_c != y_c).any():
        import pdb
        pdb.set_trace()

    if torch.max(y_c).detach().numpy() == float('inf'):
        import pdb
        pdb.set_trace()

    return torch.stack([y_a, y_b, y_c], dim=1)

def bf_layer_test(theta, t):

    y_a = t

    y_b = t

    y_c = t

    if (y_b != y_b).any():
        import pdb
        pdb.set_trace()

    return torch.stack([y_a, y_b, y_c], dim=1)

# def vapor_pressure(x, a, b, c, *args):
#     b_ = (b + 1) / 2 / 10
#     a_ = (a + 1) / 2
#     c_ = (c + 1) / 2 / 10
#     return torch.exp(-a_ - b_ / (x + 1e-5) - c_ * torch.log(x)) - (torch.exp(a_ + b_))


# def log_func(t, a, b, c, *args):
#     a_ = (a + 1) / 2 * 5
#     b_ = (b + 1) / 2
#     c_ = (c + 1) / 2 * 10
#     return (c_ + a_ * torch.log(b_ * t + 1e-10)) / 10.


# def hill_3(x, a, b, c, *args):
#     a_ = (a + 1) / 2
#     b_ = (b + 1) / 2
#     c_ = (c + 1) / 2 / 100
#     return a_ * (1. / ((c_ / x + 1e-5) ** b_ + 1.))


# def bf_layer(theta, t):
#     y_a = vapor_pressure(t, theta[:, 0], theta[:, 1], theta[:, 2])

#     y_b = log_func(t, theta[:, 3], theta[:, 4], theta[:, 5])

#     y_c = hill_3(t, theta[:, 6], theta[:, 7], theta[:, 8])

#     return torch.stack([y_a, y_b, y_c], dim=1)


def get_lc_net_architecture(input_dimensionality: int) -> torch.nn.Module:
    class Architecture(nn.Module):
        def __init__(self, n_inputs, n_hidden=100):
            super(Architecture, self).__init__()
            self.fc1 = nn.Linear(n_inputs - 1, n_hidden)
            self.fc2 = nn.Linear(n_hidden, n_hidden)
            self.fc3 = nn.Linear(n_hidden, n_hidden)
            self.theta_fc1 = nn.Linear(n_hidden, n_hidden)
            self.theta_fc2 = nn.Linear(n_hidden, n_hidden)
            self.theta_layer_c = nn.Linear(n_hidden, 3)
            self.theta_layer_rest = nn.Linear(n_hidden, 6)

            self.weight_fc1 = nn.Linear(n_hidden, n_hidden)
            self.weight_fc2 = nn.Linear(n_hidden, n_hidden)
            self.weight_layer = nn.Linear(n_hidden, 3)
            self.asymptotic_fc1 = nn.Linear(n_hidden, n_hidden)
            self.asymptotic_fc2 = nn.Linear(n_hidden, n_hidden)
            self.asymptotic_layer = nn.Linear(n_hidden, 1)
            self.sigma_layer = AppendLayer(noise=1e-3)

        def forward(self, input):
            x = input[:, :-1]
            t = input[:, -1]
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
            theta_x = torch.tanh(self.theta_fc1(x))
            theta_x = torch.tanh(self.theta_fc2(theta_x))
            theta_c = torch.sigmoid(self.theta_layer_c(theta_x))
            theta_rest = F.softplus(self.theta_layer_rest(theta_x))
            theta = torch.cat([theta_rest, theta_c], dim=-1)
            if (theta_x != theta_x).any():
                import pdb
                pdb.set_trace()

            # bf = bf_layer_test(theta, t)
            bf = bf_layer(theta, t)

            weight_x = torch.tanh(self.weight_fc1(x))
            weight_x = torch.tanh(self.weight_fc2(weight_x))
            weights = torch.softmax(self.weight_layer(weight_x), -1)
            residual = torch.relu(torch.sum(bf * weights, dim=(1,), keepdim=True))

            asymptotic_x = torch.tanh(self.asymptotic_fc1(x))
            asymptotic_x = torch.tanh(self.asymptotic_fc2(asymptotic_x))
            asymptotic = torch.relu(self.asymptotic_layer(asymptotic_x))

            mean = torch.clamp(residual + asymptotic, 0, 1.1)
            # mean = asymptotic
            # mean = residual
            rev_mean = 1.1 - mean
            # print(torch.max(rev_mean).detach().numpy(), torch.min(rev_mean).detach().numpy())
            # import pdb
            # pdb.set_trace()
            return self.sigma_layer(mean)

    return Architecture(n_inputs=input_dimensionality)


class LCNet(Bohamiann):
    def __init__(self, normalize_input=True, normalize_output=False, **kwargs) -> None:
        super(LCNet, self).__init__(get_network=get_lc_net_architecture,
                                    normalize_input=normalize_input,
                                    normalize_output=normalize_output,
                                    **kwargs)

    @staticmethod
    def normalize_input(x, m=None, s=None):
        if m is None:
            m = np.mean(x, axis=0)
        if s is None:
            s = np.std(x, axis=0)

        x_norm = deepcopy(x)
        x_norm[:, :-1] = (x[:, :-1] - m[:-1]) / s[:-1]

        return x_norm, m, s
