from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from pybnn.bohamiann_domhan import Bohamiann


def vapor_pressure(x, a, b, c, *args):
    b_ = (b + 1) / 2 / 10
    a_ = (a + 1) / 2
    c_ = (c + 1) / 2 / 10
    return torch.exp(-a_ - b_ / (x + 1e-5) - c_ * torch.log(x)) - (torch.exp(a_ + b_))


def log_func(t, a, b, c, *args):
    a_ = (a + 1) / 2 * 5
    b_ = (b + 1) / 2
    c_ = (c + 1) / 2 * 10
    return (c_ + a_ * torch.log(b_ * t + 1e-10)) / 10.


def hill_3(x, a, b, c, *args):
    a_ = (a + 1) / 2
    b_ = (b + 1) / 2
    c_ = (c + 1) / 2 / 100
    return a_ * (1. / ((c_ / x + 1e-5) ** b_ + 1.))


def bf_layer(theta, t):
    y_a = vapor_pressure(t, theta[:, 0], theta[:, 1], theta[:, 2])

    y_b = log_func(t, theta[:, 3], theta[:, 4], theta[:, 5])

    y_c = hill_3(t, theta[:, 6], theta[:, 7], theta[:, 8])

    return torch.stack([y_a, y_b, y_c], dim=1)


def get_lc_net_architecture(n_curves: int) -> torch.nn.Module:
    class Architecture(nn.Module):
        def __init__(self, num_datapoints, n_hidden=50):
            super(Architecture, self).__init__()
            self.theta_logits = nn.Parameter(torch.empty(num_datapoints, 9))
            self.weight_logits = nn.Parameter(torch.empty(num_datapoints, 3))
            self.sigma_logits = nn.Parameter(torch.empty(num_datapoints, 1))

            nn.init.xavier_uniform_(self.theta_logits, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.weight_logits, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_uniform_(self.sigma_logits, gain=nn.init.calculate_gain('relu'))

        def forward(self, input):
            t = input[:, -1]
            theta = torch.tanh(self.theta_logits)
            bf = bf_layer(theta, t)
            weights = torch.softmax(self.weight_logits, -1)
            mean = torch.sum(bf * weights, dim=(1,), keepdim=True)
            log_var = self.sigma_logits
            return torch.cat((mean, log_var), dim=1)

    return Architecture(num_datapoints=n_curves)


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