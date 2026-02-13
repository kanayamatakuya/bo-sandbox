# ============================================================================
# problems.bench
# ============================================================================

import math
import torch; torch.set_default_dtype(torch.double)
from torch import Tensor

from problems.base import Problem


class Branin(Problem):
    n_dim = 2
    bounds = [(0.0, 1.0)] * 2

    def evaluate_true(
        self,
        X: Tensor,  # shape: [n, d_in]
    ) -> Tensor:  # shape: [n, n_obj]
        x0, x1 = 15 * X[..., 0] - 5, 15 * X[..., 1]
        t0 = x1 - (5.1 * x0**2) / (4 * math.pi**2) + (5 * x0) / math.pi - 6
        t1 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(x0)
        f = -(t0**2 + t1 - 44.81) / 51.95
        return f.reshape(*X.shape[:-1], self.n_obj)


class GoldsteinPrice(Problem):
    n_dim = 2
    bounds = [(0.0, 1.0)] * 2

    def evaluate_true(
        self,
        X: Tensor,  # shape: [n, d_in]
    ) -> Tensor:  # shape: [n, n_obj]
        x0, x1 = 4 * X[..., 0] - 2, 4 * X[..., 1] - 2
        t0 = (x0 + x1 + 1)**2
        t1 = 3 * x0**2 + 3 * x1**2 + 6 * x0 * x1 - 14 * x0 - 14 * x1 + 19
        t2 = (2 * x0 - 3 * x1)**2
        t3 = 12 * x0**2 + 27 * x1**2 - 36 * x0 * x1 - 32 * x0 + 48 * x1 + 18
        f = -(torch.log((1 + t0 * t1) * (30 + t2 * t3)) - 8.693) / 2.427
        return f.reshape(*X.shape[:-1], self.n_obj)


class SixHumpCamel(Problem):
    n_dim = 2
    bounds = [(0.0, 1.0)] * 2

    def evaluate_true(
        self,
        X: Tensor,  # shape: [n, d_in]
    ) -> Tensor:  # shape: [n, n_obj]
        x0, x1 = 6 * X[..., 0] - 3, 4 * X[..., 1] - 2
        t0 = (4 - 2.1 * x0**2 + x0**4 / 3) * x0**2
        t1 = x0 * x1
        t2 = (-4 + 4 * x1**2) * x1**2
        f = -(torch.log(t0 + t1 + t2 + 2) - 2.392) / 1.240
        return f.reshape(*X.shape[:-1], self.n_obj)


class DixonPrice(Problem):
    n_dim = 2
    bounds = [(0.0, 1.0)] * 2

    def evaluate_true(
        self,
        X: Tensor,  # shape: [n, d_in]
    ) -> Tensor:  # shape: [n, n_obj]
        x0, x1 = 20 * X[..., 0] - 10, 20 * X[..., 1] - 10
        t0 = (x0 - 1)**2
        t1 = 2 * (2 * x1**2 - x0)**2
        f = -(torch.log(t0 + t1 + 1) - 7.940) / 2.565
        return f.reshape(*X.shape[:-1], self.n_obj)
