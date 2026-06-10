# ============================================================================
# test_functions.synthetic
# ============================================================================

import torch
from torch import Tensor
from botorch.test_functions.synthetic import (
    SyntheticTestFunction, ConstrainedSyntheticTestFunction,
)


class Forrester(SyntheticTestFunction):

    dim = 1
    continuous_inds = list(range(dim))
    _bounds = [(0.0, 1.0)]
    _optimal_value = -6.02074
    _optimizers = [(0.75725)]

    def _evaluate_true(self, X: Tensor) -> Tensor:
        return (6 * X - 2)**2 * torch.sin(12 * X - 4)


class GoldsteinPrice(SyntheticTestFunction):

    dim = 2
    continuous_inds = list(range(dim))
    _bounds = [(-2.0, 2.0), (-2.0, 2.0)]
    _optimal_value = 3.0
    _optimizers = [(0.0, -1.0)]

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        t1 = (x1 + x2 + 1)**2
        t2 = 19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2
        t3 = (2 * x1 - 3 * x2)**2
        t4 = 18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2
        return (1 + t1 * t2) * (30 + t3 * t4)
 