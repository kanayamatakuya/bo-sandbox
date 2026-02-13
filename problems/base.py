# ============================================================================
# problems.base
# ============================================================================

from pathlib import Path
from abc import ABC, abstractmethod

import torch; torch.set_default_dtype(torch.double)
from torch import Tensor

from utils.func import make_grid


class Problem(ABC):
    """Abstract base class for optimization problems."""
    n_dim: int = 1
    n_obj: int = 1
    n_con: int = 0
    bounds: list[tuple[float, float]] = [(0.0, 1.0)]
    cands_path: Path | None = None
    outs_path: Path | None = None

    def __init__(
        self,
        noise_std: float | list[float] | None = None,
        has_cands: bool = False,
        n_disc: int | list[int] | None = None,
    ) -> None:
        self.noise_std = noise_std
        self.has_cands = has_cands
        self.n_disc = n_disc
        self.split_list = self.n_obj + self.n_con
        self.d_in = self.n_dim
        self.d_out = sum(self.split_list)
        if self.has_cands:
            if self.cands_path is not None:
                self.cands = torch.load(self.cands_path, weights_only=False)
            else:
                self.cands = make_grid(self.n_dim, self.n_disc, self.bounds)
            if self.outs_path is not None:
                self.outs = torch.load(self.outs_path, weights_only=False)
            shape = self.cands.shape[:-1]
            self.mask = torch.zeros(*shape, self.d_out, dtype=torch.bool)
    
    def __call__(
        self,
        X: Tensor,  # shape: [n, d_in]
        noise: bool = True,
    ) -> Tensor:  # shape: [n, d_out]
        F = self.evaluate_true(X=X)
        C = self.evaluate_slack(X=X)
        out = torch.cat([F, C], dim=-1)
        if noise and self.noise_std is not None:
            out += torch.tensor(self.noise_std) * torch.randn_like(out)
        return out

    @abstractmethod
    def evaluate_true(
        self,
        X: Tensor,  # shape: [n, d_in]
    ) -> Tensor:  # shape: [n, n_obj]
        raise NotImplementedError

    def evaluate_slack(
        self,
        X: Tensor,  # shape: [n, d_in]
    ) -> Tensor:  # shape: [n, n_con]
        return torch.empty(*X.shape[:-1], self.n_con)

