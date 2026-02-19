# ============================================================================
# acquisitions.base
# ============================================================================

from abc import ABC, abstractmethod

import torch; torch.set_default_dtype(torch.double)
from torch import Tensor

from problems import Problem
from utils.model import EasyGP


class Acquisition(ABC):
    def __init__(
        self,
        problem: Problem,
        model: EasyGP,
    ) -> None:
        self.problem = problem
        self.model = model
        
    def optimize(
        self,
        **kwargs
    ) -> tuple[Tensor, Tensor]:
        if self.problem.has_cands:
            new_X, mask_Y = self.optimize_pool(
                cands=self.problem.cands,
                mask=self.problem.mask,
                **kwargs
            )
        else:
            new_X, mask_Y = self.optimize_query(
                bounds=self.problem.bounds,
                **kwargs
            )
        return new_X, mask_Y
    
    @abstractmethod
    def optimize_pool(
        self,
        cands: Tensor,  # shape: [N, d_in]
        mask: Tensor,  # shape: [N, d_out]
        **kwargs
    ) -> tuple[Tensor, Tensor]:
        return NotImplementedError

    @abstractmethod
    def optimize_query(
        self,
        bounds: Tensor,  # shape: [2, d_in]
        **kwargs
    ) -> tuple[Tensor, Tensor]:
        return NotImplementedError
