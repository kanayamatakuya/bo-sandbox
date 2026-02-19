# ============================================================================
# runner
# ============================================================================

from types import SimpleNamespace

import torch; torch.set_default_dtype(torch.double)
from torch import Tensor
from botorch.utils.transforms import unnormalize

from acquisitions import Acquisition, BLAcquisition
from problems import Problem, BLProblem
from utils.data import DataCollection
from utils.model import EasyGP, EasyBLGP


class BORunner:
    def __init__(
        self,
        problem: Problem,
        config: SimpleNamespace,
    ) -> None:
        self.problem = problem
        self.config = config

    def run(self) -> DataCollection:
        pass


class BLBORunner(BORunner):
    def __init__(
        self,
        problem: BLProblem,
        config: SimpleNamespace,
    ) -> None:
        super().__init__(problem)

    def run(self) -> DataCollection:
        pass

    def generate_initial_data(self) -> tuple[Tensor, Tensor]:
        if self.problem.has_cands:
            valid = (~self.problem.cands.isnan().any(-1)).nonzero()
            idx = valid[torch.randperm(len(valid))[:self.config.n_init]]
            init_X = self.problem.cands[idx[:, 0], idx[:, 1]]
            self.problem.mask[idx[:, 0], idx[:, 1]] = True
        else:
            bounds = torch.tensor(self.problem.bounds).T
            init_X = torch.rand(self.config.n_init, bounds.size(-1))
            init_X = unnormalize(init_X, bounds)
        init_Y = self.problem(init_X, noise=self.config.noise)
        return init_X, init_Y

    def fit_model(
        self,
        train_list: list[list[tuple[Tensor, Tensor]]],
    ) -> dict[str, EasyBLGP | None]:
        model_F_ul = EasyBLGP(train_list[0], **self.config.model_F_ul).fit()
        model_F_ll = EasyBLGP(train_list[1], **self.config.model_F_ll).fit()
        model_C_ul, model_C_ll = None, None
        if len(train_list[2]) > 0:
            model_C_ul = EasyBLGP(train_list[2], **self.config.model_C_ul).fit()
        if len(train_list[3]) > 0:
            model_C_ll = EasyBLGP(train_list[3], **self.config.model_C_ll).fit()
        model_dict = {"model_F_ul": model_F_ul, "model_F_ll": model_F_ll,
                      "model_C_ul": model_C_ul, "model_C_ll": model_C_ll,}
        return model_dict

    def optimize_acqf_and_get_obs(
        self,
        acqf: BLAcquisition
    ) -> tuple[Tensor, Tensor]:
        pass


"""
class MultiFidelityBORunner:
    def __init__(
        self,
        problem: MultiFidelityProblem,
        dataset: DataCollection,
    ) -> None:
        pass
"""