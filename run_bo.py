# ============================================================================
# run_bo.py
# ============================================================================

import argparse
from typing import Any
from pathlib import Path

import yaml
import numpy as np
import torch
from torch import Tensor
from botorch.models.model import Model
from gpytorch.mlls import MarginalLogLikelihood

from utils import BOConfig


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


class BayesianOptimizer:

    def __init__(self, args: argparse.Namespace) -> None:
        self.exp_path = Path(f"experiments/{args.id}")
        self.local_path = self.exp_path / args.local / str(args.seed)
        self.config = self._get_config(args=args)
        self.logger = self._get_logger(args=args)
        self._set_seed(self.config.seed)
        self.problem = self.config.problem
        self.current_iter: int = 0
        self.train_X: Tensor = torch.empty((0, self.problem.dim))
        self.train_Y: Tensor = torch.empty((0, ))
        self.test_X: Tensor | None = None
        self.test_Y: Tensor | None = None
        self.rng: dict[str, Any] = {
            "numpy": np.random.get_state(),
            "cpu": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state()
                     if torch.cuda.is_available() else None),
        }
        
    def _get_config(self, args: argparse.Namespace) -> BOConfig:
        cfg_path = self.exp_path / "config.yaml"
        with open(cfg_path, mode="r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f)
        config = BOConfig(**{**cfg_dict["global"], **cfg_dict[args.local]})
        config.id, config.local = args.id, args.local
        config.train_data.seed = args.seed
        return config
    
    def _get_logger(self, args: argparse.Namespace):
        pass

    def _set_seed(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _save_checkpoint(self) -> None:
        pass

    def _load_checkpoint(self) -> None:
        pass

    def _proceed_iter(self) -> None:
        self._save_checkpoint()
        self.current_iter += 1
        

    def _get_observation(self) -> Tensor:
        pass

    def _get_initial_points(self) -> Tensor:
        pass

    def _initialize_model(self) -> tuple[MarginalLogLikelihood, Model]:
        pass

    def _evaluate_model(self) -> None:
        pass

    def _optimize_acqf(self) -> tuple[Tensor, Tensor]:
        pass

    def run(self) -> None:
        pass



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("id", type=str)
    parser.add_argument("local", type=str)
    parser.add_argument("seed", type=int)
    parser.add_argument("--checkpoint", type=str)
    args = parser.parse_args()

