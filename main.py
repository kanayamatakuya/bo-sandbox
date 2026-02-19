# ============================================================================
# main
# ============================================================================

import os
import yaml
import argparse
from pathlib import Path
from types import SimpleNamespace

import torch; torch.set_default_dtype(torch.double)

from runner import BORunner, BLBORunner
from problems import Problem, BLProblem, get_problem


def main(config: SimpleNamespace) -> None:
    problem = get_problem(**config.problem)
    if isinstance(problem, BLProblem):
        runner = BLBORunner(problem)
    elif isinstance(problem, BORunner):
        runner = BORunner(problem)
    else:
        print("error")
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("local", type=str)
    parser.add_argument("--seed", "-s", type=int)
    args = parser.parse_args()
    path = Path(f"experiments/config/{args.config}.yaml")
    cfg_global = yaml.safe_load(path.read_text())["global"]
    cfg_local = yaml.safe_load(path.read_text())["local"][args.local]
    config = SimpleNamespace({**cfg_global, **cfg_local})
    config.path += f"{args.local}/{str(args.seed)}"
    os.makedirs(config.path, exist_ok=True)
    main()
