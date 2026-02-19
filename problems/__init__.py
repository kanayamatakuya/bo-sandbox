# ============================================================================
# problems
# ============================================================================

import inspect

from problems.base import Problem
from problems.bilevel.base import BLProblem
from problems.bench import (
    Forrester, Branin, GoldsteinPrice, SixHumpCamel, DixonPrice,
)
from problems.bilevel.bench import (
    BraninGoldstein, CamelBranin, DixonBranin,
)


def get_problem(
    name: str,
    noise_std: float | list[float] | None = None,
    has_cands: bool = False,
    n_disc: int | list[int] | None = None,
    **kwargs
) -> Problem:
    problems: dict[str, Problem] = {
        "Forrester": Forrester,
        "Branin": Branin,
        "GoldsteinPrice": GoldsteinPrice,
        "SixHumpCamel": SixHumpCamel,
        "DixonPrice": DixonPrice,
        ### BilevelProblem ###
        "BraninGoldstein": BraninGoldstein,
        "CamelBranin": CamelBranin,
        "DixonBranin": DixonBranin,
    }
    problem_cls = problems[name]
    sig = inspect.signature(problem_cls.__init__).parameters.values()
    args = [p.name for p in sig if p.name != "self"]
    params = {
        "noise_std": noise_std,
        "has_cands": has_cands,
        "n_disc": n_disc,
    }
    params |= {k: v for k, v in kwargs.items() if k in args}
    return problem_cls(**params)


__all__ = [
    "Problem",
    "BLProblem",
    "get_problem",
]