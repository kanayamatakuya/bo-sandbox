# ============================================================================
# test_functions.base
# ============================================================================

from abc import ABC, abstractmethod

from botorch.test_functions.base import BaseTestProblem



class BiLevelTestProblem(BaseTestProblem, ABC):

    num_objectives: int
    num_constraints: int
    constraint_noise_std: None | float | list[float] = None
    