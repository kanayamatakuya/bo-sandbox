# ============================================================================
# utils.config
# ============================================================================

from pathlib import Path

import torch
from torch import Tensor
from pydantic import BaseModel, ConfigDict, Field, field_validator


model_config = ConfigDict(
    extra="allow",
    arbitrary_types_allowed=True,
    validate_assignment=True,
)


class ProblemConfig(BaseModel):
    model_config = model_config
    target: str = Field(alias="_target_")
    noise_std: None | float | list[float] = None
    negate: bool = False
    dtype: torch.dtype = torch.double


class SamplingConfig(BaseModel):
    model_config = model_config
    n: int
    q: int | None = None
    batch_shape: list[int] | None = None
    seed: int | None = None
    inequality_constraints: list[tuple[Tensor, Tensor, float]] | None = None
    equality_constraints: list[tuple[Tensor, Tensor, float]] | None = None
    n_burnin: int = 10_000
    n_thinning: int = 32

    @field_validator(
        "inequality_constraints",
        "equality_constraints",
        mode="before",
    )
    @classmethod
    def parse_constraints(
        self,
        value: list[list[list[int], list[float], float]] | None,
    ) -> list[tuple[Tensor, Tensor, float]] | None:
        pass


class ModelConfig(BaseModel):
    model_config = model_config
    target: str = Field(alias="_target_")



class AcqfConfig(BaseModel):
    model_config = model_config
    target: str = Field(alias="_target_")


class OptimConfig(BaseModel):
    model_config = model_config
    q: int



class BOConfig(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    seed: int = 42
    noise: bool = True
    max_iter: int = 150
    interval: int = 1
    candidates_path: Path | None = None

    problem: ProblemConfig
    train_data: SamplingConfig
    test_data: SamplingConfig | None = None
    model: ModelConfig | list[ModelConfig]
    acqf: AcqfConfig
    optim: OptimConfig

