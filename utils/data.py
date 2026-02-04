# ============================================================================
# utils.data
# ============================================================================

from pathlib import Path

import torch; torch.set_default_dtype(torch.double)
from torch import Tensor


class DataCollection:
    def __init__(
        self,
        inputs: Tensor,  # shape: [n_init, d_in]
        outputs: Tensor,  # shape: [n_init, d_out]
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs

    def __str__(self) -> str:
        pass

    def add(
        self,
        inputs: Tensor,  # shape: [n, d_in]
        outputs: Tensor,  # shape: [n, d_out]
    ) -> None:
        self.inputs = torch.cat([self.inputs, inputs], dim=0)
        self.outputs = torch.cat([self.outputs, outputs], dim=0)

    def get(
        self,
        idx: int | None = None,
    ) -> list[tuple[Tensor, Tensor]]:
        train_pairs = []
        for i in range(self.outputs.size(-1)):
            if idx is not None and i != idx: continue
            mask = ~self.outputs[:, i].isnan()
            train_X = self.inputs[mask, :]
            train_Y = self.outputs[mask, i:i+1]
            train_pairs.append((train_X, train_Y))
        return train_pairs

    def save(
        self,
        dir: Path,
    ) -> None:
        torch.save(self.inputs, dir / "inputs.pt")
        torch.save(self.outputs, dir / "outputs.pt")

    @classmethod
    def load(
        cls,
        dir: Path,
    ) -> "DataCollection":
        inputs = torch.load(dir / "inputs.pt", weights_only=False)
        outputs = torch.load(dir / "outputs.pt", weights_only=False)
        return cls(inputs, outputs)
