# ============================================================================
# utils.model
# ============================================================================

import torch; torch.set_default_dtype(torch.double)
from torch import Tensor
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
import gpytorch.settings as gpts
from gpytorch.mlls import SumMarginalLogLikelihood

from utils.func import get_kernel, get_mean


class EasyGP:
    def __init__(
        self,
        train_pairs: list[tuple[Tensor, Tensor]],
        **kwargs
    ) -> Tensor:
        gp_models = []
        for train_X, train_Y in train_pairs:
            gp_models.append(SingleTaskGP(
                train_X=train_X, train_Y=train_Y,
                covar_module=get_kernel(**kwargs["covar_module"]),
                mean_module=get_mean(**kwargs["mean_module"]),
                outcome_transform=Standardize(m=train_Y.size(-1)),
                input_transform=Normalize(d=train_X.size(-1)),
            ))
        self.gp = ModelListGP(*gp_models)
        self.mll = SumMarginalLogLikelihood(self.gp.likelihood, self.gp)
        self.d_in = ...
        self.d_out = len(self.gp.models)

    def fit(self) -> None:
        fit_gpytorch_mll(self.mll)

    def mean(
        self,
        X: Tensor,  # shape: [*batch_shape, d_in]
        idx: int | None = None,
    ) -> Tensor:  # shape: [*batch_shape, d_out] or [*batch_shape]
        self.gp.eval()
        Xf = X.flatten(0, -2); valid = ~(Xf.isnan().any(-1))
        outs = []
        for i, model in enumerate(self.gp.models):
            if idx is not None and i != idx: continue
            out = torch.full((Xf.size(0), 1), torch.nan)
            with gpts.fast_pred_var():
                posterior = model.posterior(Xf[valid])
                out[valid] = posterior.mean
            outs.append(out)
        return torch.cat(outs, dim=-1).squeeze(-1)

    def var(
        self,
        X: Tensor,  # shape: [*batch_shape, d_in]
        noise: bool = False,
        idx: int | None = None,
    ) -> Tensor:  # shape: [*batch_shape, d_out] or [*batch_shape]
        self.gp.eval()
        Xf = X.flatten(0, -2); valid = ~(Xf.isnan().any(-1))
        outs = []
        for i, model in enumerate(self.gp.models):
            if idx is not None and i != idx: continue
            out = torch.full((Xf.size(0), 1), torch.nan)
            with gpts.fast_pred_var():
                posterior = model.posterior(Xf[valid], observation_noise=noise)
                out[valid] = posterior.variance
            outs.append(out)
        return torch.cat(outs, dim=-1).squeeze(-1)

    def cov(
        self,
        X0: Tensor,  # shape: [*batch_shape0, d_in]
        X1: Tensor,  # shape: [*batch_shape1, d_in]
        noise: bool = False,
        idx: int | None = None,
    ) -> Tensor:  # shape: [*batch_shape0, *batch_shape1, d_out] or [*batch_shape0, *batch_shape1]
        self.gp.eval()
        X0f = X0.flatten(0, -2); valid = ~(X0f.isnan().any(-1))

    def rsample(
        self,
        X: Tensor,  # shape: [*batch_shape, d_in]
        n_samples: int = 1,
        noise: bool = False,
        idx: int | None = None,
    ) -> Tensor:  # shape: [n_samples, *batch_shape, d_out] or [n_samples, *batch_shape]
        self.gp.eval()
        Xf = X.flatten(0, -2); valid = ~(Xf.isnan().any(-1))
