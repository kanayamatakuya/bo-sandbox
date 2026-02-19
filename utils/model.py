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

from utils.func import get_kernel, get_mean, naninv


class EasyGP:
    def __init__(
        self,
        train_pairs: list[tuple[Tensor, Tensor]],
        n_features: int = 500,
        **kwargs
    ) -> Tensor:
        self.n_features = n_features
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
        self.d_in = self.gp.models[0].train_inputs[0].shape[-1]
        self.d_out = len(self.gp.models)

    def fit(self) -> None:
        fit_gpytorch_mll(self.mll)
        self.xi = torch.randn(self.n_features, 1)
        self.W = torch.randn(self.d_out, self.n_features, self.d_in)
        self.b = 2 * torch.pi * torch.rand(self.d_out, self.n_features)
        self.norm, self.mean_w, self.std_w = [], [], []
        for i, model in enumerate(self.gp.models):
            X, Y = model.train_inputs[0], model.train_targets.unsqueeze(-1)
            self.W[i] /= model.covar_module.base_kernel.lengthscale.detach()
            oscale = model.covar_module.outputscale.detach()
            self.norm.append((2 * oscale / self.n_features)**0.5)
            phi = self.norm[i] * torch.cos(X @ self.W[i].T + self.b[i])
            noise = model.likelihood.noise
            inv = naninv(phi.T @ phi / noise + torch.eye(self.n_features))
            self.mean_w.append(inv @ phi.T @ Y / noise)
            self.std_w.append(inv)

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
        X0f, X1f = X0.flatten(0, -2), X1.flatten(0, -2)
        v0, v1 = ~(X0f.isnan().any(-1)), ~(X1f.isnan().any(-1))
        i0, i1 = v0.nonzero(as_tuple=True)[0], v1.nonzero(as_tuple=True)[0]
        outs = []
        for i, model in enumerate(self.gp.models):
            if idx is not None and i != idx: continue
            out = torch.full((X0f.size(0), X1f.size(0), 1), torch.nan)
            Xt = model.train_inputs[0]
            X0t, X1t = model.input_transform(X0f), model.input_transform(X1f)
            K01 = model.covar_module(X0t[v0], X1t[v1]).to_dense()
            K0t = model.covar_module(X0t[v0], Xt).to_dense()
            Kt1 = model.covar_module(Xt, X1t[v1]).to_dense()
            Ktt = model.covar_module(Xt, Xt).to_dense()
            Ktt += model.likelihood.noise * torch.eye(Ktt.size(0))
            L = torch.linalg.cholesky(Ktt + 1e-6 * torch.eye(Ktt.size(0)))
            A = torch.linalg.solve_triangular(L, Kt1, upper=False)
            C = torch.linalg.solve_triangular(L, K0t.T, upper=False)
            cov = (K01 - C.T @ A)
            if noise:
                same = (X0f[v0][:, None, :] == X1f[v1][None, :, :]).all(-1)
                cov += model.likelihood.noise * same
            cov *= model.outcome_transform.stdvs**2
            out[i0[:, None], i1[None, :]] = cov
            outs.append(out)
        return torch.cat(outs, dim=-1).squeeze(-1)

    def rsample_(
        self,
        X: Tensor,  # shape: [*batch_shape, d_in]
        n_samples: int = 1,
        noise: bool = False,
        idx: int | None = None,
    ) -> Tensor:  # shape: [n_samples, *batch_shape, d_out] or [n_samples, *batch_shape]
        self.gp.eval()
        Xf = X.flatten(0, -2); valid = ~(Xf.isnan().any(-1))
        base_sample_shape = (n_samples, Xf.size(0), self.d_out)

    def rsample(
        self,
        X: Tensor,  # shape: [*batch_shape, d_in]
        n_samples: int = 1,
        noise: bool = False,
        idx: int | None = None,
    ) -> Tensor:  # shape: [n_samples, *batch_shape, d_out] or [n_samples, *batch_shape]
        self.gp.eval()
        Xf = X.flatten(0, -2); valid = ~(Xf.isnan().any(-1))
        base_sample_shape = (n_samples, Xf.size(0), self.d_out)
        


class EasyBLGP(EasyGP):
    def __init__(
        self,
        train_pairs: list[tuple[Tensor, Tensor]],
        n_dim: list[int],
        n_features: int = 500,
        **kwargs
    ) -> Tensor:
        self.n_dim = n_dim
        self.n_features = n_features
        kwargs["covar_module"]["base_kernel"]["kernels"][0].update(
            active_dims=list(range(0, self.n_dim[0])),
        )
        kwargs["covar_module"]["base_kernel"]["kernels"][1].update(
            active_dims=list(range(self.n_dim[0], sum(self.n_dim))),
        )
        gp_models = []
        for train_X, train_Y in train_pairs:
            gp_models.append(SingleTaskGP(
                train_X=train_X,
                train_Y=train_Y,
                covar_module=get_kernel(**kwargs["covar_module"]),
                mean_module=get_mean(**kwargs["mean_module"]),
                outcome_transform=Standardize(m=train_Y.size(-1)),
                input_transform=Normalize(d=train_X.size(-1)),
            ))
        self.gp = ModelListGP(*gp_models)
        self.mll = SumMarginalLogLikelihood(self.gp.likelihood, self.gp)
        self.d_in = sum(n_dim)
        self.d_out = len(self.gp.models)

    def fit(self) -> None:
        fit_gpytorch_mll(self.mll)

    def rsample(
        self,
        X: Tensor,  # shape: [*batch_shape, d_in]
        n_samples: int = 1,
        noise: bool = False,
        idx: int | None = None,
    ) -> Tensor:  # shape: [n_samples, *batch_shape, d_out] or [n_samples, *batch_shape]
        self.gp.eval()
        