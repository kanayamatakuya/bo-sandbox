# ============================================================================
# utils.func
# ============================================================================

import inspect

import torch; torch.set_default_dtype(torch.double)
from torch import Tensor
from gpytorch.constraints import Interval, Positive, LessThan, GreaterThan
from gpytorch.means import Mean, ZeroMean, ConstantMean, LinearMean
from gpytorch.priors import (
    Prior, NormalPrior, HalfNormalPrior, LogNormalPrior, UniformPrior,
    HalfCauchyPrior, GammaPrior, SmoothedBoxPrior, HorseshoePrior,
)
from gpytorch.kernels import (
    Kernel, ConstantKernel, CosineKernel, CylindricalKernel, LinearKernel,
    MaternKernel, PeriodicKernel, PiecewisePolynomialKernel, PolynomialKernel,
    RBFKernel, RQKernel, SpectralDeltaKernel, SpectralMixtureKernel,
    AdditiveKernel, AdditiveStructureKernel, ProductKernel,
    ProductStructureKernel, ScaleKernel, 
)


def atleast(
    n: int,
    input: Tensor,
    dim: int | None = None,
) -> Tensor:
    """Make tensor at least n-D by adding singleton dimensions."""
    out = input.clone()
    while out.ndim < n: out = out.unsqueeze(0 if dim is None else dim)
    return out


def nanmax(
    input: Tensor,
    nan: float | None = None,
    dim: int | None = None,
    keepdim: bool = False,
) -> Tensor:
    out = input.clone()
    if nan is not None: out = torch.where(~out.isnan(), out, nan)
    else: out = torch.where(~out.isnan(), out, -torch.inf)
    return out.max(dim, keepdim)


def nanmin(
    input: Tensor,
    nan: float | None = None,
    dim: int | None = None,
    keepdim: bool = False,
) -> Tensor:
    out = input.clone()
    if nan is not None: out = torch.where(~out.isnan(), out, nan)
    else: out = torch.where(~out.isnan(), out, torch.inf)
    return out.min(dim, keepdim)


def nanmean(
    input: Tensor,
    nan: float | None = None,
    dim: int | None = None,
    keepdim: bool = False,
) -> Tensor:
    out = input.clone()
    if nan is not None: out = torch.where(~out.isnan(), out, nan)
    return out.nanmean(dim, keepdim)


def nanstd(
    input: Tensor,
    nan: float | None = None,
    unbiased: bool = True,
    dim: int | None = None,
    keepdim: bool = False,
) -> Tensor:
    out = input.clone()
    if nan is not None: out = torch.where(~out.isnan(), out, nan)
    rss = ((out - out.nanmean(dim, keepdim))**2).nansum(dim, keepdim)
    std = (rss / (~out.isnan()).sum(dim, keepdim) - float(unbiased)).sqrt()
    return std

def nanargmax(
    input: Tensor,
    nan: float | None = None,
    dim: int | None = None,
    keepdim: bool = False,
) -> Tensor:
    out = input.clone()
    if nan is not None: out = torch.where(~out.isnan(), out, nan)
    else: out = torch.where(~out.isnan(), out, -torch.inf)
    return out.argmax(dim, keepdim)


def nanargmin(
    input: Tensor,
    nan: float | None = None,
    dim: int | None = None,
    keepdim: bool = False,
) -> Tensor:
    out = input.clone()
    if nan is not None: out = torch.where(~out.isnan(), out, nan)
    else: out = torch.where(~out.isnan(), out, torch.inf)
    return out.argmin(dim, keepdim)


def nanpdf(
    value: Tensor,
) -> Tensor:
    out = torch.where(~value.isnan(), out, 0.0)
    pdf = torch.exp(-0.5 * out**2) / (2.0 * torch.pi)**0.5
    return torch.where(~value.isnan(), pdf, torch.nan)


def nancdf(
    value: Tensor,
) -> Tensor:
    out = torch.where(~value.isnan(), out, 0.0)
    cdf = 0.5 * (1 + torch.erf(out) / 2.0**0.5)
    return torch.where(~value.isnan(), cdf, torch.nan)


def naninv(
    input: Tensor,
    jitter: float = 1e-6,
    max_tries: int = 4,
) -> Tensor:
    mat = atleast(3, input, dim=0).flatten(0, -3)
    mask = mat.isnan().any(dim=(-2, -1))
    mask |= (torch.linalg.eigvalsh(mat).min(-1).values < 0.0)
    inv = torch.full_like(mat, float("nan"))
    valid = (~mask).nonzero(as_tuple=True)[0]
    if len(valid) == 0: return inv.reshape(*input.shape)
    for i in range(max_tries):
        jitter_i = jitter * 10**i * torch.eye(mat.size(-1))
        try:
            L = torch.linalg.cholesky(mat[valid] + jitter_i)
            inv[valid] = torch.cholesky_inverse(L); break
        except RuntimeError: continue
    return inv.reshape(*input.shape)


def make_grid(
    n_dim: int | list[int],
    n_disc: int | list[int],
    bounds: list[tuple[float, float]],
) -> Tensor:
    pass


def get_constraint(
    name: str,
    **kwargs
) -> Interval:
    constraints: dict[str, Interval] = {
        "Interval": Interval,
        "Positive": Positive,
        "LessThan": LessThan,
        "GreaterThan": GreaterThan,
    }
    constraint_cls = constraints[name]
    sig = inspect.signature(constraint_cls.__init__).parameters.values()
    args = [param.name for param in sig if param.name != "self"]
    params = {k: v for k, v in kwargs.items() if k in args}
    return constraint_cls(**params)


def get_prior(
    name: str,
    **kwargs
) -> Prior:
    priors: dict[str, Prior] = {
        "NormalPrior": NormalPrior,
        "HalfNormalPrior": HalfNormalPrior,
        "LogNormalPrior": LogNormalPrior,
        "UniformPrior": UniformPrior,
        "HalfCauchyPrior": HalfCauchyPrior,
        "GammaPrior": GammaPrior,
        "SmoothedBoxPrior": SmoothedBoxPrior,
        "HorseshoePrior": HorseshoePrior,
    }
    prior_cls = priors[name]
    sig = inspect.signature(prior_cls.__init__).parameters.values()
    args = [param.name for param in sig if param.name != "self"]
    params = {k: v for k, v in kwargs.items() if k in args}
    return prior_cls(**params)


def get_mean(
    name: str,
    **kwargs
) -> Mean:
    means: dict[str, Mean] = {
        "ZeroMean": ZeroMean,
        "ConstantMean": ConstantMean,
        "LinearMean": LinearMean,
    }
    mean_cls = means[name]
    sig = inspect.signature(mean_cls.__init__).parameters.values()
    args = [param.name for param in sig if param.name != "self"]
    params = {}
    for k, v in kwargs.items():
        if k in args:
            if k.endswith("constraint"): params[k] = get_constraint(**v)
            elif k.endswith("prior"): params[k] = get_prior(**v)
            else: params[k] = v
    return mean_cls(**params)


def get_kernel(
    name: str,
    **kwargs
) -> Kernel:
    kernels: dict[str, Kernel] = {
        "ConstantKernel": ConstantKernel,
        "CosineKernel": CosineKernel,
        "CylindricalKernel": CylindricalKernel,
        "LinearKernel": LinearKernel,
        "MaternKernel": MaternKernel,
        "PeriodicKernel": PeriodicKernel,
        "PiecewisePolynomialKernel": PiecewisePolynomialKernel,
        "PolynomialKernel": PolynomialKernel,
        "RBFKernel": RBFKernel,
        "RQKernel": RQKernel,
        "SpectralDeltaKernel": SpectralDeltaKernel,
        "SpectralMixtureKernel": SpectralMixtureKernel,
        "AdditiveKernel": AdditiveKernel,
        "AdditiveStructureKernel": AdditiveStructureKernel,
        "ProductKernel": ProductKernel,
        "ProductStructureKernel": ProductStructureKernel,
        "ScaleKernel": ScaleKernel,
    }
    kernel_cls = kernels[name]
    sig = inspect.signature(kernel_cls.__init__).parameters.values()
    args = [param.name for param in sig if param.name != "self"]
    params = {}
    for k, v in kwargs.items():
        if k in args:
            if k.endswith("constraint"): params[k] = get_constraint(**v)
            elif k.endswith("prior"): params[k] = get_prior(**v)
            elif k.endswith("kernel"): params[k] = get_kernel(**v)
            elif k.endswith("covar_module"): params[k] = get_kernel(**v)
            elif k.endswith("kernels"): params[k] = [get_kernel(**vv) for vv in v]
            else: params[k] = v
    return kernel_cls(**params)
