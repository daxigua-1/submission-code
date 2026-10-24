import math
import warnings

import torch


CRPS_LEVELS = tuple(i / 20 for i in range(1, 20))


def std_normal_ppf(p) -> torch.Tensor:
    if not torch.is_tensor(p):
        p = torch.tensor(p, dtype=torch.float32)
    return math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)


_SCIPY_T = None


def _get_scipy_t():
    global _SCIPY_T
    if _SCIPY_T is None:
        try:
            from scipy.stats import t as scipy_t
            _SCIPY_T = scipy_t
        except ImportError:
            _SCIPY_T = False
    return _SCIPY_T


def student_t_ppf_scalar_dof(p: float, dof: float) -> float:
    scipy_t = _get_scipy_t()
    if scipy_t is not False:
        return float(scipy_t.ppf(p, df=dof))
    warnings.warn(RuntimeWarning)
    return float(std_normal_ppf(torch.tensor(p)).item())


def pinball(y: torch.Tensor, q: torch.Tensor, tau: float) -> torch.Tensor:
    err = y - q
    return torch.maximum(tau * err, (tau - 1.0) * err)


def crps_from_quantiles(y: torch.Tensor, quantiles: torch.Tensor, levels) -> torch.Tensor:
    assert quantiles.size(-1) == len(levels)
    total = torch.zeros_like(y)
    for i, tau in enumerate(levels):
        total = total + pinball(y, quantiles[..., i], tau)
    return 2.0 * total / len(levels)


def gaussian_quantiles(mu: torch.Tensor, sigma: torch.Tensor, levels=CRPS_LEVELS) -> torch.Tensor:
    z = std_normal_ppf(torch.tensor(list(levels), device=mu.device, dtype=mu.dtype))
    return mu.unsqueeze(-1) + sigma.unsqueeze(-1) * z.view(*([1] * mu.dim()), -1)


def studentt_quantiles(mu: torch.Tensor, sigma: torch.Tensor, dof: float,
                       levels=CRPS_LEVELS) -> torch.Tensor:
    t_vals = [student_t_ppf_scalar_dof(p, dof) for p in levels]
    t_vals = torch.tensor(t_vals, device=mu.device, dtype=mu.dtype)
    return mu.unsqueeze(-1) + sigma.unsqueeze(-1) * t_vals.view(*([1] * mu.dim()), -1)


def gaussian_cdf(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    z = (y - mu) / sigma.clamp(min=1e-8)
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def studentt_cdf_scalar_dof(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, dof: float) -> torch.Tensor:
    scipy_t = _get_scipy_t()
    if scipy_t is False:
        warnings.warn("scipy 不可用，Student-t CDF 退化为高斯近似", RuntimeWarning)
        return gaussian_cdf(y, mu, sigma)
    z = ((y - mu) / sigma.clamp(min=1e-8)).detach().cpu().numpy()
    cdf = scipy_t.cdf(z, df=dof)
    import numpy as np
    return torch.from_numpy(np.asarray(cdf)).to(y.device, dtype=y.dtype)


def quantile_pit(y: torch.Tensor, q: torch.Tensor, levels) -> torch.Tensor:
    K = q.shape[-1]
    levels_t = torch.tensor(list(levels), device=y.device, dtype=y.dtype)
    below = (q <= y.unsqueeze(-1)).sum(dim=-1) - 1
    below = below.clamp(0, K - 2)
    q_lo = torch.gather(q, -1, below.unsqueeze(-1)).squeeze(-1)
    q_hi = torch.gather(q, -1, (below + 1).unsqueeze(-1)).squeeze(-1)
    l_lo = levels_t[below]
    l_hi = levels_t[below + 1]
    frac = ((y - q_lo) / (q_hi - q_lo).clamp(min=1e-8)).clamp(0.0, 1.0)
    pit = l_lo + frac * (l_hi - l_lo)
    pit = torch.where(y < q[..., 0], torch.zeros_like(pit), pit)
    pit = torch.where(y > q[..., -1], torch.ones_like(pit), pit)
    return pit


class QICEMeter:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.counts = torch.zeros(n_bins, dtype=torch.float64)

    @torch.no_grad()
    def update(self, pit: torch.Tensor):
        pit = pit.detach().flatten().clamp(0.0, 1.0 - 1e-6).cpu()
        idx = (pit * self.n_bins).long()
        self.counts += torch.bincount(idx, minlength=self.n_bins).double()

    def compute(self) -> float:
        total = self.counts.sum()
        if total == 0:
            return float('nan')
        empirical = self.counts / total
        expected = 1.0 / self.n_bins
        return (empirical - expected).abs().mean().item()


def crps_gaussian_closed_form(y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma = sigma.clamp(min=1e-8)
    z = (y - mu) / sigma
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    phi = torch.exp(-0.5 * z.pow(2)) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))