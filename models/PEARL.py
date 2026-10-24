import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def predictive_std(nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """sqrt(σ_a² + σ_e²) = sqrt( β(1+ν) / (ν(α-1)) )"""
    alpha_m1 = (alpha - 1.0).clamp(min=1e-6)
    var = beta * (1.0 + nu) / (nu.clamp(min=1e-6) * alpha_m1)
    return var.clamp(min=1e-12).sqrt()


class UncertaintyProbe(nn.Module):

    def __init__(self, d_model: int, nu_min: float = 0.1, alpha_m1_min: float = 0.5):
        super().__init__()
        hidden = max(d_model // 4, 8)
        self.nu_min = nu_min
        self.alpha_m1_min = alpha_m1_min
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor):
        h = x.mean(dim=1)                                       # [B, D]
        p = self.net(h)                                         # [B, 3]
        log_nu, log_alpha_m1, log_beta = p.unbind(-1)
        nu       = F.softplus(log_nu)       + self.nu_min
        alpha_m1 = F.softplus(log_alpha_m1) + self.alpha_m1_min
        beta     = F.softplus(log_beta)     + 1e-4
        sigma_a = (beta / alpha_m1)[:, None, None]
        sigma_e = (beta / (nu * alpha_m1))[:, None, None]
        return sigma_a, sigma_e


class ScoreNetwork(nn.Module):

    def __init__(self, enc_in: int, d_model: int, kernel: int = 3):
        super().__init__()
        assert kernel % 2 == 1, "kernel must be odd"
        pad = kernel // 2
        self.encoder = nn.Sequential(
            nn.Conv1d(enc_in,        d_model // 4, kernel, stride=2, padding=pad), nn.ReLU(),
            nn.Conv1d(d_model // 4, d_model // 2, kernel, stride=2, padding=pad), nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model,       kernel, stride=2, padding=pad), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(d_model,       d_model // 2, kernel, stride=2, padding=pad, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(d_model // 2, d_model // 4, kernel, stride=2, padding=pad, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(d_model // 4, enc_in,        kernel, stride=2, padding=pad, output_padding=1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip    = x
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        L       = min(skip.size(2), decoded.size(2))
        return decoded[:, :, :L] + skip[:, :, :L]


class MLP_attention(nn.Module):

    def __init__(self, d_model: int, enc_in: int, kernel: int, use_uats: bool = True):
        super().__init__()
        self.use_uats = use_uats
        self.value = nn.Linear(d_model, d_model)
        self.score = ScoreNetwork(enc_in, d_model, kernel)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.norm  = nn.LayerNorm(d_model)
        self.probe = UncertaintyProbe(d_model)
        self._lambda_a = nn.Parameter(torch.full((1,), -2.25))   # softplus(-2.25) ≈ 0.10
        self._lambda_e = nn.Parameter(torch.full((1,), -2.25))

    @property
    def lambda_a(self) -> torch.Tensor:
        return F.softplus(self._lambda_a)

    @property
    def lambda_e(self) -> torch.Tensor:
        return F.softplus(self._lambda_e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        if self.use_uats:
            sigma_a, sigma_e = self.probe(x)
            tau = 1.0 + self.lambda_a * sigma_a + self.lambda_e * sigma_e   # [B,1,1]
        else:
            tau = torch.ones(B, 1, 1, device=x.device, dtype=x.dtype)

        raw_score    = self.score(x)
        attn         = torch.softmax(raw_score / tau, dim=1)

        self._record_tau = tau.detach().view(B)
        self._record_entropy = -(attn * torch.log(attn + 1e-9)).sum(dim=1).mean(dim=-1).detach()

        x1 = attn * self.value(x) + x
        x1 = self.norm(x1.reshape(-1, D)).reshape(B, N, D)
        out = self.norm((self.mlp(x1) + x1).reshape(-1, D)).reshape(B, N, D)
        return out


class EvidentialHead(nn.Module):
    def __init__(self, d_model: int, pred_len: int, init_log_scale,
                 nu_min: float = 0.1, alpha_min: float = 1.5):
        super().__init__()
        self.pred_len = pred_len
        self.nu_min = nu_min
        self.alpha_min = alpha_min
        self.proj = nn.Linear(d_model, pred_len * 4)
        self.log_scale = nn.Parameter(torch.tensor(init_log_scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        B, N, _ = x.shape
        raw  = self.proj(x).reshape(B, N, self.pred_len, 4)

        mu    = raw[..., 0]
        nu    = F.softplus(raw[..., 1]) + self.nu_min
        alpha = F.softplus(raw[..., 2]) + self.alpha_min
        beta  = F.softplus(raw[..., 3]) + 1e-4

        log_s = self.log_scale.clamp(-3.0, 3.0)
        beta  = beta * torch.exp(2.0 * log_s)
        return mu, nu, alpha, beta

    @staticmethod
    def get_uncertainty(nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
        alpha_m1 = (alpha - 1.0).clamp(min=1e-6)
        sigma_a  = (beta / alpha_m1).clamp(min=1e-12).sqrt()
        sigma_e  = (beta / (nu.clamp(min=1e-6) * alpha_m1)).clamp(min=1e-12).sqrt()
        return sigma_a, sigma_e


class IntervalScore(nn.Module):
    _Z = {0.50: 0.6745, 0.75: 1.1503, 0.90: 1.6449, 0.95: 1.9600}

    def __init__(self, confidence_levels: list = None):
        super().__init__()
        self.levels = confidence_levels or [0.50, 0.75, 0.90, 0.95]
        self.z_scores = [self._Z[c] for c in self.levels]

    def forward(self, y, mu, nu, alpha, beta):
        sigma_tot = predictive_std(nu, alpha, beta)
        total = 0.0
        for c, z in zip(self.levels, self.z_scores):
            a = 1.0 - c
            lower, upper = mu - z * sigma_tot, mu + z * sigma_tot
            miss = F.relu(lower - y) + F.relu(y - upper)
            total = total + ((upper - lower) + (2.0 / a) * miss).mean()
        return total / len(self.levels)


class EvidentialLoss(nn.Module):

    def __init__(self, lam: float = 0.05, nll_scale: float = 2.0, lam_is: float = 0.1,
                 detach_mu: bool = True):
        super().__init__()
        self.lam = lam
        self.nll_scale = nll_scale
        self.lam_is = lam_is
        self.detach_mu = detach_mu
        self._interval_score = IntervalScore()

    def forward(self, y, mu, nu, alpha, beta):
        if self.detach_mu:
            mu = mu.detach()
        eps   = 1e-7
        omega = (2.0 * beta * (1.0 + nu)).clamp(min=eps)
        err   = y - mu

        nll = (0.5 * torch.log(math.pi / nu.clamp(min=eps))
               - alpha * torch.log(omega)
               + (alpha + 0.5) * torch.log((omega + nu * err.pow(2)).clamp(min=eps))
               + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5))
        reg = err.abs() * (2.0 * nu + alpha)

        loss = self.nll_scale * nll.mean() + self.lam * reg.mean()
        if self.lam_is > 0:
            loss = loss + self.lam_is * self._interval_score(y, mu, nu, alpha, beta)
        return loss


QUANTILE_LEVELS = (0.025, 0.05, 0.125, 0.25, 0.5, 0.75, 0.875, 0.95, 0.975)

QUANTILE_PAIR_IDX = {0.50: (3, 5), 0.75: (2, 6), 0.90: (1, 7), 0.95: (0, 8)}


class GaussianHead(nn.Module):

    def __init__(self, d_model: int, pred_len: int, init_log_scale):
        super().__init__()
        self.pred_len = pred_len
        self.proj = nn.Linear(d_model, pred_len * 2)
        self.log_scale = nn.Parameter(torch.tensor(init_log_scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        B, N, _ = x.shape
        raw = self.proj(x).reshape(B, N, self.pred_len, 2)
        mu = raw[..., 0]
        log_sigma = raw[..., 1].clamp(-5.0, 5.0) + self.log_scale.clamp(-3.0, 3.0)
        sigma = torch.exp(log_sigma).clamp(min=1e-4)
        return mu, sigma


class GaussianNLL(nn.Module):
    def forward(self, y, mu, sigma):
        z = (y - mu) / sigma
        return (torch.log(sigma) + 0.5 * z.pow(2) + 0.5 * math.log(2 * math.pi)).mean()


class StudentTHead(nn.Module):
    def __init__(self, d_model: int, pred_len: int, init_log_scale, dof_min: float = 2.1):
        super().__init__()
        self.pred_len = pred_len
        self.dof_min = dof_min
        self.proj = nn.Linear(d_model, pred_len * 2)
        self.log_scale = nn.Parameter(torch.tensor(init_log_scale, dtype=torch.float32))
        self._log_dof = nn.Parameter(torch.tensor(2.0))

    @property
    def dof(self) -> torch.Tensor:
        return F.softplus(self._log_dof) + self.dof_min

    def forward(self, x: torch.Tensor):
        B, N, _ = x.shape
        raw = self.proj(x).reshape(B, N, self.pred_len, 2)
        mu = raw[..., 0]
        log_sigma = raw[..., 1].clamp(-5.0, 5.0) + self.log_scale.clamp(-3.0, 3.0)
        sigma = torch.exp(log_sigma).clamp(min=1e-4)
        return mu, sigma, self.dof


class StudentTNLL(nn.Module):
    def forward(self, y, mu, sigma, dof):
        z = (y - mu) / sigma
        nll = (torch.lgamma(dof / 2) - torch.lgamma((dof + 1) / 2)
               + 0.5 * torch.log(dof * math.pi) + torch.log(sigma)
               + (dof + 1) / 2 * torch.log1p(z.pow(2) / dof))
        return nll.mean()


class QuantileHead(nn.Module):

    def __init__(self, d_model: int, pred_len: int, levels=QUANTILE_LEVELS):
        super().__init__()
        self.pred_len = pred_len
        self.levels = levels
        self.proj = nn.Linear(d_model, pred_len * len(levels))

    def forward(self, x: torch.Tensor):
        B, N, _ = x.shape
        raw = self.proj(x).reshape(B, N, self.pred_len, len(self.levels))
        q, _ = torch.sort(raw, dim=-1)
        return q


class PinballLoss(nn.Module):
    def __init__(self, levels=QUANTILE_LEVELS):
        super().__init__()
        self.levels = levels

    def forward(self, y, q):
        y = y.unsqueeze(-1)
        loss = 0.0
        for i, tau in enumerate(self.levels):
            err = y[..., 0] - q[..., i]
            loss = loss + torch.maximum(tau * err, (tau - 1.0) * err).mean()
        return loss / len(self.levels)


class HybridEmbedding(nn.Module):

    def __init__(self, seq_len: int, d_model: int, patch_len: int = 16, stride: int = 8):
        super().__init__()
        self.point_proj = nn.Linear(seq_len, d_model)
        self.patch_len  = patch_len
        self.stride     = stride
        self.n_patches  = (seq_len - patch_len) // stride + 1
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.patch_fuse = nn.Linear(self.n_patches, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_point = self.point_proj(x)
        patches   = x.unfold(-1, self.patch_len, self.stride)
        out_patch = self.patch_proj(patches)
        out_patch = self.patch_fuse(out_patch.transpose(-1, -2)).squeeze(-1)
        return out_point + out_patch


@dataclass
class _ModelOutput:
    pred: torch.Tensor
    mu_norm: torch.Tensor
    norm_mu: torch.Tensor
    norm_std: torch.Tensor
    mean_tau: torch.Tensor
    mean_entropy: torch.Tensor
    head_type: str = 'nig'
    nu: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None
    beta: Optional[torch.Tensor] = None
    sigma: Optional[torch.Tensor] = None
    dof: Optional[torch.Tensor] = None
    q: Optional[torch.Tensor] = None


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.head_type = getattr(configs, 'head_type', 'nig')
        use_uats = getattr(configs, 'ablation', 'none') != 'no_uats'
        self.embedding = HybridEmbedding(configs.seq_len, configs.d_model, configs.patch_len)
        self.blocks = nn.ModuleList([
            MLP_attention(configs.d_model, configs.enc_in, configs.kernel, use_uats=use_uats)
            for _ in range(configs.e_layers)
        ])
        if self.head_type == 'nig':
            self.head = EvidentialHead(
                configs.d_model, configs.pred_len, configs.log_scale,
                nu_min=getattr(configs, 'nu_min', 0.1),
                alpha_min=getattr(configs, 'alpha_min', 1.5),
            )
        elif self.head_type == 'gaussian':
            self.head = GaussianHead(configs.d_model, configs.pred_len, configs.log_scale)
        elif self.head_type == 'studentt':
            self.head = StudentTHead(configs.d_model, configs.pred_len, configs.log_scale,
                                     dof_min=getattr(configs, 'dof_min', 2.1))
        elif self.head_type == 'quantile':
            self.head = QuantileHead(configs.d_model, configs.pred_len)
        else:
            raise ValueError(f"unknown head_type: {self.head_type!r}, "
                             f"expected one of nig/gaussian/studentt/quantile")

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        norm_mu  = x.mean(1, keepdim=True)
        norm_std = torch.sqrt(x.var(1, keepdim=True) + 1e-5)
        x_n = (x - norm_mu) / norm_std

        h = self.embedding(x_n.permute(0, 2, 1))
        h = F.relu(h)
        for blk in self.blocks:
            h = F.relu(blk(h))

        std_n = norm_std.squeeze(1).unsqueeze(-1)
        mu_n  = norm_mu.squeeze(1).unsqueeze(-1)
        mean_tau     = sum(blk._record_tau for blk in self.blocks) / len(self.blocks)
        mean_entropy = sum(blk._record_entropy for blk in self.blocks) / len(self.blocks)
        common = dict(norm_mu=norm_mu, norm_std=norm_std, mean_tau=mean_tau, mean_entropy=mean_entropy)

        if self.head_type == 'nig':
            mu_norm, nu, alpha, beta = self.head(h)
            pred = (mu_norm * std_n + mu_n).permute(0, 2, 1)
            return _ModelOutput(pred=pred, mu_norm=mu_norm, head_type='nig',
                                nu=nu, alpha=alpha, beta=beta, **common)

        if self.head_type == 'gaussian':
            mu_norm, sigma_norm = self.head(h)
            pred = (mu_norm * std_n + mu_n).permute(0, 2, 1)
            return _ModelOutput(pred=pred, mu_norm=mu_norm, head_type='gaussian',
                                sigma=sigma_norm, **common)

        if self.head_type == 'studentt':
            mu_norm, sigma_norm, dof = self.head(h)
            pred = (mu_norm * std_n + mu_n).permute(0, 2, 1)
            return _ModelOutput(pred=pred, mu_norm=mu_norm, head_type='studentt',
                                sigma=sigma_norm, dof=dof, **common)

        # quantile
        q_norm = self.head(h)                                  # [B,N,P,K]
        mid = len(QUANTILE_LEVELS) // 2
        mu_norm = q_norm[..., mid]
        pred = (mu_norm * std_n + mu_n).permute(0, 2, 1)
        return _ModelOutput(pred=pred, mu_norm=mu_norm, head_type='quantile',
                            q=q_norm, **common)