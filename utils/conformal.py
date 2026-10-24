import math
import torch

LEVELS = (0.50, 0.75, 0.90, 0.95)
GAUSS_Z = {0.50: 0.6745, 0.75: 1.1503, 0.90: 1.6449, 0.95: 1.9600}


class ConformalCalibrator:
    def __init__(self, levels=LEVELS, mode: str = 'horizon',
                 max_channels: int = 32, seed: int = 2021):
        assert mode in ('horizon', 'global')
        self.levels = tuple(levels)
        self.mode = mode
        self.max_channels = max_channels
        self._gen = torch.Generator().manual_seed(seed)
        self._buf = []
        self.q = None
        self.n_cal = 0

    @torch.no_grad()
    def update(self, mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor):
        r = (target - mu).abs() / sigma.clamp(min=1e-8)
        N = r.size(-1)
        if N > self.max_channels:
            idx = torch.randperm(N, generator=self._gen)[: self.max_channels].to(r.device)
            r = r[..., idx]
        self._buf.append(r.float().cpu())

    @torch.no_grad()
    def fit(self):
        R = torch.cat(self._buf, 0)
        self._buf = []
        n, P, K = R.shape
        if self.mode == 'horizon':
            R = R.permute(1, 0, 2).reshape(P, n * K)
        else:
            R = R.reshape(1, n * K * P)
        m = R.size(1)
        self.n_cal = m
        self.q = {}
        for c in self.levels:
            k = min(m, math.ceil((m + 1) * c))
            self.q[c] = R.kthvalue(k, dim=1).values
        return self

    def half_width(self, c: float, sigma: torch.Tensor) -> torch.Tensor:
        q = self.q[c].to(sigma.device).view(1, -1, 1)
        return q * sigma

    def save(self, path: str):
        torch.save({'levels': self.levels, 'mode': self.mode, 'q': self.q,
                    'n_cal': self.n_cal}, path)

    @classmethod
    def load(cls, path: str):
        d = torch.load(path, map_location='cpu')
        obj = cls(levels=d['levels'], mode=d['mode'])
        obj.q, obj.n_cal = d['q'], d.get('n_cal', 0)
        return obj


class CQRCalibrator:
    def __init__(self, levels_pairs, mode: str = 'horizon', max_channels: int = 32, seed: int = 2021):
        self.levels_pairs = dict(levels_pairs)      # {confidence: (lo_idx, hi_idx)}
        self.mode = mode
        self.max_channels = max_channels
        self._gen = torch.Generator().manual_seed(seed)
        self._buf = {c: [] for c in self.levels_pairs}
        self.Q = {}

    @torch.no_grad()
    def update(self, q: torch.Tensor, target: torch.Tensor):
        N = q.size(2)
        if N > self.max_channels:
            idx = torch.randperm(N, generator=self._gen)[: self.max_channels].to(q.device)
            q, target = q[:, :, idx, :], target[:, :, idx]
        for c, (lo, hi) in self.levels_pairs.items():
            e = torch.maximum(q[..., lo] - target, target - q[..., hi])   # [B,P,K']
            self._buf[c].append(e.float().cpu())

    @torch.no_grad()
    def fit(self):
        for c, bufs in self._buf.items():
            E = torch.cat(bufs, 0)                  # [n, P, K']
            n, P, _ = E.shape
            Ef = E.reshape(n, P, -1).permute(1, 0, 2).reshape(P, -1)   # [P, n*K']
            if self.mode == 'horizon':
                Q = torch.empty(P)
                for p in range(P):
                    m = Ef.size(1)
                    k = min(m, math.ceil((m + 1) * c))
                    Q[p] = Ef[p].kthvalue(k).values
                self.Q[c] = Q
            else:
                flat = Ef.reshape(-1)
                k = min(flat.numel(), math.ceil((flat.numel() + 1) * c))
                self.Q[c] = flat.kthvalue(k).values.expand(P)
        self._buf = {c: [] for c in self._buf}
        return self

    def widen(self, c: float, q_lo: torch.Tensor, q_hi: torch.Tensor):
        Q = self.Q[c].to(q_lo.device).view(1, -1, 1)
        return q_lo - Q, q_hi + Q

    def save(self, path: str):
        torch.save({'levels_pairs': self.levels_pairs, 'mode': self.mode, 'Q': self.Q}, path)

    @classmethod
    def load(cls, path: str):
        d = torch.load(path, map_location='cpu')
        obj = cls(levels_pairs=d['levels_pairs'], mode=d['mode'])
        obj.Q = d['Q']
        return obj


class QuantileIntervalMetrics:
    def __init__(self, levels=LEVELS):
        self.levels = tuple(levels)
        self.n = 0
        self.cov = {c: 0.0 for c in self.levels}
        self.wid = {c: 0.0 for c in self.levels}
        self.isc = {c: 0.0 for c in self.levels}

    @torch.no_grad()
    def update(self, y: torch.Tensor, bounds: dict):
        self.n += y.numel()
        for c, (lo, hi) in bounds.items():
            miss = torch.relu(lo - y) + torch.relu(y - hi)
            self.cov[c] += ((y >= lo) & (y <= hi)).double().sum().item()
            self.wid[c] += (hi - lo).double().sum().item()
            self.isc[c] += ((hi - lo) + (2.0 / (1.0 - c)) * miss).double().sum().item()

    def compute(self) -> dict:
        picp = {c: self.cov[c] / self.n for c in self.levels}
        mpiw = {c: self.wid[c] / self.n for c in self.levels}
        isc  = {c: self.isc[c] / self.n for c in self.levels}
        ece  = sum(abs(picp[c] - c) for c in self.levels) / len(self.levels)
        avg_is = sum(isc.values()) / len(self.levels)
        return {'picp': picp, 'mpiw': mpiw, 'is': isc, 'ece': ece, 'avg_is': avg_is}


class IntervalMetrics:
    def __init__(self, levels=LEVELS):
        self.levels = tuple(levels)
        self.n = 0
        self.cov = {c: 0.0 for c in self.levels}
        self.wid = {c: 0.0 for c in self.levels}
        self.isc = {c: 0.0 for c in self.levels}

    @torch.no_grad()
    def update(self, y: torch.Tensor, mu: torch.Tensor, halves: dict):
        self.n += y.numel()
        for c in self.levels:
            h = halves[c]
            lo, hi = mu - h, mu + h
            miss = torch.relu(lo - y) + torch.relu(y - hi)
            self.cov[c] += ((y >= lo) & (y <= hi)).double().sum().item()
            self.wid[c] += (hi - lo).double().sum().item()
            self.isc[c] += ((hi - lo) + (2.0 / (1.0 - c)) * miss).double().sum().item()

    def compute(self) -> dict:
        picp = {c: self.cov[c] / self.n for c in self.levels}
        mpiw = {c: self.wid[c] / self.n for c in self.levels}
        isc  = {c: self.isc[c] / self.n for c in self.levels}
        ece  = sum(abs(picp[c] - c) for c in self.levels) / len(self.levels)
        avg_is = sum(isc.values()) / len(self.levels)
        return {'picp': picp, 'mpiw': mpiw, 'is': isc, 'ece': ece, 'avg_is': avg_is}