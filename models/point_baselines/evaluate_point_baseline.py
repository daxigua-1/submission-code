import argparse
import json
import math

import numpy as np
import torch

from utils.conformal import IntervalMetrics, LEVELS
from utils.proper_scoring import CRPS_LEVELS, crps_from_quantiles, QICEMeter


def _k(m: int, c: float) -> int:
    return min(m, math.ceil((m + 1) * c))


def cell_halves(val_res: torch.Tensor, levels):
    m = val_res.size(0)
    s = torch.sort(val_res.abs(), dim=0).values
    return {c: s[_k(m, c) - 1] for c in levels}


def pooled_halves(val_res: torch.Tensor, levels):
    m, P, N = val_res.shape
    s = torch.sort(val_res.abs().permute(1, 0, 2).reshape(P, m * N), dim=1).values
    M = m * N
    return {c: s[:, _k(M, c) - 1].view(P, 1) for c in levels}


def interval_eval(pred, target, halves, levels, chunk):
    meter = IntervalMetrics(levels)
    for i in range(0, pred.size(0), chunk):
        meter.update(target[i:i + chunk], pred[i:i + chunk],
                     {c: halves[c].unsqueeze(0) for c in levels})
    return meter.compute()


def empirical_quantiles(sorted_vals: torch.Tensor, levels):
    m = sorted_vals.size(0)
    out = []
    for tau in levels:
        pos = tau * (m - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, m - 1)
        w = pos - lo
        out.append(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w)
    return torch.stack(out, 0)


def crps_eval(pred, target, q_res, chunk):
    q = q_res.permute(1, 2, 0)                      # [P,N,K]
    total, cnt = 0.0, 0
    for i in range(0, pred.size(0), chunk):
        mu, y = pred[i:i + chunk], target[i:i + chunk]
        quants = mu.unsqueeze(-1) + q.unsqueeze(0)  # [b,P,N,K]
        c = crps_from_quantiles(y, quants, CRPS_LEVELS)
        total += c.double().sum().item()
        cnt += c.numel()
    return total / cnt


def qice_eval(val_sorted, test_res, chunk, n_bins=10):
    m = val_sorted.size(0)
    s = val_sorted.permute(1, 2, 0).contiguous()    # [P,N,m]
    meter = QICEMeter(n_bins=n_bins)
    for i in range(0, test_res.size(0), chunk):
        r = test_res[i:i + chunk].permute(1, 2, 0).contiguous()   # [P,N,b]
        idx = torch.searchsorted(s, r, right=True)
        meter.update(idx.float() / m)
    return meter.compute()


def jsonable(m: dict) -> dict:
    return {'picp': {str(c): float(v) for c, v in m['picp'].items()},
            'mpiw': {str(c): float(v) for c, v in m['mpiw'].items()},
            'is': {str(c): float(v) for c, v in m['is'].items()},
            'ece': float(m['ece']), 'avg_is': float(m['avg_is'])}


def evaluate_point(val_pred, val_target, test_pred, test_target):
    val_pred, val_target = torch.as_tensor(val_pred).float(), torch.as_tensor(val_target).float()
    test_pred, test_target = torch.as_tensor(test_pred).float(), torch.as_tensor(test_target).float()
    assert val_pred.shape == val_target.shape and test_pred.shape == test_target.shape, "pred/target Inconsistent shapes"
    assert val_pred.shape[1:] == test_pred.shape[1:], "The [P, N] counts for the validation and test sets do not match."
    if val_pred.size(0) < 500:
        print(f"[warn] Only {val_pred.size(0)} validation windows; taking the 95th percentile per channel and prediction step may be unstable")

    P, N = test_pred.shape[1:]
    chunk = max(1, int(1.5e7 // (P * N * len(CRPS_LEVELS))))
    levels = tuple(LEVELS)

    val_res = val_target - val_pred
    test_res = test_target - test_pred
    val_sorted = torch.sort(val_res, dim=0).values

    res = {'mse': float((test_res ** 2).mean()), 'mae': float(test_res.abs().mean())}
    res['conformal'] = jsonable(interval_eval(test_pred, test_target, cell_halves(val_res, levels), levels, chunk))
    res['conformal_pooled'] = jsonable(interval_eval(test_pred, test_target, pooled_halves(val_res, levels), levels, chunk))
    res['crps'] = crps_eval(test_pred, test_target, empirical_quantiles(val_sorted, CRPS_LEVELS), chunk)
    res['qice'] = qice_eval(val_sorted, test_res, chunk)
    res['n_val_windows'] = int(val_pred.size(0))
    res['n_test_windows'] = int(test_pred.size(0))
    res['method'] = 'point forecast + empirical validation residuals (per channel x horizon)'
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--out_json', default=None)
    args = ap.parse_args()

    p = args.prefix
    res = evaluate_point(np.load(f"{p}_val_pred.npy"), np.load(f"{p}_val_target.npy"),
                         np.load(f"{p}_test_pred.npy"), np.load(f"{p}_test_target.npy"))
    out = args.out_json or f"{p}_metrics.json"
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)

    print(f"MSE {res['mse']:.4f}  MAE {res['mae']:.4f}  CRPS {res['crps']:.4f}  QICE {res['qice']:.4f}")
    for tag in ('conformal', 'conformal_pooled'):
        m = res[tag]
        print(f"[{tag}] PICP@90 {m['picp']['0.9']:.4f}  MPIW@90 {m['mpiw']['0.9']:.4f}  "
              f"ECE {m['ece']:.4f}  Avg_IS {m['avg_is']:.4f}")
    print(f"-> {out}")


if __name__ == '__main__':
    main()