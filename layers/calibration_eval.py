import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import pandas as pd


def nig_predictive_std(nu: torch.Tensor,
                       alpha: torch.Tensor,
                       beta: torch.Tensor) -> torch.Tensor:

    alpha_m1 = (alpha - 1.0).clamp(min=1e-6)
    var = beta * (nu + 1.0) / (nu.clamp(min=1e-6) * alpha_m1)
    return var.clamp(min=1e-12).sqrt()


def nig_aleatoric_std(alpha: torch.Tensor,
                      beta: torch.Tensor) -> torch.Tensor:
    """Aleatoric std:  sqrt( β / (α-1) )"""
    alpha_m1 = (alpha - 1.0).clamp(min=1e-6)
    return (beta / alpha_m1).clamp(min=1e-12).sqrt()


def nig_epistemic_std(nu: torch.Tensor,
                      alpha: torch.Tensor,
                      beta: torch.Tensor) -> torch.Tensor:
    """Epistemic std:  sqrt( β / (ν(α-1)) )"""
    alpha_m1 = (alpha - 1.0).clamp(min=1e-6)
    return (beta / (nu.clamp(min=1e-6) * alpha_m1)).clamp(min=1e-12).sqrt()


def nig_predictive_interval(mu: torch.Tensor,
                             nu: torch.Tensor,
                             alpha: torch.Tensor,
                             beta: torch.Tensor,
                             confidence: float = 0.90
                             ) -> Tuple[torch.Tensor, torch.Tensor]:

    _Z = {0.50: 0.6745, 0.75: 1.1503, 0.90: 1.6449, 0.95: 1.9600}
    if confidence in _Z:
        z = _Z[confidence]
    else:
        from scipy.stats import norm
        z = float(norm.ppf((1 + confidence) / 2))

    sigma = nig_predictive_std(nu, alpha, beta)
    return mu - z * sigma, mu + z * sigma


def nig_crps(mu: torch.Tensor,
             nu: torch.Tensor,
             alpha: torch.Tensor,
             beta: torch.Tensor,
             y: torch.Tensor) -> torch.Tensor:

    N_SAMPLES = 200
    df    = (2.0 * alpha).clamp(min=2.01)          # df > 2 for finite variance
    scale = nig_predictive_std(nu, alpha, beta)     # approx; exact scale below
    scale_exact = (beta * (nu + 1.0) / (
        nu.clamp(min=1e-6) * alpha.clamp(min=1e-6)
    )).clamp(min=1e-12).sqrt()

    shape = (*mu.shape, N_SAMPLES)
    Z = torch.randn(shape, device=mu.device)
    V = torch.distributions.Chi2(df.unsqueeze(-1).expand(shape)
                                  ).sample()
    T = Z / (V / df.unsqueeze(-1).expand(shape)).clamp(min=1e-6).sqrt()
    samples = mu.unsqueeze(-1) + scale_exact.unsqueeze(-1) * T  # [B,N,P,S]

    y_exp = y.unsqueeze(-1).expand_as(samples)

    # E[|X - y|]
    term1 = (samples - y_exp).abs().mean(dim=-1)

    # ½ E[|X - X'|]  via two independent sample sets
    idx   = torch.randperm(N_SAMPLES, device=mu.device)
    term2 = 0.5 * (samples - samples[..., idx]).abs().mean(dim=-1)

    return (term1 - term2).mean()


class CalibrationEvaluator:


    def __init__(self,
                 confidence_levels: List[float] = None,
                 device: str = 'cpu'):
        self.levels = sorted(confidence_levels or [0.5, 0.75, 0.9, 0.95])
        self.device = device
        self._reset()


    def _reset(self):
        self.cover_sum = {lv: 0.0 for lv in self.levels}
        self.width_sum = {lv: 0.0 for lv in self.levels}
        self.total_count = 0

        self.total_std_sum = 0.0
        self.alea_std_sum = 0.0
        self.epi_std_sum = 0.0

    def update(self, mu, nu, alpha, beta, y):

        with torch.no_grad():
            for lv in self.levels:
                lower, upper = nig_predictive_interval(mu, nu, alpha, beta, lv)

                covered = ((y >= lower) & (y <= upper)).float()
                width = (upper - lower)

                self.cover_sum[lv] += covered.sum().item()
                self.width_sum[lv] += width.sum().item()

            self.total_count += y.numel()

            total_std = nig_predictive_std(nu, alpha, beta)
            alea_std = nig_aleatoric_std(alpha, beta)
            epi_std = nig_epistemic_std(nu, alpha, beta)

            self.total_std_sum += total_std.sum().item()
            self.alea_std_sum += alea_std.sum().item()
            self.epi_std_sum += epi_std.sum().item()


    def _cat(self):
        return tuple(
            torch.cat(lst, dim=0)
            for lst in [self._mu, self._nu, self._alpha, self._beta, self._y]
        )


    @staticmethod
    def _picp_mpiw(mu, nu, alpha, beta, y, confidence):
        lower, upper = nig_predictive_interval(mu, nu, alpha, beta, confidence)
        covered = ((y >= lower) & (y <= upper)).float()
        picp    = covered.mean().item()
        mpiw    = (upper - lower).mean().item()
        return picp, mpiw

    @staticmethod
    def _nmpiw(mpiw, y):
        y_range = (y.max() - y.min()).item()
        return mpiw / (y_range + 1e-8)

    def compute(self, fast_mode=False):

        results = {'picp': {}, 'mpiw': {}, 'nmpiw': {}}
        coverage_errors = []

        for lv in self.levels:
            picp = self.cover_sum[lv] / self.total_count
            mpiw = self.width_sum[lv] / self.total_count

            results['picp'][lv] = picp
            results['mpiw'][lv] = mpiw
            results['nmpiw'][lv] = mpiw

            coverage_errors.append(abs(picp - lv))

        results['ece'] = float(np.mean(coverage_errors))

        total_std = self.total_std_sum / self.total_count
        alea_std = self.alea_std_sum / self.total_count
        epi_std = self.epi_std_sum / self.total_count

        results['sharpness'] = total_std
        results['mean_aleatoric'] = alea_std
        results['mean_epistemic'] = epi_std
        results['epi_ratio'] = epi_std / (total_std + 1e-8)

        results['crps'] = None

        return results

    @staticmethod
    def print_report(results: Dict, dataset: str = ''):
        header = f"  Calibration Report  {dataset}  "
        print("=" * 60)
        print(header.center(60))
        print("=" * 60)

        print(f"\n{'Level':>8}  {'PICP':>8}  {'Nominal':>8}  "
              f"{'Error':>8}  {'MPIW':>10}  {'NMPIW':>8}")
        print("-" * 60)
        for level in sorted(results['picp']):
            picp  = results['picp'] [level]
            mpiw  = results['mpiw'] [level]
            nmpiw = results['nmpiw'][level]
            err   = abs(picp - level)
            flag  = "✓" if err < 0.02 else ("~" if err < 0.05 else "✗")
            print(f"  {level:>6.0%}  {picp:>8.4f}  {level:>8.4f}  "
                  f"{err:>8.4f}{flag}  {mpiw:>10.4f}  {nmpiw:>8.4f}")

        print("-" * 60)
        print(f"  ECE         : {results['ece']:.4f}  "
              f"(< 0.02 excellent, < 0.05 good)")
        crps_str = f"{results['crps']:.4f}" if results['crps'] is not None else "skipped (fast_mode)"
        print(f"  CRPS        : {crps_str}  (lower = better)")
        print(f"  Sharpness   : {results['sharpness']:.4f}  "
              f"(mean total predictive std)")
        print(f"  Aleatoric σ : {results['mean_aleatoric']:.4f}")
        print(f"  Epistemic σ : {results['mean_epistemic']:.4f}")
        print(f"  Epi ratio   : {results['epi_ratio']:.4f}  "
              f"(epistemic / total std)")
        print("=" * 60)


    @staticmethod
    def plot_reliability_diagram(results: Dict,
                                 save_path: str = 'reliability_diagram.pdf',
                                 dataset: str = ''):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            print("[CalibrationEvaluator] matplotlib not found; skipping plot.")
            return None

        levels  = sorted(results['picp'].keys())
        picps   = [results['picp'][l] for l in levels]

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle(f'Calibration Analysis  {dataset}', fontsize=13, y=1.01)

        ax = axes[0]
        xs = np.linspace(0, 1, 200)

        ax.fill_between(xs, xs - 0.05, xs + 0.05,
                        color='#B0C4DE', alpha=0.4, label='±5% tolerance')
        ax.fill_between(xs, xs - 0.02, xs + 0.02,
                        color='#6495ED', alpha=0.4, label='±2% tolerance')
        ax.plot(xs, xs, 'k--', linewidth=1.0, label='Perfect calibration')

        ax.plot(levels, picps, 'o-', color='#E05C2A',
                linewidth=2, markersize=7, label='PICP (our model)')

        ax.set_xlabel('Nominal confidence level', fontsize=11)
        ax.set_ylabel('Empirical coverage (PICP)',  fontsize=11)
        ax.set_title('Reliability Diagram', fontsize=11)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(alpha=0.3)

        # Annotate ECE
        ece = results['ece']
        ax.text(0.97, 0.03, f'ECE = {ece:.4f}',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=10, color='#333333',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5',
                          edgecolor='#AAAAAA'))

        ax2 = axes[1]
        sigma_a = results['mean_aleatoric']
        sigma_e = results['mean_epistemic']
        total   = sigma_a + sigma_e

        bar_vals  = [sigma_a, sigma_e, total]
        bar_label = ['Aleatoric σ_a\n(data noise)',
                     'Epistemic σ_e\n(model unc.)',
                     'Total σ']
        bar_colors = ['#5B9BD5', '#ED7D31', '#70AD47']

        bars = ax2.bar(bar_label, bar_vals, color=bar_colors,
                       edgecolor='white', width=0.45)
        for bar, v in zip(bars, bar_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + total * 0.02,
                     f'{v:.4f}', ha='center', va='bottom', fontsize=10)

        ax2.set_ylabel('Mean predictive std (normalised)', fontsize=11)
        ax2.set_title('Uncertainty Decomposition', fontsize=11)
        ax2.set_ylim(0, total * 1.25)
        ax2.grid(axis='y', alpha=0.3)

        epi_pct = results['epi_ratio'] * 100
        ax2.text(0.97, 0.97,
                 f'Epistemic ratio = {epi_pct:.1f}%',
                 transform=ax2.transAxes, ha='right', va='top',
                 fontsize=10, color='#333333',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5',
                           edgecolor='#AAAAAA'))

        fig.tight_layout()
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[CalibrationEvaluator] Diagram saved → {save_path}")
        return fig


class HorizonCalibrationEvaluator:

    def __init__(self,
                 pred_len: int,
                 confidence_levels: List[float] = None):
        self.pred_len = pred_len
        self.levels   = sorted(confidence_levels or [0.9])
        self._buckets = {t: {'mu':[], 'nu':[], 'alpha':[], 'beta':[], 'y':[]}
                         for t in range(pred_len)}

    def update(self, mu, nu, alpha, beta, y):
        # tensors: [B, N, pred_len]
        for t in range(min(self.pred_len, mu.shape[-1])):
            b = self._buckets[t]
            for key, tensor in zip(
                ['mu', 'nu', 'alpha', 'beta', 'y'],
                [mu[..., t], nu[..., t], alpha[..., t], beta[..., t], y[..., t]]
            ):
                b[key].append(tensor.detach().cpu())

    def compute_dataframe(self):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas required for compute_dataframe()")

        rows = []
        for t in range(self.pred_len):
            b  = self._buckets[t]
            mu, nu, alpha, beta, y_t = (
                torch.cat(b[k], dim=0) for k in ['mu','nu','alpha','beta','y']
            )
            row = {'horizon': t + 1}
            for level in self.levels:
                lower, upper = nig_predictive_interval(
                    mu, nu, alpha, beta, level
                )
                picp = ((y_t >= lower) & (y_t <= upper)).float().mean().item()
                mpiw = (upper - lower).mean().item()
                row[f'PICP@{int(level*100)}'] = round(picp, 4)
                row[f'MPIW@{int(level*100)}'] = round(mpiw, 4)

            row['sigma_a'] = nig_aleatoric_std(alpha, beta).mean().item()
            row['sigma_e'] = nig_epistemic_std(nu, alpha, beta).mean().item()
            rows.append(row)

        return pd.DataFrame(rows)


def run_calibration_eval(model, test_loader, args, device, setting,
                         save_dir='./calibration_results/',
                         fast_mode: bool = True,
                         skip_horizon: bool = False):

    from models.PEARL import EvidentialHead  # adjust import if needed

    os.makedirs(save_dir, exist_ok=True)

    evaluator       = CalibrationEvaluator(
        confidence_levels=[0.5, 0.75, 0.9, 0.95]
    )
    horizon_eval    = HorizonCalibrationEvaluator(
        pred_len=args.pred_len, confidence_levels=[0.9]
    )

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            if 'PEMS' in args.data or 'Solar' in args.data:
                batch_x_mark = batch_y_mark = None
            else:
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)

            out = model(batch_x, batch_x_mark, None, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0

            # Normalise targets to match NIG output space
            y_full = batch_y[:, -args.pred_len:, :]             # [B, P, N]
            y_norm = ((y_full - out.norm_mu) / out.norm_std
                      ).permute(0, 2, 1)                         # [B, N, P]

            if f_dim == -1:
                mu_f    = out.mu_norm[:, -1:, :]
                nu_f    = out.nu     [:, -1:, :]
                alpha_f = out.alpha  [:, -1:, :]
                beta_f  = out.beta   [:, -1:, :]
                y_f     = y_norm     [:, -1:, :]
            else:
                mu_f, nu_f, alpha_f, beta_f, y_f = (
                    out.mu_norm, out.nu, out.alpha, out.beta, y_norm
                )

            evaluator   .update(mu_f, nu_f, alpha_f, beta_f, y_f)
            horizon_eval.update(mu_f, nu_f, alpha_f, beta_f, y_f)

    results = evaluator.compute(fast_mode=fast_mode)
    CalibrationEvaluator.print_report(results, dataset=setting)
    evaluator.plot_reliability_diagram(
        results,
        save_path=os.path.join(save_dir, f'{setting}_reliability.pdf'),
        dataset=setting,
    )

    if not skip_horizon:
        try:
            df = horizon_eval.compute_dataframe()
            csv_path = os.path.join(save_dir, f'{setting}_horizon_calibration.csv')
            df.to_csv(csv_path, index=False)
            print(f"[CalibrationEvaluator] Horizon CSV saved → {csv_path}")
            n = len(df)
            preview = pd.concat([df.head(5), df.tail(5)]) if n > 10 else df
            print("\nHorizon-wise PICP@90 (preview):")
            print(preview[['horizon','PICP@90','sigma_a','sigma_e']].to_string(
                index=False, float_format='{:.4f}'.format
            ))
        except Exception as e:
            print(f"[CalibrationEvaluator] Horizon table skipped: {e}")

    crps_val = f"{results['crps']:.4f}" if results['crps'] is not None else "N/A"
    with open("result.txt", 'a') as f:
        f.write(f"[Calibration] {setting}\n")
        f.write(f"  ECE  : {results['ece']:.4f}\n")
        f.write(f"  CRPS : {crps_val}\n")
        for lv in [0.9, 0.95]:
            f.write(f"  PICP@{int(lv*100)}: {results['picp'].get(lv, float('nan')):.4f}"
                    f"  MPIW@{int(lv*100)}: {results['mpiw'].get(lv, float('nan')):.4f}\n")
        f.write(f"  Aleatoric: {results['mean_aleatoric']:.4f}  "
                f"Epistemic: {results['mean_epistemic']:.4f}  "
                f"Epi-ratio: {results['epi_ratio']:.4f}\n\n")

    return results