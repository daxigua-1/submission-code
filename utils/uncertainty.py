import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class UncertaintyEvaluator:

    def __init__(self, intervals, trues, preds, setting):
        self.intervals = intervals
        self.trues = trues
        self.preds = preds
        self.setting = setting
        self.conf_levels_int = [50, 75, 90, 95]
        self.conf_levels_float = [0.50, 0.75, 0.90, 0.95]
        self.metrics_results = {}

    def compute(self):
        """
        Computes probabilistic metrics exactly aligning with PEARL's output dict structure.
        """
        results = {'picp': {}, 'mpiw': {}, 'nmpiw': {}, 'is': {}}
        coverage_errors = []

        y_range = (np.max(self.trues) - np.min(self.trues)) + 1e-8
        total_is = 0.0

        for lv_int, lv_float in zip(self.conf_levels_int, self.conf_levels_float):
            L = self.intervals[lv_int]['lower']
            U = self.intervals[lv_int]['upper']
            Y = self.trues

            # 1. PICP (Coverage)
            covered = (Y >= L) & (Y <= U)
            picp = np.mean(covered)
            results['picp'][lv_float] = picp

            # 2. MPIW (Width)
            width = U - L
            mpiw = np.mean(width)
            results['mpiw'][lv_float] = mpiw
            results['nmpiw'][lv_float] = mpiw / y_range

            # 3. Interval Score (IS) - Proper scoring rule for intervals
            alpha = 1.0 - lv_float
            miss_lo = np.maximum(0.0, L - Y)
            miss_hi = np.maximum(0.0, Y - U)
            is_score = np.mean(width + (2.0 / alpha) * (miss_lo + miss_hi))
            results['is'][lv_float] = is_score
            total_is += is_score

            # Accumulate ECE error
            coverage_errors.append(abs(picp - lv_float))

        # 4. ECE (Expected Calibration Error)
        results['ece'] = float(np.mean(coverage_errors))
        results['avg_is'] = total_is / len(self.conf_levels_float)

        results['crps'] = None
        results['sharpness'] = None
        results['mean_aleatoric'] = None
        results['mean_epistemic'] = None

        self.metrics_results = results
        return results

    def export_table6_csv(self, save_dir='./results/'):
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, 'Table6_Conformal_Baseline.csv')

        if not self.metrics_results:
            self.compute()

        res = self.metrics_results

        row_data = {
            'Model': f'Baseline+CP ({self.setting})',
            'PICP50': f"{res['picp'][0.50]:.3f}",
            'PICP75': f"{res['picp'][0.75]:.3f}",
            'PICP90': f"{res['picp'][0.90]:.3f}",
            'PICP95': f"{res['picp'][0.95]:.3f}",
            'MPIW@90': f"{res['mpiw'][0.90]:.3f}",
            'ECE': f"{res['ece']:.3f}",
            'Avg_IS': f"{res['avg_is']:.3f}"
        }

        df = pd.DataFrame([row_data])
        df.to_csv(csv_path, index=False)

    def generate_review_plots(self, save_dir='./results/'):
        if not self.metrics_results:
            self.compute()

        fig_dir = os.path.join(save_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        res = self.metrics_results

        plt.figure(figsize=(6, 6))
        empirical = [res['picp'][c] for c in self.conf_levels_float]

        plt.plot([0, 1], [0, 1], 'k--', label='Ideal Calibration')
        plt.plot(self.conf_levels_float, empirical, 'ro-', linewidth=2, markersize=8, label='Conformal Prediction')
        plt.xlabel('Nominal Confidence', fontsize=12)
        plt.ylabel('Empirical Coverage (PICP)', fontsize=12)
        plt.title(f'Reliability Diagram - {self.setting}', fontsize=13)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'calibration_curve.pdf'), dpi=300)
        plt.close()

        plt.figure(figsize=(10, 4))
        sample_idx = 0
        channel_idx = -1
        steps = min(96, self.preds.shape[1])

        t = np.arange(steps)
        y_true = self.trues[sample_idx, :steps, channel_idx]
        y_pred = self.preds[sample_idx, :steps, channel_idx]
        y_lower = self.intervals[90]['lower'][sample_idx, :steps, channel_idx]
        y_upper = self.intervals[90]['upper'][sample_idx, :steps, channel_idx]

        plt.plot(t, y_true, 'black', label='Ground Truth', linewidth=1.5)
        plt.plot(t, y_pred, 'blue', label='Point Forecast', linewidth=1.5)
        plt.fill_between(t, y_lower, y_upper, color='orange', alpha=0.3, label='90% Prediction Interval')
        plt.xlabel('Prediction Horizon Steps', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title(f'Prediction Interval (90% Level) - {self.setting}', fontsize=13)
        plt.legend(loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'prediction_interval_sample.pdf'), dpi=300)
        plt.close()