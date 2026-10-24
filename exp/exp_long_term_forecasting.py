from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter

import torch
import torch.nn as nn
from torch import optim

import os
import time
import warnings
import numpy as np
warnings.filterwarnings('ignore')

from models.PEARL import EvidentialLoss, GaussianNLL, StudentTNLL, PinballLoss, predictive_std, \
    QUANTILE_LEVELS, QUANTILE_PAIR_IDX
from utils.conformal import ConformalCalibrator, CQRCalibrator, IntervalMetrics, \
    QuantileIntervalMetrics, GAUSS_Z, LEVELS
from utils.proper_scoring import CRPS_LEVELS, gaussian_quantiles, studentt_quantiles, \
    crps_from_quantiles, student_t_ppf_scalar_dof, gaussian_cdf, studentt_cdf_scalar_dof, \
    quantile_pit, QICEMeter


class Exp_Long_Term_Forecast(Exp_Basic):

    def __init__(self, args):
        super().__init__(args)
        self.calibrator = None
        self.cqr_calibrator = None
        self.ctrl_cals = {}
        if self.args.head_type != 'nig' and self.args.ablation not in ('none', 'no_uats'):
            print(f"[Warn] ablation={self.args.ablation!r} only makes sense for head_type='nig' "
                 f"(no_is/no_nig are NIG-specific ablations). Current head_type="
                 f"{self.args.head_type!r}; this setting will be ignored, only no_uats remains effective.")

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _is_probabilistic(self):
        return not (self.args.head_type == 'nig' and self.args.ablation == 'no_nig')

    def _select_criterion(self):
        ht = self.args.head_type
        if ht == 'nig':
            if self.args.ablation == 'no_nig':
                return None
            lam_is = 0.0 if self.args.ablation == 'no_is' else self.args.evid_lam_is
            return EvidentialLoss(lam=self.args.evid_lam,
                                  nll_scale=self.args.evid_nll_scale,
                                  lam_is=lam_is,
                                  detach_mu=bool(self.args.detach_mu))
        if ht == 'gaussian':
            return GaussianNLL()
        if ht == 'studentt':
            return StudentTNLL()
        if ht == 'quantile':
            return PinballLoss()
        raise ValueError(f"unknown head_type: {ht!r}")

    def _unpack(self, batch):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        no_cal = ('PEMS' in self.args.data or 'Solar' in self.args.data
                 or bool(getattr(self.args, 'no_calendar', 0)))
        if no_cal:
            batch_x_mark = batch_y_mark = None
        else:
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
        return batch_x, batch_y, batch_x_mark, batch_y_mark

    @staticmethod
    def _normalise_target(batch_y, norm_mu, norm_std):
        return ((batch_y - norm_mu) / norm_std).permute(0, 2, 1)     # [B, N, P]

    def _slice_head(self, out, f_dim):
        def sl(t):
            return t[:, -1:, :] if f_dim == -1 else t
        mu_f = sl(out.mu_norm)
        if out.head_type == 'nig':
            return mu_f, dict(nu=sl(out.nu), alpha=sl(out.alpha), beta=sl(out.beta))
        if out.head_type == 'gaussian':
            return mu_f, dict(sigma=sl(out.sigma))
        if out.head_type == 'studentt':
            return mu_f, dict(sigma=sl(out.sigma), dof=out.dof)
        if out.head_type == 'quantile':
            return mu_f, dict(q=sl(out.q))
        raise ValueError(out.head_type)

    @torch.no_grad()
    def _forward_unc(self, batch):
        batch_x, batch_y, bxm, bym = self._unpack(batch)
        out = self.model(batch_x, bxm, None, bym)
        assert out.head_type != 'quantile', "Use _forward_quantile for quantile head"
        P = self.args.pred_len
        f_dim = -1 if self.args.features == 'MS' else 0
        mu = out.pred[:, -P:, f_dim:].float()
        target = batch_y[:, -P:, f_dim:]
        std_in = out.norm_std.expand(-1, P, -1)[:, :, f_dim:].float()
        scale = out.norm_std.squeeze(1).unsqueeze(-1)          # [B,N,1]

        if out.head_type == 'nig':
            sigma_n = predictive_std(out.nu.float(), out.alpha.float(), out.beta.float())
            dof = None
        elif out.head_type == 'gaussian':
            sigma_n = out.sigma.float()
            dof = None
        elif out.head_type == 'studentt':
            sigma_n = out.sigma.float()
            dof = float(out.dof.item())
        else:
            raise ValueError(out.head_type)

        sigma = (sigma_n * scale).permute(0, 2, 1)[:, -P:, f_dim:]
        return mu, sigma, target, std_in, dof

    @torch.no_grad()
    def _forward_quantile(self, batch):
        batch_x, batch_y, bxm, bym = self._unpack(batch)
        out = self.model(batch_x, bxm, None, bym)
        assert out.head_type == 'quantile'
        P = self.args.pred_len
        f_dim = -1 if self.args.features == 'MS' else 0
        target = batch_y[:, -P:, f_dim:]
        scale = out.norm_std.squeeze(1).unsqueeze(-1).unsqueeze(-1)   # [B,N,1,1]
        shift = out.norm_mu.squeeze(1).unsqueeze(-1).unsqueeze(-1)    # [B,N,1,1]
        q = (out.q * scale + shift).permute(0, 2, 1, 3)[:, -P:, f_dim:, :].float()  # [B,P,N',K]
        mid = len(QUANTILE_LEVELS) // 2
        mu = q[..., mid]
        return q, mu, target

    def export_mu(self, out_prefix):
        from torch.utils.data import DataLoader
        assert self.args.head_type != 'quantile'
        self.model.eval()
        for tag in ('val', 'test'):
            ds, _ = self._get_data(tag)
            loader = DataLoader(ds, batch_size=self.args.batch_size, shuffle=False,
                                drop_last=False, num_workers=0)
            mus, sigs, ys = [], [], []
            for batch in loader:
                mu, sigma, target, _std_in, _dof = self._forward_unc(batch)
                mus.append(mu.cpu().numpy());
                sigs.append(sigma.cpu().numpy());
                ys.append(target.cpu().numpy())
            out_dir = os.path.dirname(out_prefix)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            for name, arr in (('pred', mus), ('sigma', sigs), ('target', ys)):
                np.save(f"{out_prefix}_{tag}_{name}.npy", np.concatenate(arr, 0).astype(np.float32))

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for batch in vali_loader:
                batch_x, batch_y, bxm, bym = self._unpack(batch)
                out = self.model(batch_x, bxm, None, bym)
                f_dim = -1 if self.args.features == 'MS' else 0
                pred   = out.pred[:, -self.args.pred_len:, f_dim:]
                target = batch_y[:, -self.args.pred_len:, f_dim:]
                total_loss.update(nn.MSELoss()(pred, target).item(), batch_x.size(0))
        self.model.train()
        return total_loss.avg

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data,  vali_loader  = self._get_data(flag='val')
        test_data,  test_loader  = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        train_steps    = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim    = self._select_optimizer()
        criterion      = self._select_criterion()
        mse_criterion  = nn.MSELoss()
        mae_criterion  = nn.L1Loss()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            time_now   = time.time()

            for i, batch in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)
                batch_x, batch_y, bxm, bym = self._unpack(batch)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        out = self.model(batch_x, bxm, None, bym)
                else:
                    out = self.model(batch_x, bxm, None, bym)

                P = self.args.pred_len
                f_dim  = -1 if self.args.features == 'MS' else 0
                pred   = out.pred[:, -P:, f_dim:].float()
                target = batch_y[:, -P:, f_dim:]

                loss_point = mse_criterion(pred, target) + mae_criterion(pred, target)
                loss_unc = None
                if criterion is None:                                  # ablation = no_nig
                    loss = loss_point
                else:
                    mu_f, extra = self._slice_head(out, f_dim)
                    y_norm = self._normalise_target(batch_y[:, -P:, :], out.norm_mu, out.norm_std)
                    if f_dim == -1:
                        y_norm = y_norm[:, -1:, :]
                    ht = out.head_type
                    if ht == 'nig':
                        loss_unc = criterion(y_norm.float(), mu_f.float(), extra['nu'].float(),
                                             extra['alpha'].float(), extra['beta'].float())
                    elif ht == 'gaussian':
                        mu_l = mu_f.detach() if self.args.detach_mu else mu_f
                        loss_unc = criterion(y_norm.float(), mu_l.float(), extra['sigma'].float())
                    elif ht == 'studentt':
                        mu_l = mu_f.detach() if self.args.detach_mu else mu_f
                        loss_unc = criterion(y_norm.float(), mu_l.float(),
                                             extra['sigma'].float(), extra['dof'].float())
                    elif ht == 'quantile':
                        loss_unc = criterion(y_norm.float(), extra['q'].float())
                    else:
                        raise ValueError(ht)
                    loss = loss_point + self.args.unc_weight * loss_unc

                if (i + 1) % 100 == 0:
                    train_loss.append(mae_criterion(pred, target).item())
                    unc_str = f" | unc: {loss_unc.item():.4f}" if loss_unc is not None else ""
                    print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.6f}"
                          f" | point: {loss_point.item():.6f}{unc_str} | MAE: {train_loss[-1]:.6f}")
                    speed     = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f"\tspeed: {speed:.4f}s/iter; left time: {left_time:.1f}s")
                    iter_count = 0
                    time_now   = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print(f"Epoch {epoch+1} cost time: {time.time() - epoch_time:.1f}s")
            train_loss = np.mean(train_loss) if train_loss else float('nan')
            vali_loss  = self.vali(vali_data, vali_loader, mse_criterion)
            test_loss  = self.vali(test_data, test_loader, mse_criterion)
            print(f"Epoch: {epoch+1}, Steps: {train_steps} | Train MAE: {train_loss:.6f}"
                  f" | Vali MSE: {vali_loss:.6f} | Test MSE: {test_loss:.6f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        if self._is_probabilistic():
            if self.args.head_type == 'quantile':
                self._fit_cqr_calibrator(vali_loader, path)
            else:
                self._fit_calibrator(vali_loader, path)
        return self.model


    def _make_cal(self):
        return ConformalCalibrator(mode=self.args.calib_mode,
                                   max_channels=self.args.calib_max_channels,
                                   seed=getattr(self.args, 'seed', 2021))

    def _fit_calibrator(self, vali_loader, path=None):
        cal = self._make_cal()
        use_ctrl = bool(getattr(self.args, 'controls', 1))
        ctrl = {'const': self._make_cal(), 'winstd': self._make_cal()} if use_ctrl else {}
        self.model.eval()
        with torch.no_grad():
            for batch in vali_loader:
                mu, sigma, target, std_in, _dof = self._forward_unc(batch)
                cal.update(mu, sigma, target)
                if use_ctrl:
                    ctrl['const'].update(mu, torch.ones_like(sigma), target)
                    ctrl['winstd'].update(mu, std_in, target)
        cal.fit()
        for c in ctrl.values():
            c.fit()
        self.calibrator, self.ctrl_cals = cal, ctrl
        if path is not None:
            cal.save(os.path.join(path, 'calibrator.pt'))
            for name, c in ctrl.items():
                c.save(os.path.join(path, f'calibrator_{name}.pt'))
        print(f"[Calibrator] fitted on val: mode={cal.mode}, n_cal={cal.n_cal}")
        return cal

    def _fit_cqr_calibrator(self, vali_loader, path=None):
        cal = CQRCalibrator(QUANTILE_PAIR_IDX, mode=self.args.calib_mode,
                            max_channels=self.args.calib_max_channels,
                            seed=getattr(self.args, 'seed', 2021))
        self.model.eval()
        with torch.no_grad():
            for batch in vali_loader:
                q, _mu, target = self._forward_quantile(batch)
                cal.update(q, target)
        cal.fit()
        self.cqr_calibrator = cal
        if path is not None:
            cal.save(os.path.join(path, 'calibrator_cqr.pt'))
        print(f"[CQR Calibrator] fitted on val: mode={cal.mode}")
        return cal

    def _load_calibrators(self, path):
        if self.args.head_type == 'quantile':
            cp = os.path.join(path, 'calibrator_cqr.pt')
            if os.path.exists(cp):
                self.cqr_calibrator = CQRCalibrator.load(cp)
            return
        cal_path = os.path.join(path, 'calibrator.pt')
        if os.path.exists(cal_path):
            self.calibrator = ConformalCalibrator.load(cal_path)
        for name in ('const', 'winstd'):
            pc = os.path.join(path, f'calibrator_{name}.pt')
            if os.path.exists(pc):
                self.ctrl_cals[name] = ConformalCalibrator.load(pc)

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)

        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join(path, 'checkpoint.pth'), map_location=self.device))
            self._load_calibrators(path)

        if not self._is_probabilistic():
            return self._test_deterministic(test_loader, setting)
        if self.args.head_type == 'quantile':
            return self._test_quantile(test_loader, path, setting)
        return self._test_scalar_sigma(test_loader, path, setting)

    def _test_deterministic(self, test_loader, setting):
        mse_meter, mae_meter = AverageMeter(), AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch_x, batch_y, bxm, bym = self._unpack(batch)
                out = self.model(batch_x, bxm, None, bym)
                f_dim = -1 if self.args.features == 'MS' else 0
                pred = out.pred[:, -self.args.pred_len:, f_dim:]
                target = batch_y[:, -self.args.pred_len:, f_dim:]
                mse_meter.update(nn.MSELoss()(pred, target).item(), batch_x.size(0))
                mae_meter.update(nn.L1Loss()(pred, target).item(), batch_x.size(0))
        mse, mae = mse_meter.avg, mae_meter.avg
        os.makedirs(os.path.join('./results/', setting), exist_ok=True)
        print(f'MSE: {mse:.6f}  MAE: {mae:.6f}')
        with open("result.txt", 'a') as f:
            f.write(f"{setting}\nmse:{mse:.6f}, mae:{mae:.6f}\n\n")
        return mse, mae, None

    @staticmethod
    def _show_interval(tag, m):
        print(f"--- {tag} ---  Avg IS: {m['avg_is']:.4f} | ECE: {m['ece']:.4f}")
        for c in LEVELS:
            print(f"  -> PICP@{int(c*100)}: {m['picp'][c]*100:.2f}% | "
                  f"MPIW@{int(c*100)}: {m['mpiw'][c]:.4f} | IS@{int(c*100)}: {m['is'][c]:.4f}")

    def _test_scalar_sigma(self, test_loader, path, setting):
        use_ctrl = bool(getattr(self.args, 'controls', 1))
        if self.calibrator is None or (use_ctrl and len(self.ctrl_cals) < 2):
            if self.args.calibrate == 'conformal':
                vali_data, vali_loader = self._get_data(flag='val')
                self._fit_calibrator(vali_loader, path)

        mse_meter, mae_meter, crps_meter = AverageMeter(), AverageMeter(), AverageMeter()
        qice_meter = QICEMeter(n_bins=10)
        raw_m  = IntervalMetrics(LEVELS)
        cal_m  = IntervalMetrics(LEVELS)
        ctrl_m = {name: IntervalMetrics(LEVELS) for name in self.ctrl_cals} if use_ctrl else {}

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                mu, sigma, target, std_in, dof = self._forward_unc(batch)
                mse_meter.update(nn.MSELoss()(mu, target).item(), mu.size(0))
                mae_meter.update(nn.L1Loss()(mu, target).item(), mu.size(0))

                if dof is not None:      # studentt: use true t-quantiles/CDF instead of gaussian approximation
                    quants = studentt_quantiles(mu, sigma, dof, CRPS_LEVELS)
                    halves = {c: student_t_ppf_scalar_dof(0.5 + c / 2.0, dof) * sigma for c in LEVELS}
                    pit = studentt_cdf_scalar_dof(target, mu, sigma, dof)
                else:
                    quants = gaussian_quantiles(mu, sigma, CRPS_LEVELS)
                    halves = {c: GAUSS_Z[c] * sigma for c in LEVELS}
                    pit = gaussian_cdf(target, mu, sigma)
                crps_vals = crps_from_quantiles(target, quants, CRPS_LEVELS)
                crps_meter.update(crps_vals.mean().item(), target.numel())
                qice_meter.update(pit)

                raw_m.update(target, mu, halves)
                if self.calibrator is not None:
                    cal_m.update(target, mu, {c: self.calibrator.half_width(c, sigma) for c in LEVELS})
                for name, cc in self.ctrl_cals.items():
                    if name not in ctrl_m:
                        continue
                    sig_c = torch.ones_like(sigma) if name == 'const' else std_in
                    ctrl_m[name].update(target, mu, {c: cc.half_width(c, sig_c) for c in LEVELS})

        mse, mae, crps, qice = mse_meter.avg, mae_meter.avg, crps_meter.avg, qice_meter.compute()
        os.makedirs(os.path.join('./results/', setting), exist_ok=True)
        print(f'MSE: {mse:.6f}  MAE: {mae:.6f}  CRPS: {crps:.6f}  QICE: {qice:.6f}')

        raw = raw_m.compute()
        self._show_interval(f'Raw {self.args.head_type} intervals', raw)
        metrics, tag = raw, 'raw'
        if self.calibrator is not None and self.args.calibrate == 'conformal':
            cal = cal_m.compute()
            self._show_interval(f'Conformal-calibrated intervals ({self.calibrator.mode})', cal)
            metrics, tag = cal, 'conformal'
        metrics['raw'] = raw
        metrics['crps'] = crps
        metrics['qice'] = qice
        metrics['controls'] = {}
        for name, m_ in ctrl_m.items():
            r_ = m_.compute()
            metrics['controls'][name] = r_
            self._show_interval(
                f'Control [{name}]: conformal with sigma = ' +
                ('1 (constant width)' if name == 'const' else 'input-window std'), r_)

        with open("result.txt", 'a') as f:
            f.write(f"{setting}\n")
            f.write(f"mse:{mse:.6f}, mae:{mae:.6f}, crps:{crps:.6f}, qice:{qice:.6f}\n")
            f.write(f"[{tag}] Avg_IS:{metrics['avg_is']:.4f}, MPIW@90:{metrics['mpiw'][0.90]:.4f}, "
                    f"ECE:{metrics['ece']:.4f}\n")
            f.write(f"[raw] Avg_IS:{raw['avg_is']:.4f}, MPIW@90:{raw['mpiw'][0.90]:.4f}, "
                    f"ECE:{raw['ece']:.4f}\n")
            for name, r_ in metrics['controls'].items():
                f.write(f"[ctrl-{name}] Avg_IS:{r_['avg_is']:.4f}, MPIW@90:{r_['mpiw'][0.90]:.4f}, "
                        f"ECE:{r_['ece']:.4f}\n")
            f.write("\n")
        return mse, mae, metrics

    def _test_quantile(self, test_loader, path, setting):
        if self.cqr_calibrator is None and self.args.calibrate == 'conformal':
            vali_data, vali_loader = self._get_data(flag='val')
            self._fit_cqr_calibrator(vali_loader, path)

        mse_meter, mae_meter, crps_meter = AverageMeter(), AverageMeter(), AverageMeter()
        qice_meter = QICEMeter(n_bins=10)
        raw_m = QuantileIntervalMetrics(LEVELS)
        cqr_m = QuantileIntervalMetrics(LEVELS)

        self.model.eval()
        with torch.no_grad():
            for batch in test_loader:
                q, mu, target = self._forward_quantile(batch)
                mse_meter.update(nn.MSELoss()(mu, target).item(), mu.size(0))
                mae_meter.update(nn.L1Loss()(mu, target).item(), mu.size(0))

                crps_vals = crps_from_quantiles(target, q, QUANTILE_LEVELS)
                crps_meter.update(crps_vals.mean().item(), target.numel())
                qice_meter.update(quantile_pit(target, q, QUANTILE_LEVELS))

                bounds = {c: (q[..., lo], q[..., hi]) for c, (lo, hi) in QUANTILE_PAIR_IDX.items()}
                raw_m.update(target, bounds)
                if self.cqr_calibrator is not None:
                    cqr_bounds = {c: self.cqr_calibrator.widen(c, q[..., lo], q[..., hi])
                                 for c, (lo, hi) in QUANTILE_PAIR_IDX.items()}
                    cqr_m.update(target, cqr_bounds)

        mse, mae, crps, qice = mse_meter.avg, mae_meter.avg, crps_meter.avg, qice_meter.compute()
        os.makedirs(os.path.join('./results/', setting), exist_ok=True)
        print(f'MSE: {mse:.6f}  MAE: {mae:.6f}  CRPS: {crps:.6f}  QICE: {qice:.6f}')

        raw = raw_m.compute()
        self._show_interval('Raw quantile-regression intervals', raw)
        metrics, tag = raw, 'raw'
        if self.cqr_calibrator is not None:
            cqr = cqr_m.compute()
            self._show_interval(f'CQR-calibrated intervals ({self.cqr_calibrator.mode})', cqr)
            metrics, tag = cqr, 'cqr'
        metrics['raw'] = raw
        metrics['crps'] = crps
        metrics['qice'] = qice
        metrics['controls'] = {}   # quantile head has no scalar sigma, does not participate in const/winstd control

        with open("result.txt", 'a') as f:
            f.write(f"{setting}\n")
            f.write(f"mse:{mse:.6f}, mae:{mae:.6f}, crps:{crps:.6f}, qice:{qice:.6f}\n")
            f.write(f"[{tag}] Avg_IS:{metrics['avg_is']:.4f}, MPIW@90:{metrics['mpiw'][0.90]:.4f}, "
                    f"ECE:{metrics['ece']:.4f}\n")
            f.write(f"[raw] Avg_IS:{raw['avg_is']:.4f}, MPIW@90:{raw['mpiw'][0.90]:.4f}, "
                    f"ECE:{raw['ece']:.4f}\n\n")
        return mse, mae, metrics