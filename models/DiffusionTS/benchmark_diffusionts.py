import argparse
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffusionts_on_pearl_data import (UnitScaler, WindowDataset, ForecastSampler, build_model,  # noqa: E402
                                       load_cfg, load_repo, load_sampler_model, pick_indices, set_seed)
from data_provider.data_factory import data_provider                                              # noqa: E402

N_TEST_ETTH, N_TEST_ETTM = 2689, 11329


def _sync(dev):
    if dev.type == 'cuda':
        torch.cuda.synchronize()


def time_train(repo, cfg, train_unit, W, N, dev, n_steps=4):
    model = build_model(repo, cfg, W, N, dev)
    model.train()
    opt = Adam(model.parameters(), lr=1e-5, betas=[0.9, 0.96])
    bs, accum = cfg['dataloader']['batch_size'], cfg['solver']['gradient_accumulate_every']
    dl = DataLoader(WindowDataset(train_unit, W), batch_size=bs, shuffle=True, drop_last=True)
    it = iter(dl)

    def one_step():
        for _ in range(accum):
            data = next(it).to(dev)
            (model(data, target=data) / accum).backward()
        opt.step()
        opt.zero_grad()
    one_step()
    _sync(dev)
    t0 = time.time()
    for _ in range(n_steps):
        one_step()
    _sync(dev)
    per_step = (time.time() - t0) / n_steps
    total = cfg['solver']['max_steps']
    print(f"{per_step:.3f} s/step（batch={bs}×{accum}）-> {total} {per_step * total / 60:.0f}")
    del model, opt
    if dev.type == 'cuda':
        torch.cuda.empty_cache()


def time_sampling(sampler, unit_win, dev, sizes):
    results = []
    for bs in sizes:
        xb = unit_win[:1].repeat(bs, 1, 1)
        try:
            if dev.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            sampler.run_chunk(xb[:min(bs, 4)])
            _sync(dev)
            t0 = time.time()
            sampler.run_chunk(xb)
            _sync(dev)
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 2 ** 20 if dev.type == 'cuda' else float('nan')
            results.append((bs, dt, dt / bs, peak))
            print(f"{bs:4d}：{dt:7.2f} s，{dt / bs * 1000:8.1f}，{peak:8.0f} MiB")
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"{bs:4d}")
                if dev.type == 'cuda':
                    torch.cuda.empty_cache()
                break
            raise
    return results


def extrapolate(sec_per_seq, S, val_windows):
    rows = []
    for name, n_test in (('ETTh', N_TEST_ETTH), ('ETTm', N_TEST_ETTM)):
        n_seq = (n_test + val_windows) * S
        hours = n_seq * sec_per_seq / 3600
        rows.append((name, n_test, hours))
    total = 3 * (2 * rows[0][2] + 2 * rows[1][2])



def eta_compare(repo, cfg, model, H, unit_win, scaler, tgt_win, S, dev, sample_batch, steps, coef, step_size):
    ideal = np.sqrt((S + 1) / S)
    for eta in (0.0, 1.0):
        set_seed(0)
        sm = ForecastSampler(model, H, steps, coef, step_size, eta=eta, sample_batch=sample_batch)
        pred_u = sm(unit_win, S).numpy().astype(np.float64)
        pred = scaler.from_unit(pred_u)                                    # [n,S,P,N]
        spread = pred.std(axis=1, ddof=1).mean()
        rmse = np.sqrt(((pred.mean(axis=1) - tgt_win) ** 2).mean())
        print(f"  {eta:5.1f} {spread:10.4f} {rmse:12.4f} {spread / rmse:10.3f} {ideal:8.3f}")
    model.eta = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo_root', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--config', default=None)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--n_samples', type=int, default=100)
    ap.add_argument('--val_windows', type=int, default=300)
    ap.add_argument('--batch_sizes', default='64,128,256,512,1024')
    ap.add_argument('--eta_windows', type=int, default=8)
    ap.add_argument('--eta_samples', type=int, default=50)
    ap.add_argument('--margin', type=float, default=0.0)
    ap.add_argument('--skip_train_timing', action='store_true')
    ap.add_argument('--pearl_root_path', required=True)
    ap.add_argument('--pearl_data_path', required=True)
    ap.add_argument('--pearl_data', required=True)
    ap.add_argument('--pearl_features', default='M')
    ap.add_argument('--pearl_target', default='OT')
    ap.add_argument('--pearl_seq_len', type=int, default=96)
    ap.add_argument('--pearl_label_len', type=int, default=48)
    ap.add_argument('--pearl_pred_len', type=int, required=True)
    ap.add_argument('--pearl_enc_in', type=int, required=True)
    ap.add_argument('--pearl_embed', default='timeF')
    ap.add_argument('--pearl_freq', default='h')
    ap.add_argument('--pearl_batch_size', type=int, default=32)
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    repo = load_repo(a.repo_root)
    cfg = load_cfg(a.config, a.repo_root)
    pa = SimpleNamespace(root_path=os.path.abspath(a.pearl_root_path), data_path=a.pearl_data_path,
                         data=a.pearl_data, features=a.pearl_features, target=a.pearl_target,
                         seq_len=a.pearl_seq_len, label_len=a.pearl_label_len, pred_len=a.pearl_pred_len,
                         enc_in=a.pearl_enc_in, embed=a.pearl_embed, freq=a.pearl_freq,
                         batch_size=a.pearl_batch_size, num_workers=0, seasonal_patterns='Monthly')
    H, P, N = pa.seq_len, pa.pred_len, pa.enc_in
    W = H + P

    train_set, _ = data_provider(pa, 'train')
    val_set, _ = data_provider(pa, 'val')
    scaler = UnitScaler(a.margin).fit(train_set.data_x)

    if not a.skip_train_timing:
        time_train(repo, cfg, scaler.to_unit(train_set.data_x), W, N, dev)

    if a.ckpt:
        model = load_sampler_model(repo, cfg, W, N, a.ckpt, dev)
    else:
        model = build_model(repo, cfg, W, N, dev).eval().requires_grad_(False)

    smp = cfg['sampling']
    steps, coef, step_size = int(smp['sampling_steps']), float(smp['coef']), float(smp['step_size'])
    idx = pick_indices(len(val_set), a.eta_windows)
    win = np.stack([val_set.data_x[j:j + W] for j in idx])
    unit = torch.from_numpy(scaler.to_unit(win).astype(np.float32)).to(dev)

    sizes = [int(s) for s in a.batch_sizes.split(',')]
    sampler = ForecastSampler(model, H, steps, coef, step_size, eta=0.0, sample_batch=max(sizes))
    res = time_sampling(sampler, unit, dev, sizes)
    if res:
        best = min(r[2] for r in res)
        extrapolate(best, a.n_samples, a.val_windows)
        best_bs = min(res, key=lambda r: r[2])[0]
    else:
        best_bs = 64

    if a.ckpt:
        eta_compare(repo, cfg, model, H, unit, scaler, win[:, H:, :], a.eta_samples, dev, best_bs,
                    steps, coef, step_size)


if __name__ == '__main__':
    main()