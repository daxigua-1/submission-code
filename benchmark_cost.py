import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

BATCHES = (1, 8)
BATCHES_SINGLE = (1, 8, 64)


def _sync(dev):
    if str(dev).startswith('cuda'):
        torch.cuda.synchronize()


def count_params(*models):
    return sum(p.numel() for m in models for p in m.parameters()) / 1e6


@torch.no_grad()
def gflops_per_window(fn, B):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None
    try:
        with FlopCounterMode(display=False) as fc:
            fn()
        return fc.get_total_flops() / B / 1e9
    except Exception as e:
        return None


@torch.no_grad()
def peak_act_mb(fn, dev):
    if not str(dev).startswith('cuda'):
        return None
    torch.cuda.empty_cache()
    _sync(dev)
    base = torch.cuda.memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    fn()
    _sync(dev)
    return (torch.cuda.max_memory_allocated(dev) - base) / 2 ** 20


@torch.no_grad()
def time_ms_per_window(fn, B, dev, warmup=2, repeats=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        _sync(dev)
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2] / B * 1000.0


def _gpu():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'


def save(out_json, d):
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(d, f, indent=2)
    print('[cost]', json.dumps(d, indent=2))


def real_batch(pearl_args, B):
    from data_provider.data_factory import data_provider
    ds, _ = data_provider(pearl_args, 'test')
    items = [ds[i] for i in range(B)]
    return tuple(torch.from_numpy(np.stack([it[k] for it in items])).float() for k in range(4))


def bench_pearl(exp, out_json, S_note=1):
    _, loader = exp._get_data('test')
    batch = next(iter(loader))
    if batch[0].shape[0] < max(BATCHES_SINGLE):
        ds = loader.dataset
        items = [ds[i] for i in range(max(BATCHES_SINGLE))]
        batch = tuple(torch.from_numpy(np.stack([it[k] for it in items])).float() for k in range(4))
    exp.model.eval()
    dev = exp.device
    res = {}
    for B in BATCHES_SINGLE:
        b = tuple(t[:B].to(dev) for t in batch)

        def fn():
            mu, sigma, *_ = exp._forward_unc(b)
            half = 1.645 * sigma
            return (mu - half).cpu(), (mu + half).cpu()
        res[f'B{B}'] = time_ms_per_window(fn, B, dev, warmup=10, repeats=50)
    if os.environ.get('COST_PROFILE'):
        from torch.profiler import profile, ProfilerActivity
        b = tuple(t[:64].to(dev) for t in batch)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                exp._forward_unc(b)
            _sync(dev)
        print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=15))
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    b64 = tuple(t[:64].to(dev) for t in batch)
    gf = gflops_per_window(lambda: exp._forward_unc(b64), 64)
    pm = peak_act_mb(lambda: exp._forward_unc(b64), dev)
    save(out_json, dict(method='PEARL', params_M=count_params(exp.model), infer_ms_per_window=res,
                        gflops_per_window=gf, peak_act_mb_B64=pm,
                        n_samples='single pass', data=exp.args.data, pred_len=exp.args.pred_len, gpu=_gpu()))


def bench_point(a, dev):
    sys.path.insert(0, os.path.join('models', 'point_baselines'))
    from models.point_baselines.point_baseline_on_pearl_data import build_configs, build_model, forward_model
    name = {'casa': 'CASA', 'itransformer': 'iTransformer', 'patchtst': 'PatchTST'}[a.method]
    pa = pearl_args_from(a)
    configs, _ = build_configs(name, a.data, pa, None)
    model = build_model(name, configs).to(dev).eval()
    res = {}
    for B in BATCHES_SINGLE:
        x, y, xm, ym = [t.to(dev) for t in real_batch(pa, B)]

        def fn():
            out, _ = forward_model(model, name, x, xm, y, ym, configs)
            half = 1.645 * out.abs().mean()
            return (out - half).cpu(), (out + half).cpu()
        res[f'B{B}'] = time_ms_per_window(fn, B, dev, warmup=10, repeats=50)
    return dict(method=name + '+CP', params_M=count_params(model), infer_ms_per_window=res, n_samples='single pass')


# --------------------------------------------------------------------------- TMDM
def bench_tmdm(a, dev):
    here = os.path.join('models', 'TMDM')
    sys.path.insert(0, here)
    from models.TMDM.tmdm_on_pearl_data import TMDMOnPearlData, build_tmdm_args
    cfg = a.tmdm_config
    if not os.path.exists(cfg):
        cfg = os.path.join(here, cfg)
    targs = build_tmdm_args(gpu=int(str(a.device).split(':')[-1]) if ':' in str(a.device) else 0,
                            seq_len=96, label_len=48, pred_len=a.pred_len, enc_in=a.enc_in, dec_in=a.enc_in,
                            c_out=a.enc_in, features='M', freq=a.freq, embed='timeF', diffusion_config_dir=cfg)
    runner = TMDMOnPearlData(targs, n_z_samples=a.n_samples)
    runner.model.eval(); runner.cond_pred_model.eval()
    pa = pearl_args_from(a)
    res = {}
    for B in BATCHES:
        bx, by, bxm, bym = [t.to(dev) for t in real_batch(pa, B)]
        res[f'B{B}'] = time_ms_per_window(lambda: runner._sample_batch(bx, by, bxm, bym), B, dev,
                                          warmup=1, repeats=3)
    steps = int(runner.model.num_timesteps)
    return dict(method='TMDM', params_M=count_params(runner.model, runner.cond_pred_model),
                infer_ms_per_window=res, n_samples=a.n_samples, diffusion_steps=steps)


# --------------------------------------------------------------------------- NsDiff
def bench_nsdiff(a, dev):
    here = os.path.join('models', 'NsDiff')
    sys.path.insert(0, here)
    import models.NsDiff.nsdiff_on_pearl_data as N
    root, cwd = N._find_nsdiff_root(os.path.abspath(here)), os.getcwd()
    os.chdir(root)
    try:
        exp = N.NsDiffForecast(dataset_type=a.data, device=a.device, horizon=1, windows=96,
                               pred_len=a.pred_len, rolling_length=24, load_pretrain=False,
                               epochs=1, patience=1)
        exp._setup_run(2021)
    finally:
        os.chdir(cwd)
    N.set_mode(exp, False)
    N.set_num_samples(exp, a.n_samples)
    pa = pearl_args_from(a)
    res = {}
    for B in BATCHES:
        batch = real_batch(pa, B)
        res[f'B{B}'] = time_ms_per_window(
            lambda: N.sample_batch(exp, batch, exp.device, a.pred_len, a.n_samples), B, dev, warmup=1, repeats=3)
    t = N._find_testing_cfg(exp)
    return dict(method='NsDiff', params_M=count_params(exp.model, exp.cond_pred_model, exp.cond_pred_model_g),
                infer_ms_per_window=res, n_samples=a.n_samples,
                diffusion_steps=int(getattr(exp.model, 'num_timesteps', -1)))


# --------------------------------------------------------------------------- Diffusion-TS
def bench_diffusionts(a, dev):
    here = os.path.join('models', 'DiffusionTS')
    sys.path.insert(0, here)
    from models.DiffusionTS.diffusionts_on_pearl_data import (UnitScaler, ForecastSampler, build_model, load_cfg, load_repo)
    from data_provider.data_factory import data_provider
    repo_root = os.path.abspath(here)
    repo, cfg = load_repo(repo_root), load_cfg(None, repo_root)
    H, P, N = 96, a.pred_len, a.enc_in
    W = H + P
    pa = pearl_args_from(a)
    train_set, _ = data_provider(pa, 'train')
    test_set, _ = data_provider(pa, 'test')
    scaler = UnitScaler(0.0).fit(train_set.data_x)
    model = build_model(repo, cfg, W, N, dev).eval().requires_grad_(False)
    smp = cfg['sampling']
    sampler = ForecastSampler(model, H, int(smp['sampling_steps']), float(smp['coef']), float(smp['step_size']),
                              eta=0.0, sample_batch=256)
    res = {}
    for B in BATCHES:
        win = np.stack([test_set.data_x[j:j + W] for j in range(B)])
        unit = torch.from_numpy(scaler.to_unit(win).astype(np.float32)).to(dev)
        res[f'B{B}'] = time_ms_per_window(lambda: sampler(unit, a.n_samples), B, dev, warmup=1, repeats=3)
    return dict(method='Diffusion-TS', params_M=count_params(model), infer_ms_per_window=res,
                n_samples=a.n_samples, diffusion_steps=int(smp['sampling_steps']))


def pearl_args_from(a):
    return SimpleNamespace(root_path=os.path.abspath(a.root_path), data_path=a.data + '.csv', data=a.data,
                           features='M', target='OT', seq_len=96, label_len=48, pred_len=a.pred_len,
                           enc_in=a.enc_in, embed='timeF', freq=a.freq, batch_size=32, num_workers=0,
                           seasonal_patterns='Monthly')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True,
                    choices=['casa', 'itransformer', 'patchtst', 'tmdm', 'nsdiff', 'diffusionts'])
    ap.add_argument('--data', default='ETTh1')
    ap.add_argument('--root_path', default='./dataset/ETT-small/')
    ap.add_argument('--pred_len', type=int, default=192)
    ap.add_argument('--enc_in', type=int, default=7)
    ap.add_argument('--freq', default='h')
    ap.add_argument('--n_samples', type=int, default=20)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--tmdm_config', default='./model9_NS_transformer/configs/toy_8gauss.yml')
    ap.add_argument('--out_dir', default='./cost')
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    fn = {'casa': bench_point, 'itransformer': bench_point, 'patchtst': bench_point,
          'tmdm': bench_tmdm, 'nsdiff': bench_nsdiff, 'diffusionts': bench_diffusionts}[a.method]
    d = fn(a, dev)
    d.update(data=a.data, pred_len=a.pred_len, gpu=_gpu(), batches=sorted(int(k[1:]) for k in d['infer_ms_per_window']))
    save(os.path.join(a.out_dir, f"cost_{a.method}_{a.data}_pl{a.pred_len}.json"), d)


if __name__ == '__main__':
    main()