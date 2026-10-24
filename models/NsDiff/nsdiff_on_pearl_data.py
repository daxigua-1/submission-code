import json
import math
import os
import random
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
import types


def _install_windows_stubs():
    try:
        import resource  # noqa: F401
    except ImportError:
        res = types.ModuleType('resource')
        res.RLIMIT_NOFILE = 7
        res.getrlimit = lambda *a, **k: (1024, 1024)
        res.setrlimit = lambda *a, **k: None

        def _res_getattr(name):
            if name.startswith(('RLIMIT_', 'RUSAGE_')):
                return 0
            raise AttributeError(name)
        res.__getattr__ = _res_getattr
        sys.modules['resource'] = res
    try:
        import wandb  # noqa: F401
    except ImportError:
        sys.modules['wandb'] = types.ModuleType('wandb')


_install_windows_stubs()

from src.experiments.NsDiff import NsDiffForecast
from data_provider.data_factory import data_provider

NsDiffForecast._init_metrics = lambda self: None

CRPS_LEVELS = tuple(i / 20 for i in range(1, 20))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def fixed_sampling_rng(seed: int, device):
    dev = torch.device(device)
    cuda_ids = []
    if dev.type == 'cuda':
        cuda_ids = [dev.index if dev.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=cuda_ids):
        torch.manual_seed(seed)
        yield


def set_mode(exp, train: bool):
    for m in (exp.model, exp.cond_pred_model, exp.cond_pred_model_g):
        m.train(train)


def get_optimizer(exp):
    opt = getattr(exp, 'model_optim', None)
    if isinstance(opt, torch.optim.Optimizer):
        return opt
    cands = [k for k, v in vars(exp).items() if isinstance(v, torch.optim.Optimizer)]
    if len(cands) == 1:
        return getattr(exp, cands[0])


def _find_testing_cfg(exp):
    for name in ('diffusion_config', 'config', 'args'):
        t = getattr(getattr(exp, name, None), 'testing', None)
        if t is not None and hasattr(t, 'n_z_samples'):
            return t
    for v in vars(exp).values():
        t = getattr(v, 'testing', None)
        if t is not None and hasattr(t, 'n_z_samples'):
            return t


def set_num_samples(exp, S: int):
    t = _find_testing_cfg(exp)
    mini = int(getattr(t, 'minisample', S))
    if S % mini != 0:
        mini = math.gcd(S, mini)
    t.n_z_samples = int(S)
    t.minisample = int(mini)
    if int(getattr(t, 'n_z_samples_depart', 1)) != 1:
        t.n_z_samples_depart = 1


def to_nsdiff_batch(batch, device, pred_len: int):
    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    return (batch_x.float().to(device),
            batch_y[:, -pred_len:, :].float().to(device),
            batch_x_mark.float().to(device),
            batch_y_mark[:, -pred_len:, :].float().to(device))


def build_pearl_loaders(pearl_args):
    _, train_loader = data_provider(pearl_args, 'train')
    val_set, _ = data_provider(pearl_args, 'val')
    test_set, _ = data_provider(pearl_args, 'test')

    def seq(ds):
        return DataLoader(ds, batch_size=pearl_args.batch_size, shuffle=False,
                          num_workers=pearl_args.num_workers, drop_last=False)
    return train_loader, seq(val_set), seq(test_set)


@torch.no_grad()
def sample_batch(exp, batch, device, pred_len: int, S: int):
    bx, by, bxm, bym = to_nsdiff_batch(batch, device, pred_len)
    outs, truths = exp._process_val_batch(bx, by, bxm, bym)     # outs: [B,P,N,S]
    outs = torch.as_tensor(outs).detach().cpu()
    samples = outs.permute(0, 3, 1, 2).contiguous().float()
    if samples.size(1) != S:
        raise RuntimeError(f"Actual sample count={samples.size(1)}, expected S={S}; "
                           f"set_num_samples did not modify the effective configuration. Please check the source of NsDiff's sampling config.")
    return samples, truths.detach().cpu().float()

def _sorted_quantiles(xs, levels):
    S = xs.size(1)
    pos = (torch.tensor(list(levels), dtype=torch.float64) * (S + 1) - 1.0).clamp(0.0, S - 1.0)
    lo, hi = pos.floor().long(), pos.ceil().long()
    w = (pos - pos.floor()).to(xs.dtype).view(1, -1, 1, 1)
    q = xs[:, lo] * (1.0 - w) + xs[:, hi] * w
    return q.permute(0, 2, 3, 1)


def crps_sum_from_samples(samples, y):
    xs, _ = torch.sort(samples, dim=1)
    q = _sorted_quantiles(xs, CRPS_LEVELS)
    tau = torch.tensor(CRPS_LEVELS, dtype=q.dtype).view(1, 1, 1, -1)
    err = y.unsqueeze(-1) - q
    pin = torch.maximum(tau * err, (tau - 1.0) * err)
    return (2.0 * pin.mean(-1)).double().sum().item()


@torch.no_grad()
def validate(exp, loader, device, pred_len, S, select_by, seed):
    set_mode(exp, False)
    set_num_samples(exp, S)
    se, crps, n = 0.0, 0.0, 0
    with fixed_sampling_rng(seed, device):
        for batch in loader:
            samples, y = sample_batch(exp, batch, device, pred_len, S)
            mu = samples.mean(dim=1)
            se += ((mu - y) ** 2).double().sum().item()
            if select_by == 'crps':
                crps += crps_sum_from_samples(samples, y)
            n += y.numel()
    return {'mse': se / n, 'crps': (crps / n) if select_by == 'crps' else float('nan')}


@torch.no_grad()
def export_split(exp, loader, tag, device, pred_len, n_channels, S, seed, out_prefix):
    set_mode(exp, False)
    set_num_samples(exp, S)
    n = len(loader.dataset)
    p_s, p_t = f"{out_prefix}_{tag}_samples.npy", f"{out_prefix}_{tag}_target.npy"
    mm_s = np.lib.format.open_memmap(p_s, mode='w+', dtype=np.float32,
                                     shape=(n, S, pred_len, n_channels))
    mm_t = np.lib.format.open_memmap(p_t, mode='w+', dtype=np.float32,
                                     shape=(n, pred_len, n_channels))
    i = 0
    with fixed_sampling_rng(seed, device):
        for batch in tqdm(loader, desc=f"export {tag}"):
            samples, y = sample_batch(exp, batch, device, pred_len, S)
            b = samples.size(0)
            assert tuple(samples.shape[1:]) == (S, pred_len, n_channels), samples.shape
            mm_s[i:i + b] = samples.numpy()
            mm_t[i:i + b] = y.numpy()
            i += b
    assert i == n, f"{i} != {n}"
    mm_s.flush(); mm_t.flush()
    del mm_s, mm_t
    print(f"{tag}: samples ({n},{S},{pred_len},{n_channels}) -> {p_s}")


def _find_nsdiff_root(hint=None):
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (hint, here, os.getcwd()):
        if cand and os.path.exists(os.path.join(cand, 'configs', 'nsdiff.yml')):
            return os.path.abspath(cand)
    raise FileNotFoundError("Not found")


def run(nsdiff_kwargs: dict, pearl_args, seed: int, epochs: int, patience: int,
        ckpt_dir: str, out_prefix: str, n_samples: int = 100,
        val_n_samples: int = None, select_by: str = 'mse', nsdiff_root: str = None, **extra_kwargs):
    assert select_by in ('mse', 'crps')
    val_S = val_n_samples or n_samples
    P, N = pearl_args.pred_len, pearl_args.enc_in

    set_seed(seed)
    pearl_args.root_path = os.path.abspath(pearl_args.root_path)
    root, cwd = _find_nsdiff_root(nsdiff_root), os.getcwd()
    os.chdir(root)
    try:
        exp = NsDiffForecast(**nsdiff_kwargs)

        def _dummy_init_data_loader():
            exp.dataset = types.SimpleNamespace(
                num_features=int(pearl_args.enc_in),
                freq=str(pearl_args.freq),
                name=str(pearl_args.data),
            )
        exp._init_data_loader = _dummy_init_data_loader

        exp._setup_run(seed)
    finally:
        os.chdir(cwd)
    device = exp.device
    optimizer = get_optimizer(exp)

    train_loader, val_loader, test_loader = build_pearl_loaders(pearl_args)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"nsdiff_pearl_data_seed{seed}.pt")

    best, best_epoch, no_improve, epochs_run = float('inf'), -1, 0, 0
    for epoch in range(epochs):
        set_mode(exp, True)
        losses = []
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch + 1}")):
            bx, by, bxm, bym = to_nsdiff_batch(batch, device, P)
            if epoch == 0 and step == 0:
                print(f"[check] x {tuple(bx.shape)} | y {tuple(by.shape)} | "
                      f"x_mark {tuple(bxm.shape)} | y_mark {tuple(bym.shape)}  "
                      f"(pred_len={P})")
            optimizer.zero_grad()
            loss = exp._process_train_batch(bx, by, bxm, bym)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epochs_run = epoch + 1

        v = validate(exp, val_loader, device, P, val_S, select_by, seed + 1000)
        score = v[select_by]
        print(f"[epoch {epoch + 1}] train loss = {np.mean(losses):.6f} | "
              f"val MSE = {v['mse']:.6f}" +
              (f" | val CRPS = {v['crps']:.6f}" if select_by == 'crps' else "") +
              f" | select_by={select_by}")

        if score < best:
            best, best_epoch, no_improve = score, epoch + 1, 0
            torch.save({'model': exp.model.state_dict(),
                        'cond_pred_model': exp.cond_pred_model.state_dict(),
                        'cond_pred_model_g': exp.cond_pred_model_g.state_dict()}, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"early stopping at epoch {epoch + 1}")
                break

    ckpt = torch.load(ckpt_path, map_location=device)
    exp.model.load_state_dict(ckpt['model'])
    exp.cond_pred_model.load_state_dict(ckpt['cond_pred_model'])
    exp.cond_pred_model_g.load_state_dict(ckpt['cond_pred_model_g'])

    export_split(exp, val_loader, 'val', device, P, N, n_samples, seed + 1, out_prefix)
    export_split(exp, test_loader, 'test', device, P, N, n_samples, seed + 2, out_prefix)

    with open(f"{out_prefix}_meta.json", 'w') as f:
        json.dump({'seed': seed, 'n_samples': n_samples, 'val_n_samples': val_S,
                   'select_by': select_by, 'best_val': best, 'best_epoch': best_epoch,
                   'epochs_run': epochs_run, 'pred_len': P, 'n_channels': N}, f, indent=2)