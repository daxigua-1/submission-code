import copy
import importlib.util
import json
import os
import random
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_provider.data_factory import data_provider

DEFAULT_CFG = {
    'model': dict(n_layer_enc=3, n_layer_dec=2, d_model=64, timesteps=500, sampling_timesteps=500,
                  loss_type='l1', beta_schedule='cosine', n_heads=4, mlp_hidden_times=4,
                  attn_pd=0.0, resid_pd=0.0, kernel_size=1, padding_size=0),
    'solver': dict(base_lr=1.0e-5, max_steps=18000, gradient_accumulate_every=2, save_cycle=1800,
                   ema_decay=0.995, ema_update_every=10,
                   scheduler=dict(factor=0.5, patience=4000, min_lr=1.0e-5, threshold=1.0e-1,
                                  threshold_mode='rel', warmup_lr=8.0e-4, warmup=500, verbose=False)),
    'dataloader': dict(batch_size=128, shuffle=True),
    'sampling': dict(coef=1.0e-2, step_size=5.0e-2, sampling_steps=200),
}


def cfg_from_yaml(path):
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        y = yaml.safe_load(f)
    model = dict(y['model']['params'])
    model.pop('seq_length', None)
    model.pop('feature_size', None)
    s = y['solver']
    td = y['dataloader']['test_dataset']
    return {
        'model': model,
        'solver': dict(base_lr=s.get('base_lr', 1.0e-4), max_steps=s['max_epochs'],
                       gradient_accumulate_every=s['gradient_accumulate_every'],
                       save_cycle=s['save_cycle'], ema_decay=s['ema']['decay'],
                       ema_update_every=s['ema']['update_interval'],
                       scheduler=dict(s['scheduler']['params'])),
        'dataloader': dict(batch_size=y['dataloader']['batch_size'],
                           shuffle=y['dataloader'].get('shuffle', True)),
        'sampling': dict(coef=td['coefficient'], step_size=td['step_size'],
                         sampling_steps=td['sampling_steps']),
    }


def load_cfg(path=None, repo_root=None):
    if path is None and repo_root:
        cand = os.path.join(repo_root, 'Config', 'etth.yaml')
        if os.path.exists(cand):
            path = cand
    if path is None:
        return copy.deepcopy(DEFAULT_CFG)
    return cfg_from_yaml(path)

def load_repo(repo_root):
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import Models.interpretable_diffusion.gaussian_diffusion as gd
    gd.tqdm = lambda it=None, *a, **k: it
    spec = importlib.util.spec_from_file_location(
        'diffusionts_lr_sch', os.path.join(repo_root, 'engine', 'lr_sch.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {'Diffusion_TS': gd.Diffusion_TS, 'Scheduler': mod.ReduceLROnPlateauWithWarmup}


def build_model(repo, cfg, window, n_feat, device):
    return repo['Diffusion_TS'](seq_length=window, feature_size=n_feat, **cfg['model']).to(device)


def _load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def fixed_sampling_rng(seed: int, device):
    dev = torch.device(device)
    ids = []
    if dev.type == 'cuda':
        ids = [dev.index if dev.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=ids):
        torch.manual_seed(seed)
        yield


class UnitScaler:

    def __init__(self, margin: float = 0.0):
        self.margin = float(margin)

    def fit(self, train_x: np.ndarray):
        lo, hi = train_x.min(axis=0), train_x.max(axis=0)
        rng = np.maximum(hi - lo, 1e-8)
        self.lo = lo - self.margin * rng
        self.hi = hi + self.margin * rng
        return self

    def to_unit(self, x):
        return 2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0

    def from_unit(self, u):
        return (u + 1.0) * 0.5 * (self.hi - self.lo) + self.lo


class WindowDataset(Dataset):

    def __init__(self, data_unit: np.ndarray, window: int):
        self.data = torch.from_numpy(np.ascontiguousarray(data_unit, dtype=np.float32))
        self.window = window

    def __len__(self):
        return self.data.size(0) - self.window + 1

    def __getitem__(self, i):
        return self.data[i:i + self.window]


def pick_indices(n: int, max_windows=None, stride: int = 1):
    idx = np.arange(0, n, max(1, int(stride)))
    if max_windows is not None and len(idx) > max_windows:
        idx = np.unique(np.linspace(0, n - 1, int(max_windows)).round().astype(int))
    return idx


def check_windows(dataset, hist_len, pred_len):
    seq_x, seq_y, _, _ = dataset[0]
    full = np.concatenate([np.asarray(seq_x), np.asarray(seq_y)[-pred_len:]], axis=0)
    W = hist_len + pred_len
    assert np.allclose(full, dataset.data_x[:W]), "The PEARL window is inconsistent with the `data_x` slice; please check `label_len`/`seq_len`."


def train_model(repo, cfg, train_unit, window, device, run_dir, seed, max_steps=None,
                resume=True, log_every=200):
    from ema_pytorch import EMA
    set_seed(seed)
    n_feat = train_unit.shape[1]
    model = build_model(repo, cfg, window, n_feat, device)
    sol = cfg['solver']
    total = int(max_steps or sol['max_steps'])
    accum = int(sol['gradient_accumulate_every'])

    shuffle = cfg['dataloader']['shuffle']
    dl = DataLoader(WindowDataset(train_unit, window), batch_size=cfg['dataloader']['batch_size'],
                    shuffle=shuffle, num_workers=0, pin_memory=(device.type == 'cuda'),
                    drop_last=shuffle)

    def cycle(d):
        while True:
            for b in d:
                yield b
    it = cycle(dl)

    opt = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=sol['base_lr'], betas=[0.9, 0.96])
    ema = EMA(model, beta=sol['ema_decay'], update_every=sol['ema_update_every']).to(device)
    sch = repo['Scheduler'](optimizer=opt, **sol['scheduler'])

    os.makedirs(run_dir, exist_ok=True)
    resume_path = os.path.join(run_dir, 'resume.pt')
    step = 0
    if resume and os.path.exists(resume_path):
        d = _load(resume_path, device)
        model.load_state_dict(d['model'])
        opt.load_state_dict(d['opt'])
        ema.load_state_dict(d['ema'])
        sch.load_state_dict(d['sch'])
        step = int(d['step'])
        print(f"[train]{resume_path}，step={step}")

    def save_resume():
        torch.save({'step': step, 'model': model.state_dict(), 'opt': opt.state_dict(),
                    'ema': ema.state_dict(), 'sch': sch.state_dict()}, resume_path)

    model.train()
    t0, loss_val = time.time(), float('nan')
    pbar = tqdm(total=total, initial=step, desc='train Diffusion-TS')
    while step < total:
        total_loss = 0.0
        for _ in range(accum):
            data = next(it).to(device)
            loss = model(data, target=data) / accum
            loss.backward()
            total_loss += loss.item()
        clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step(total_loss)
        opt.zero_grad()
        ema.update()
        step += 1
        loss_val = total_loss
        pbar.update(1)
        if step % log_every == 0:
            pbar.set_postfix(loss=f"{loss_val:.5f}", lr=f"{opt.param_groups[0]['lr']:.2e}")
        if step % int(sol['save_cycle']) == 0 and step < total:
            save_resume()
    pbar.close()

    final_path = os.path.join(run_dir, 'final.pt')
    torch.save({'ema_model': ema.ema_model.state_dict(), 'model': model.state_dict(), 'step': step,
                'cfg': cfg, 'train_seconds': time.time() - t0}, final_path)
    print(f"[train]，step={step}，final loss={loss_val:.5f}，{(time.time() - t0) / 60:.1f} min")
    return final_path


def load_sampler_model(repo, cfg, window, n_feat, final_path, device):
    d = _load(final_path, device)
    model = build_model(repo, cfg, window, n_feat, device)
    model.load_state_dict(d['ema_model'])
    model.eval().requires_grad_(False)
    return model


class ForecastSampler:
    def __init__(self, model, hist_len, steps, coef, step_size, eta=0.0, sample_batch=256):
        self.model, self.hist_len, self.steps = model, int(hist_len), int(steps)
        self.coef, self.step_size = float(coef), float(step_size)
        self.eta, self.sample_batch = float(eta), int(sample_batch)
        self.model.eta = self.eta

    def run_chunk(self, xb):
        mask = torch.zeros_like(xb, dtype=torch.bool)
        mask[:, :self.hist_len, :] = True
        target = xb * mask
        kw = {'coef': self.coef, 'learning_rate': self.step_size}
        with torch.no_grad():
            if self.steps == self.model.num_timesteps:
                return self.model.sample_infill(shape=xb.shape, target=target, partial_mask=mask, model_kwargs=kw)
            return self.model.fast_sample_infill(shape=xb.shape, target=target, partial_mask=mask,
                                                 model_kwargs=kw, sampling_timesteps=self.steps)

    @torch.no_grad()
    def __call__(self, x, S):
        n, W, N = x.shape
        P = W - self.hist_len
        g = max(1, self.sample_batch // S)
        outs = []
        for i in range(0, n, g):
            xb = x[i:i + g].repeat_interleave(S, dim=0)
            res, j, total = [], 0, xb.size(0)
            while j < total:
                cs = min(self.sample_batch, total - j)
                try:
                    res.append(self.run_chunk(xb[j:j + cs]))
                    j += cs
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower() and self.sample_batch > 1:
                        torch.cuda.empty_cache()
                        self.sample_batch = max(1, self.sample_batch // 2)
                        continue
                    raise
            r = torch.cat(res, 0)[:, self.hist_len:, :]
            outs.append(r.reshape(-1, S, P, N).cpu())
        return torch.cat(outs, 0)

def export_split(sampler, dataset, idx, scaler, tag, out_prefix, S, hist_len, pred_len, device,
                 seed, win_chunk=8):
    data_x = dataset.data_x
    W, N = hist_len + pred_len, data_x.shape[1]
    n = len(idx)
    p_s, p_t = f"{out_prefix}_{tag}_samples.npy", f"{out_prefix}_{tag}_target.npy"
    mm_s = np.lib.format.open_memmap(p_s, mode='w+', dtype=np.float32, shape=(n, S, pred_len, N))
    mm_t = np.lib.format.open_memmap(p_t, mode='w+', dtype=np.float32, shape=(n, pred_len, N))
    t0 = time.time()
    with fixed_sampling_rng(seed, device):
        for a in tqdm(range(0, n, win_chunk), desc=f"export {tag}"):
            b = min(n, a + win_chunk)
            win = np.stack([data_x[j:j + W] for j in idx[a:b]])
            unit = torch.from_numpy(scaler.to_unit(win).astype(np.float32)).to(device)
            pred_u = sampler(unit, S).numpy()
            mm_s[a:b] = scaler.from_unit(pred_u.astype(np.float64)).astype(np.float32)
            mm_t[a:b] = win[:, hist_len:, :].astype(np.float32)
    mm_s.flush()
    mm_t.flush()
    del mm_s, mm_t
    secs = time.time() - t0
    print(f"{tag}: samples ({n},{S},{pred_len},{N}) -> {p_s}  | {secs / 60:.1f} min "
          f"({secs / max(n, 1):.2f})")
    return secs


def run(pearl_args, repo_root, cfg, seed, out_prefix, ckpt_dir, n_samples=100, val_windows=300,
        test_stride=1, test_windows=None, margin=0.0, eta=0.0, sample_batch=256,
        sampling_steps=None, coef=None, step_size=None, max_steps=None, skip_train=False,
        device='cuda:0'):
    assert pearl_args.features == 'M', "only support features=M"
    set_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    repo = load_repo(repo_root)
    pearl_args.root_path = os.path.abspath(pearl_args.root_path)

    train_set, _ = data_provider(pearl_args, 'train')
    val_set, _ = data_provider(pearl_args, 'val')
    test_set, _ = data_provider(pearl_args, 'test')
    H, P, N = pearl_args.seq_len, pearl_args.pred_len, pearl_args.enc_in
    W = H + P
    for ds in (train_set, val_set, test_set):
        check_windows(ds, H, P)

    scaler = UnitScaler(margin).fit(train_set.data_x)
    run_dir = os.path.join(ckpt_dir, f"diffusionts_{pearl_args.data}_pl{P}_s{seed}")
    final_path = os.path.join(run_dir, 'final.pt')
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)

    if os.path.exists(final_path):
        print(f"[train] {final_path}")
    elif skip_train:
        raise FileNotFoundError(f"--skip_train can not find {final_path}")
    else:
        train_model(repo, cfg, scaler.to_unit(train_set.data_x), W, dev, run_dir, seed, max_steps=max_steps)

    model = load_sampler_model(repo, cfg, W, N, final_path, dev)
    smp = cfg['sampling']
    steps = int(sampling_steps or smp['sampling_steps'])
    coef = float(coef if coef is not None else smp['coef'])
    step_size = float(step_size if step_size is not None else smp['step_size'])
    sampler = ForecastSampler(model, H, steps, coef, step_size, eta=eta, sample_batch=sample_batch)

    val_idx = pick_indices(len(val_set), max_windows=val_windows)
    test_idx = pick_indices(len(test_set), max_windows=test_windows, stride=test_stride)
    print(f"[export] val{len(val_idx)}/{len(val_set)}，test {len(test_idx)}/{len(test_set)}，"
          f"S={n_samples}，steps={steps}，eta={eta}，coef={coef}，step_size={step_size}")

    s_val = export_split(sampler, val_set, val_idx, scaler, 'val', out_prefix, n_samples, H, P, dev, seed + 1)
    s_test = export_split(sampler, test_set, test_idx, scaler, 'test', out_prefix, n_samples, H, P, dev, seed + 2)

    with open(f"{out_prefix}_meta.json", 'w') as f:
        json.dump({'model': 'Diffusion-TS', 'seed': seed, 'n_samples': n_samples, 'pred_len': P,
                   'hist_len': H, 'n_channels': N, 'sampling_steps': steps, 'eta': eta, 'coef': coef,
                   'step_size': step_size, 'margin': margin, 'sample_batch_final': sampler.sample_batch,
                   'val_windows_used': int(len(val_idx)), 'val_windows_total': int(len(val_set)),
                   'test_windows_used': int(len(test_idx)), 'test_windows_total': int(len(test_set)),
                   'test_stride': test_stride, 'train_steps': int(max_steps or cfg['solver']['max_steps']),
                   'export_seconds': {'val': s_val, 'test': s_test}}, f, indent=2)