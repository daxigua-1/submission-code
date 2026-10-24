import importlib
import json
import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_provider.data_factory import data_provider

NATIVE = {
    'iTransformer': {
        'common': dict(e_layers=2, n_heads=8, factor=1, dropout=0.1, activation='gelu',
                       use_norm=True, class_strategy='projection', output_attention=False,
                       d_model=128, d_ff=128, lr=1e-4, batch_size=32, lradj='type1'),
        'ETTh1': dict(d_model=256, d_ff=256),
    },
    'PatchTST': {
        'common': dict(e_layers=3, n_heads=16, d_model=128, d_ff=256, dropout=0.2, fc_dropout=0.2,
                       head_dropout=0.0, individual=0, patch_len=16, stride=8, padding_patch='end',
                       revin=1, affine=0, subtract_last=0, decomposition=0, kernel_size=25,
                       lr=1e-4, batch_size=128, lradj='TST'),
        'ETTh1': dict(n_heads=4, d_model=16, d_ff=128, dropout=0.3, fc_dropout=0.3),
        'ETTh2': dict(n_heads=4, d_model=16, d_ff=128, dropout=0.3, fc_dropout=0.3),
    },

    'CASA': {
        'common': dict(d_model=512, e_layers=3, kernel=3, lr=2.4e-4, batch_size=8, lradj='type1'),
    },

    'DLinear': {
        'common': dict(individual=False, lr=1e-4, batch_size=32, lradj='type1'),
    },
}
X_ONLY_MODELS = ('PatchTST', 'DLinear')
TRAIN_KEYS = ('lr', 'batch_size', 'lradj')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_configs(model_name, data_name, pearl_args, overrides):
    spec = NATIVE[model_name]
    cfg = {**spec['common'], **spec.get(data_name, {}), **(overrides or {})}
    train_cfg = {k: cfg.pop(k) for k in TRAIN_KEYS}
    configs = SimpleNamespace(
        seq_len=pearl_args.seq_len, label_len=pearl_args.label_len, pred_len=pearl_args.pred_len,
        enc_in=pearl_args.enc_in, dec_in=pearl_args.enc_in, c_out=pearl_args.enc_in,
        features=pearl_args.features, embed=pearl_args.embed, freq=pearl_args.freq, **cfg)
    return configs, train_cfg


def build_model(model_name, configs):
    module = importlib.import_module(f"models.{model_name}")
    return module.Model(configs).float()


def build_loaders(pearl_args):
    _, train_loader = data_provider(pearl_args, 'train')
    val_set, _ = data_provider(pearl_args, 'val')
    test_set, _ = data_provider(pearl_args, 'test')

    def seq(ds):
        return DataLoader(ds, batch_size=256, shuffle=False,
                          num_workers=pearl_args.num_workers, drop_last=False)
    return train_loader, seq(val_set), seq(test_set)


def unpack(batch, device):
    x, y, xm, ym = batch
    return (x.float().to(device), y.float().to(device),
            xm.float().to(device), ym.float().to(device))


def forward_model(model, name, x, xm, y, ym, configs):
    P, L = configs.pred_len, configs.label_len
    f_dim = -1 if configs.features == 'MS' else 0
    if name in X_ONLY_MODELS:
        out = model(x)
    else:
        dec_inp = torch.cat([y[:, :L, :], torch.zeros_like(y[:, -P:, :])], dim=1)
        out = model(x, xm, dec_inp, ym)
        if isinstance(out, tuple):
            out = out[0]
    return out[:, -P:, f_dim:], y[:, -P:, f_dim:]


@torch.no_grad()
def predict(model, name, loader, device, configs, collect=False):
    model.eval()
    se = ae = 0.0
    n = 0
    preds, ys = [], []
    for batch in loader:
        x, y, xm, ym = unpack(batch, device)
        out, tgt = forward_model(model, name, x, xm, y, ym, configs)
        se += ((out - tgt) ** 2).double().sum().item()
        ae += (out - tgt).abs().double().sum().item()
        n += tgt.numel()
        if collect:
            preds.append(out.cpu().numpy().astype(np.float32))
            ys.append(tgt.cpu().numpy().astype(np.float32))
    arrays = (np.concatenate(preds, 0), np.concatenate(ys, 0)) if collect else None
    return se / n, ae / n, arrays


def run(model_name, pearl_args, seed, epochs, patience, gpu, ckpt_dir, out_prefix,
        overrides=None, lr=None, batch_size=None, lradj=None):
    assert model_name in NATIVE, f"Unknown model {model_name}, available options: {list(NATIVE)}"
    set_seed(seed)
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    configs, tc = build_configs(model_name, pearl_args.data, pearl_args, overrides)
    if lr is not None:
        tc['lr'] = lr
    if batch_size is not None:
        tc['batch_size'] = batch_size
    if lradj is not None:
        tc['lradj'] = lradj
    pearl_args.batch_size = tc['batch_size']
    print(f"[config] model={model_name} data={pearl_args.data} train={tc}")
    print(f"[config] model configs={vars(configs)}")

    model = build_model(model_name, configs).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    train_loader, val_loader, test_loader = build_loaders(pearl_args)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc['lr'])
    scheduler = None
    if tc['lradj'] == 'TST':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=tc['lr'], steps_per_epoch=len(train_loader),
            epochs=epochs, pct_start=0.3)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_{pearl_args.data}_pl{configs.pred_len}_s{seed}.pt")

    best_val, best_epoch, no_improve, epochs_run = float('inf'), -1, 0, 0
    t_train = time.time()
    for epoch in range(epochs):
        model.train()
        losses = []
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch + 1}")):
            x, y, xm, ym = unpack(batch, device)
            if epoch == 0 and step == 0:
                print(f"[check] x {tuple(x.shape)} | y {tuple(y.shape)} | x_mark {tuple(xm.shape)} | "
                      f"y_mark {tuple(ym.shape)} (y 第1维应为 label_len+pred_len="
                      f"{configs.label_len + configs.pred_len})")
            optimizer.zero_grad()
            out, tgt = forward_model(model, model_name, x, xm, y, ym, configs)
            loss = torch.mean((out - tgt) ** 2)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(loss.item())
        epochs_run = epoch + 1

        if tc['lradj'] == 'type1':
            for g in optimizer.param_groups:
                g['lr'] = tc['lr'] * (0.5 ** (epoch + 1))

        val_mse, val_mae, _ = predict(model, model_name, val_loader, device, configs)
        print(f"[epoch {epoch + 1}] train MSE = {np.mean(losses):.6f} | val MSE = {val_mse:.6f} | "
              f"val MAE = {val_mae:.6f} | lr = {optimizer.param_groups[0]['lr']:.2e}")
        if val_mse < best_val:
            best_val, best_epoch, no_improve = val_mse, epoch + 1, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"early stopping at epoch {epoch + 1}")
                break
    train_sec = time.time() - t_train

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, _, (vp, vt) = predict(model, model_name, val_loader, device, configs, collect=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    test_mse, test_mae, (tp, tt) = predict(model, model_name, test_loader, device, configs, collect=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    test_pass_sec = time.time() - t0

    for tag, p, t in (('val', vp, vt), ('test', tp, tt)):
        np.save(f"{out_prefix}_{tag}_pred.npy", p)
        np.save(f"{out_prefix}_{tag}_target.npy", t)

    with open(f"{out_prefix}_meta.json", 'w') as f:
        json.dump({'model': model_name, 'data': pearl_args.data, 'seed': seed,
                   'pred_len': configs.pred_len, 'n_params': n_params,
                   'best_epoch': best_epoch, 'best_val_mse': best_val, 'epochs_run': epochs_run,
                   'test_mse': test_mse, 'test_mae': test_mae,
                   'train_sec': train_sec, 'test_pass_sec': test_pass_sec,
                   'n_test_windows': int(tp.shape[0]), 'n_val_windows': int(vp.shape[0]),
                   'train_cfg': tc, 'model_cfg': {k: v for k, v in vars(configs).items()}},
                  f, indent=2, default=str)