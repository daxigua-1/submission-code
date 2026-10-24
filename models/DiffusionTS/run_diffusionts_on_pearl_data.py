import argparse
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diffusionts_on_pearl_data import run, load_cfg      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo_root', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--config', default=None)
    ap.add_argument('--seed', type=int, default=2021)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--out_prefix', required=True)
    ap.add_argument('--ckpt_dir', default='./checkpoints_diffusionts')

    ap.add_argument('--n_samples', type=int, default=20)
    ap.add_argument('--val_windows', type=int, default=50)
    ap.add_argument('--test_stride', type=int, default=1)
    ap.add_argument('--test_windows', type=int, default=None)
    ap.add_argument('--sample_batch', type=int, default=256)

    ap.add_argument('--margin', type=float, default=0.0)
    ap.add_argument('--eta', type=float, default=0.0)
    ap.add_argument('--sampling_steps', type=int, default=None)
    ap.add_argument('--coef', type=float, default=None)
    ap.add_argument('--step_size', type=float, default=None)
    ap.add_argument('--max_steps', type=int, default=None)
    ap.add_argument('--skip_train', action='store_true')

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
    ap.add_argument('--pearl_num_workers', type=int, default=0)
    a = ap.parse_args()

    pearl_args = SimpleNamespace(
        root_path=a.pearl_root_path, data_path=a.pearl_data_path, data=a.pearl_data,
        features=a.pearl_features, target=a.pearl_target, seq_len=a.pearl_seq_len,
        label_len=a.pearl_label_len, pred_len=a.pearl_pred_len, enc_in=a.pearl_enc_in,
        embed=a.pearl_embed, freq=a.pearl_freq, batch_size=a.pearl_batch_size,
        num_workers=a.pearl_num_workers, seasonal_patterns='Monthly')

    cfg = load_cfg(a.config, a.repo_root)
    run(pearl_args, a.repo_root, cfg, seed=a.seed, out_prefix=a.out_prefix, ckpt_dir=a.ckpt_dir,
        n_samples=a.n_samples, val_windows=a.val_windows, test_stride=a.test_stride,
        test_windows=a.test_windows, margin=a.margin, eta=a.eta, sample_batch=a.sample_batch,
        sampling_steps=a.sampling_steps, coef=a.coef, step_size=a.step_size,
        max_steps=a.max_steps, skip_train=a.skip_train, device=a.device)


if __name__ == '__main__':
    main()