import argparse
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nsdiff_on_pearl_data import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_type', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--horizon', type=int, default=1)
    ap.add_argument('--windows', type=int, default=96)
    ap.add_argument('--rolling_length', type=int, default=24)
    ap.add_argument('--seed', type=int, default=2021)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=10)

    ap.add_argument('--n_samples', type=int, default=20)
    ap.add_argument('--val_n_samples', type=int, default=None)
    ap.add_argument('--select_by', default='mse', choices=['mse', 'crps'])

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

    ap.add_argument('--nsdiff_root', default=None)
    ap.add_argument('--ckpt_dir', default='./checkpoints_nsdiff_pearl')
    ap.add_argument('--out_prefix', required=True)
    args = ap.parse_args()

    nsdiff_kwargs = dict(
        dataset_type=args.dataset_type, device=args.device, horizon=args.horizon,
        windows=args.windows, pred_len=args.pearl_pred_len,
        rolling_length=args.rolling_length, load_pretrain=False,
        epochs=args.epochs, patience=args.patience,
    )

    pearl_args = SimpleNamespace(
        root_path=args.pearl_root_path, data_path=args.pearl_data_path,
        data=args.pearl_data, features=args.pearl_features, target=args.pearl_target,
        seq_len=args.pearl_seq_len, label_len=args.pearl_label_len, pred_len=args.pearl_pred_len,
        enc_in=args.pearl_enc_in, embed=args.pearl_embed, freq=args.pearl_freq,
        batch_size=args.pearl_batch_size, num_workers=args.pearl_num_workers,
        seasonal_patterns='Monthly',
    )

    run(nsdiff_kwargs, pearl_args, seed=args.seed, epochs=args.epochs, patience=args.patience,
        ckpt_dir=args.ckpt_dir, out_prefix=args.out_prefix, n_samples=args.n_samples,
        val_n_samples=args.val_n_samples, select_by=args.select_by,
        nsdiff_root=args.nsdiff_root)


if __name__ == '__main__':
    main()