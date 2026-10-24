import argparse
import ast
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from point_baseline_on_pearl_data import run, NATIVE


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split('=', 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=list(NATIVE))
    ap.add_argument('--seed', type=int, default=2021)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--batch_size', type=int, default=None)
    ap.add_argument('--lradj', default=None, choices=[None, 'type1', 'TST', 'cosine', 'type3', 'none'])
    ap.add_argument('--override', nargs='*', default=None)

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
    ap.add_argument('--pearl_num_workers', type=int, default=0)

    ap.add_argument('--ckpt_dir', default='./checkpoints_point_baselines')
    ap.add_argument('--out_prefix', required=True)
    args = ap.parse_args()

    pearl_args = SimpleNamespace(
        root_path=args.pearl_root_path, data_path=args.pearl_data_path,
        data=args.pearl_data, features=args.pearl_features, target=args.pearl_target,
        seq_len=args.pearl_seq_len, label_len=args.pearl_label_len, pred_len=args.pearl_pred_len,
        enc_in=args.pearl_enc_in, embed=args.pearl_embed, freq=args.pearl_freq,
        batch_size=32, num_workers=args.pearl_num_workers,
        seasonal_patterns='Monthly',
    )

    run(args.model, pearl_args, seed=args.seed, epochs=args.epochs, patience=args.patience,
        gpu=args.gpu, ckpt_dir=args.ckpt_dir, out_prefix=args.out_prefix,
        overrides=parse_overrides(args.override), lr=args.lr,
        batch_size=args.batch_size, lradj=args.lradj)


if __name__ == '__main__':
    main()