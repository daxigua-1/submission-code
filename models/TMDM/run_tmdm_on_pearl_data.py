"""
启动脚本：TMDM 自己的参数和 PEARL data_provider 的参数分开传（后者带 pearl_ 前缀）。

推荐布局：TMDM 仓库（含 model9_NS_transformer/）放在 PEARL 项目的 models/TMDM/ 下，
本脚本和 tmdm_on_pearl_data.py 与 model9_NS_transformer/ 同级；在 PEARL 项目根目录
运行（这样 data_provider、utils 直接可用，不需要复制）：

    export PYTHONPATH=./
    python models/TMDM/run_tmdm_on_pearl_data.py \\
        --device=cuda:0 --seed=2021 --epochs=50 --patience=10 \\
        --n_z_samples=100 --test_batch_size=8 \\
        --pearl_root_path=./dataset/ETT-small/ --pearl_data_path=ETTh1.csv \\
        --pearl_data=ETTh1 --pearl_features=M --pearl_seq_len=96 \\
        --pearl_label_len=48 --pearl_pred_len=192 --pearl_enc_in=7 \\
        --pearl_embed=timeF --pearl_freq=h --pearl_batch_size=32 \\
        --out_prefix=./exported_samples/tmdm_ETTh1_pl192_s2021

注意：
  * out_prefix 请以 _s{seed} 结尾，和 aggregate_baseline_results.py 的约定一致。
  * diffusion_config_dir 默认指向 TMDM 仓库自带的 toy_8gauss.yml —— 已确认
    TMDM 没有为 ETT 数据集单独建立配置，所有数据集公用这一份
    diffusion/testing/optim 超参数，这是原仓库自己的设计，不是遗漏。
  * ETTm1/ETTm2 需要 --pearl_freq=t。
  * --n_z_samples 请和 NsDiff 那边用的 --n_samples 保持同一个值，两个基线的
    采样条数不一致会让 CRPS/QICE 的估计精度不可比。
"""
import argparse
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tmdm_on_pearl_data import build_tmdm_args, run


def main():
    ap = argparse.ArgumentParser()
    # ---- TMDM 自己的参数 ----
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--seed', type=int, default=2021)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--test_batch_size', type=int, default=8,
                    help='导出采样时的 batch size，和训练用的 batch_size 分开，'
                         '扩散采样一次要处理 batch_size*n_z_samples 条序列，显存吃得多')
    ap.add_argument('--n_z_samples', type=int, default=100,
                    help='每个窗口的采样条数 S，需要和其它基线（如 NsDiff 的 --n_samples）保持一致')
    ap.add_argument('--diffusion_config_dir', default='./model9_NS_transformer/configs/toy_8gauss.yml')
    ap.add_argument('--k_z', type=float, default=1e-2)
    ap.add_argument('--k_cond', type=float, default=1.0)
    ap.add_argument('--learning_rate', type=float, default=1e-4)

    # ---- PEARL data_provider 需要的参数 ----
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

    ap.add_argument('--ckpt_dir', default='./checkpoints_tmdm_pearl')
    ap.add_argument('--out_prefix', required=True)
    args = ap.parse_args()

    tmdm_args = build_tmdm_args(
        device=args.device, gpu=args.gpu, seed=args.seed, train_epochs=args.epochs,
        patience=args.patience, test_batch_size=args.test_batch_size,
        diffusion_config_dir=args.diffusion_config_dir,
        k_z=args.k_z, k_cond=args.k_cond, learning_rate=args.learning_rate,
        seq_len=args.pearl_seq_len, label_len=args.pearl_label_len, pred_len=args.pearl_pred_len,
        enc_in=args.pearl_enc_in, dec_in=args.pearl_enc_in, c_out=args.pearl_enc_in,
        features=args.pearl_features, freq=args.pearl_freq, embed=args.pearl_embed,
    )

    pearl_args = SimpleNamespace(
        root_path=args.pearl_root_path, data_path=args.pearl_data_path,
        data=args.pearl_data, features=args.pearl_features, target=args.pearl_target,
        seq_len=args.pearl_seq_len, label_len=args.pearl_label_len, pred_len=args.pearl_pred_len,
        enc_in=args.pearl_enc_in, embed=args.pearl_embed, freq=args.pearl_freq,
        batch_size=args.pearl_batch_size, num_workers=args.pearl_num_workers,
        seasonal_patterns='Monthly',
    )

    run(tmdm_args, pearl_args, seed=args.seed, epochs=args.epochs, patience=args.patience,
       ckpt_dir=args.ckpt_dir, out_prefix=args.out_prefix,
       test_batch_size=args.test_batch_size, n_z_samples=args.n_z_samples)


if __name__ == '__main__':
    main()