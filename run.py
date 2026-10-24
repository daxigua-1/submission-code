import argparse
import os
import random

import numpy as np
import torch
import math
import sys

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
# from exp.exp_main import Exp_Long_Term_Forecast

# from utils.efficiency import benchmark_model_efficiency

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(6)


set_random_seed(2021)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SOFTS')
    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')
    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--d_core', type=int, default=512, help='dimension of core')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--attention_type', type=str, default="full", help='the attention type of transformer')
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--use_denoise', type=int, default=False, help='use denoise')
    parser.add_argument('--kernel', type=int, default=3, help='use norm and denorm')

    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--warmup_epochs', type=int, default=5)

    # Evidential Loss
    parser.add_argument('--evid_lam', type=float, default=0.05, help='weight for evidence regularization')
    parser.add_argument('--evid_lam_mse', type=float, default=0.05, help='weight for mse regularization')
    parser.add_argument('--evid_nll_scale', type=float, default=2.0, help='scale factor for NLL term')
    parser.add_argument('--evid_lam_is', type=float, default=0.1, help='importance scaling weight')
    parser.add_argument('--log_scale', type=float, default=math.log(math.sqrt(2)), help='initial log scale for evidential head (default: 0.347)')
    parser.add_argument('--ablation', type=str, default='none', choices=['none', 'no_uats', 'no_is', 'no_nig'])

    parser.add_argument('--head_type', type=str, default='nig', choices=['nig', 'gaussian', 'studentt', 'quantile'])
    parser.add_argument('--dof_min', type=float, default=2.1)
    parser.add_argument('--unc_weight', type=float, default=0.1)
    parser.add_argument('--detach_mu', type=int, default=1)
    parser.add_argument('--alpha_min', type=float, default=1.5)
    parser.add_argument('--nu_min', type=float, default=0.1)
    parser.add_argument('--calibrate', type=str, default='conformal', choices=['conformal', 'none'])
    parser.add_argument('--calib_mode', type=str, default='horizon', choices=['horizon', 'global'])
    parser.add_argument('--calib_max_channels', type=int, default=32)
    parser.add_argument('--controls', type=int, default=1)
    # parser.add_argument('--seeds', type=str, default='2021,2022,2023')
    parser.add_argument('--seeds', type=str, default='2021')  ##single_seed
    parser.add_argument('--no_calendar', type=int, default=0)

    parser.add_argument('--test_efficiency', action='store_true', help='Test model efficiency (Params, FLOPs, Speed)')
    parser.add_argument('--export_mu_prefix', type=str, default='')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.00024, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--cost_json', type=str, default='')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    print('Args in experiment:')
    print(args)
    Exp = Exp_Long_Term_Forecast

    # seeds = [int(s) for s in args.seeds.split(',') if s.strip()]  # multi-seeds
    seeds = [2021]  # single-seed

    keys = ['mse', 'mae', 'crps', 'qice', 'picp50', 'picp75', 'picp90', 'picp95', 'mpiw90', 'ece', 'avg_is']
    collected = {k: [] for k in keys}
    ctrl_keys = ['picp90', 'mpiw90', 'ece', 'avg_is']
    ctrl_collected = {}

    for seed in seeds:
        set_random_seed(seed)
        args.seed = seed
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_hd{}_abl{}_s{}'.format(
            args.task_name, args.model_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len, args.d_model, args.e_layers,
            args.d_layers, args.d_ff, args.factor, args.embed, args.distil, args.des,
            args.head_type, args.ablation, seed)

        print(f'\n{"=" * 70}\n seed = {seed} | {setting}\n{"=" * 70}')
        exp = Exp_Long_Term_Forecast(args)

        if args.cost_json:
            from benchmark_cost import bench_pearl
            bench_pearl(exp, args.cost_json)
            sys.exit(0)

        if args.is_training:
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mse, mae, m = exp.test(setting)

            if args.export_mu_prefix:
                current_seed = args.seed
                exp.export_mu(f"{args.export_mu_prefix}_s{current_seed}")
        else:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            mse, mae, m = exp.test(setting, test=1)

            if args.export_mu_prefix:
                current_seed = args.seed
                exp.export_mu(f"{args.export_mu_prefix}_s{current_seed}")

        collected['mse'].append(mse)
        collected['mae'].append(mae)
        if m is not None:
            collected['crps'].append(m.get('crps', float('nan')))
            collected['qice'].append(m.get('qice', float('nan')))
            for c in (50, 75, 90, 95):
                collected[f'picp{c}'].append(m['picp'][c / 100])
            collected['mpiw90'].append(m['mpiw'][0.90])
            collected['ece'].append(m['ece'])
            collected['avg_is'].append(m['avg_is'])
            for name, r_ in m.get('controls', {}).items():
                d = ctrl_collected.setdefault(name, {k: [] for k in ctrl_keys})
                d['picp90'].append(r_['picp'][0.90]); d['mpiw90'].append(r_['mpiw'][0.90])
                d['ece'].append(r_['ece']); d['avg_is'].append(r_['avg_is'])
        del exp
        torch.cuda.empty_cache()

    def fmt(v):
        return "N/A" if len(v) == 0 else f"{np.mean(v):.4f} ± {np.std(v):.4f}"

    summary = f"Seeds {seeds} | {args.data} pl{args.pred_len} | head_type={args.head_type} " \
              f"| calibrate={args.calibrate}\n" + \
              "\n".join(f"{k:8s}: {fmt(collected[k])}" for k in keys)
    for name, d in ctrl_collected.items():
        summary += f"\n-- control [{name}] (conformal, same mu) --\n" + \
                   "\n".join(f"{k:8s}: {fmt(d[k])}" for k in ctrl_keys)
    print(f"\n{'=' * 70}\n{summary}\n{'=' * 70}")
    with open("multi_seed_results.txt", "a") as f:
        f.write(f"\n{'=' * 60}\n{args.model_id}_{args.data}_pl{args.pred_len}"
                f"_hd{args.head_type}_abl{args.ablation}\n{summary}\n")