"""
把 TMDM（NS-Transformer 均值估计 + 条件扩散）自己的训练/采样逻辑，原样照抄自
model9_NS_transformer/exp/exp_main.py 的 Exp_Main.train()/vali()/test()，接到
PEARL 自己的 data_provider 上。

和 NsDiff 不同的地方：TMDM 的 data_provider
（model9_NS_transformer/data_provider/data_loader.py）本身就和 PEARL 的
data_provider 几乎一致——同一个只在训练集拟合的 StandardScaler、同一种逐步
滑动窗口（__len__ = len(data_x)-seq_len-pred_len+1）、batch_y 同样是
label_len+pred_len 长度。这意味着这次不需要像 NsDiff 那样写一个批次格式转换
层，PEARL 的 data_provider 产出的 batch 可以直接喂给下面这几个方法。

扩散超参数（timesteps/beta_schedule/n_z_samples 等）仍然来自
model9_NS_transformer/configs/toy_8gauss.yml —— 确认过 TMDM 仓库没有为 ETT
数据集单独建立配置文件，所有数据集公用这一份 diffusion/testing/optim 超参数，
这是原仓库自己的设计（命令行示例里也没有切换它），不是我们这边的疏漏。

已知并绕开的两个原始代码里的问题：
  1. exp_main.py 的 test() 里 store_gen_y_at_step_t 用配置里固定的
     test_batch_size 做 reshape；如果数据集窗口数不能整除 test_batch_size，
     最后一个 batch 会 reshape 错位。这里的 _sample_batch 改用每个 batch
     的真实 batch size（B = batch_x.size(0)）做 reshape，并且验证/测试的
     导出 DataLoader 都用 drop_last=True 规避这个问题（代价是丢弃数据集末
     尾不满一个 batch 的少量窗口，比冒着数据错位的风险要稳妥）。
  2. n_z_samples_depart>1 时，原代码的内层循环结果会被外层覆盖丢弃（和
     NsDiff 里同一个模式的 bug）。toy_8gauss.yml 里这个值本来就是 1，这里
     额外加一个显式检查/强制，不依赖配置文件恰好写对。

vali() 的损失口径和 train() 不同（vali 只在 pred_len 那一段上算噪声预测
MSE，train 在整个 label_len+pred_len 上算），这是原作者的设计，不是不一致
的 bug，这里原样保留这个差异。
"""
import math
import os
import types

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model9_NS_transformer.ns_models import ns_Transformer
from model9_NS_transformer.diffusion_models import diffuMTS
from model9_NS_transformer.diffusion_models.diffusion_utils import q_sample, p_sample_loop

from data_provider.data_factory import data_provider


# --------------------------------------------------------------------------- args
def build_tmdm_args(**overrides):
    """和 runner9_NS_transformer.py 里 argparse 的默认值保持一致；数据集相关
    的字段（seq_len/label_len/pred_len/enc_in/...）通过 overrides 传入覆盖。"""
    args = types.SimpleNamespace(
        is_training=True, model_id='tmdm_on_pearl', model='NS_Transformer',
        data='custom', root_path='./dataset/ETT-small/', data_path='ETTh1.csv',
        features='M', target='OT', freq='h', checkpoints='./checkpoints/',
        seq_len=96, label_len=48, pred_len=192,
        enc_in=7, dec_in=7, c_out=7, d_model=512, n_heads=8, e_layers=2, d_layers=1,
        d_ff=2048, moving_avg=25, factor=3, distil=True, dropout=0.05,
        embed='timeF', activation='gelu', output_attention=False, do_predict=False,
        k_z=1e-2, k_cond=1, d_z=8,
        num_workers=0, itr=1, train_epochs=50, batch_size=32, test_batch_size=8,
        patience=10, learning_rate=0.0001, learning_rate_Cond=0.0001,
        des='Exp', loss='mse', lradj='type1', use_amp=False,
        use_gpu=True, gpu=0, use_multi_gpu=False, devices='0', seed=2021,
        p_hidden_dims=[64, 64], p_hidden_layers=2,
        diffusion_config_dir='./model9_NS_transformer/configs/toy_8gauss.yml',
        cond_pred_model_pertrain_dir='./checkpoints/cond_pred_model_pertrain_NS_Transformer/checkpoint.pth',
        CART_input_x_embed_dim=32, mse_timestep=0, MLP_diffusion_net=False, timesteps=1000,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def log_normal(x, mu, var):
    """照抄 exp_main.py 里的同名函数，不做任何改动。"""
    eps = 1e-8
    var = var + eps
    return 0.5 * torch.mean(np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)


# --------------------------------------------------------------------------- 数据
def build_pearl_loaders(pearl_args, test_batch_size):
    """train_loader：PEARL 自带的 shuffle loader，供训练用。
    val_loader_epoch：和 train 同样的 batch_size，供每个 epoch 算验证损失用
      （这一步只是一次普通前向，不涉及扩散采样，用小 batch 没必要）。
    val_loader_export / test_loader_export：用 test_batch_size、
      shuffle=False、drop_last=True，专门给最后导出采样轨迹用（扩散采样一次
      要处理 batch_size*n_z_samples 条序列，显存吃得多，所以单独给一个更小
      的 batch size）。"""
    _, train_loader = data_provider(pearl_args, 'train')
    val_set, _ = data_provider(pearl_args, 'val')
    test_set, _ = data_provider(pearl_args, 'test')

    def seq(ds, bs):
        return DataLoader(ds, batch_size=bs, shuffle=False,
                          num_workers=pearl_args.num_workers, drop_last=True)

    val_loader_epoch = seq(val_set, pearl_args.batch_size)
    val_loader_export = seq(val_set, test_batch_size)
    test_loader_export = seq(test_set, test_batch_size)
    return train_loader, val_loader_epoch, val_loader_export, test_loader_export


def unpack(batch, device):
    x, y, xm, ym = batch
    return (x.float().to(device), y.float().to(device),
            xm.float().to(device), ym.float().to(device))


def make_dec_inp(batch_y, label_len, pred_len, device):
    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
    return torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)


# --------------------------------------------------------------------------- 主体
class TMDMOnPearlData:
    def __init__(self, args, n_z_samples=None):
        self.args = args
        self.device = torch.device(
            f'cuda:{args.gpu}' if (args.use_gpu and torch.cuda.is_available()) else 'cpu')

        # === 修复 1：动态拦截并修复 DataEmbedding 参数过多问题 (Monkey Patch) ===
        if hasattr(diffuMTS, 'DataEmbedding'):
            OriginalInit = diffuMTS.DataEmbedding.__init__
            if not getattr(OriginalInit, '_is_patched', False):
                def patched_init(self_obj, *p_args, **p_kwargs):
                    c_in = p_args[0]
                    d_model = p_args[1]
                    dropout = p_args[4] if len(p_args) >= 5 else 0.1
                    OriginalInit(self_obj, c_in, d_model, dropout)
                patched_init._is_patched = True
                diffuMTS.DataEmbedding.__init__ = patched_init
        # ========================================================================

        self.model = diffuMTS.Model(args, self.device).float().to(self.device)
        self.cond_pred_model = ns_Transformer.Model(args).float().to(self.device)
        self._check_n_z_samples_depart()
        if n_z_samples is not None:
            old = self.model.diffusion_config.testing.n_z_samples
            self.model.diffusion_config.testing.n_z_samples = int(n_z_samples)
            print(f"[info] n_z_samples: 配置文件里是 {old}，覆盖为 {n_z_samples}")

    def _check_n_z_samples_depart(self):
        depart = int(self.model.diffusion_config.testing.n_z_samples_depart)
        if depart != 1:
            print(f"[warn] n_z_samples_depart={depart} -> 强制为 1（避免内层采样结果被覆盖丢弃）")
            self.model.diffusion_config.testing.n_z_samples_depart = 1

    # ------------------------------------------------------------ 单个 batch 的损失
    def _train_step(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """原样照抄 exp_main.py Exp_Main.train() 里 else 分支的计算。"""
        args = self.args
        dec_inp = make_dec_inp(batch_y, args.label_len, args.pred_len, self.device)

        n = batch_x.size(0)
        t = torch.randint(low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

        _, y_0_hat_batch, KL_loss, _ = self.cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        loss_vae = log_normal(batch_y, y_0_hat_batch, torch.from_numpy(np.array(1)))
        loss_vae_all = loss_vae + args.k_z * KL_loss

        y_T_mean = y_0_hat_batch
        e = torch.randn_like(batch_y).to(self.device)
        y_t_batch = q_sample(batch_y, y_T_mean, self.model.alphas_bar_sqrt,
                             self.model.one_minus_alphas_bar_sqrt, t, noise=e)
        output = self.model(batch_x, batch_x_mark, batch_y, y_t_batch, y_0_hat_batch, t)

        return (e - output).square().mean() + args.k_cond * loss_vae_all

    def _vali_step(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """照抄 exp_main.py Exp_Main.vali()：噪声预测的 MSE 只算 pred_len
        那一段，这是原作者的设计（和 train() 不一样），原样保留这个差异。"""
        args = self.args
        dec_inp = make_dec_inp(batch_y, args.label_len, args.pred_len, self.device)

        n = batch_x.size(0)
        t = torch.randint(low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n]

        _, y_0_hat_batch, KL_loss, _ = self.cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        loss_vae = log_normal(batch_y, y_0_hat_batch, torch.from_numpy(np.array(1)))
        loss_vae_all = loss_vae + args.k_z * KL_loss

        y_T_mean = y_0_hat_batch
        e = torch.randn_like(batch_y).to(self.device)
        y_t_batch = q_sample(batch_y, y_T_mean, self.model.alphas_bar_sqrt,
                             self.model.one_minus_alphas_bar_sqrt, t, noise=e)
        output = self.model(batch_x, batch_x_mark, batch_y, y_t_batch, y_0_hat_batch, t)

        return ((e[:, -args.pred_len:, :] - output[:, -args.pred_len:, :]).square().mean()
               + args.k_cond * loss_vae_all)

    @torch.no_grad()
    def _sample_batch(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        """照抄 exp_main.py Exp_Main.test() 里的采样逻辑，返回
        samples: [B,S,P,N]（cpu）, truths: [B,P,N]（cpu）。
        用真实 batch size 做 reshape，不依赖配置里的 test_batch_size（见文件头说明第 1 点）。"""
        args = self.args
        dec_inp = make_dec_inp(batch_y, args.label_len, args.pred_len, self.device)
        B = batch_x.size(0)

        _, y_0_hat_batch, _, _ = self.cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        cfg = self.model.diffusion_config.testing
        S = int(cfg.n_z_samples)
        depart = int(cfg.n_z_samples_depart)        # 已在 __init__ 里强制为 1
        repeat_n = S // depart

        def tile(t):
            return t.repeat(repeat_n, 1, 1, 1).transpose(0, 1).flatten(0, 1).to(self.device)

        y_0_hat_tile = tile(y_0_hat_batch)
        y_T_mean_tile = y_0_hat_tile
        x_tile = tile(batch_x)
        x_mark_tile = tile(batch_x_mark)

        y_tile_seq = p_sample_loop(self.model, x_tile, x_mark_tile, y_0_hat_tile, y_T_mean_tile,
                                   self.model.num_timesteps, self.model.alphas,
                                   self.model.one_minus_alphas_bar_sqrt)
        gen_y = y_tile_seq[self.model.num_timesteps]            # [B*repeat_n, label_len+pred_len, N]
        gen_y = gen_y.reshape(B, repeat_n, args.label_len + args.pred_len, args.c_out)

        f_dim = -1 if args.features == 'MS' else 0
        samples = gen_y[:, :, -args.pred_len:, f_dim:].detach().cpu().float()   # [B,S,P,N]
        truths = batch_y[:, -args.pred_len:, f_dim:].detach().cpu().float()      # [B,P,N]
        if samples.size(1) != S:
            raise RuntimeError(f"实际采样条数={samples.size(1)}，期望 S={S}，检查 n_z_samples 配置是否生效")
        return samples, truths


# --------------------------------------------------------------------------- 训练 + 导出
def run(tmdm_args, pearl_args, seed, epochs, patience, ckpt_dir, out_prefix,
       test_batch_size=8, n_z_samples=100):
    torch.manual_seed(seed)
    np.random.seed(seed)

    runner = TMDMOnPearlData(tmdm_args, n_z_samples=n_z_samples)
    device = runner.device
    optimizer = optim.Adam(
        [{'params': runner.model.parameters()}, {'params': runner.cond_pred_model.parameters()}],
        lr=tmdm_args.learning_rate)

    train_loader, val_loader_epoch, val_loader_export, test_loader_export = \
        build_pearl_loaders(pearl_args, test_batch_size)

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)) or '.', exist_ok=True)
    # === 修复 2：防止多卡并发运行时，不同数据集的最佳权重互相覆盖 ===
    ckpt_path = os.path.join(ckpt_dir, f"tmdm_pearl_{pearl_args.data}_seed{seed}.pt")
    best_val, no_improve = float('inf'), 0

    for epoch in range(epochs):
        runner.model.train(); runner.cond_pred_model.train()
        train_losses = []
        for step, batch in enumerate(tqdm(train_loader, desc=f"epoch {epoch + 1}")):
            batch_x, batch_y, batch_x_mark, batch_y_mark = unpack(batch, device)
            if epoch == 0 and step == 0:
                print(f"[check] x {tuple(batch_x.shape)} | y {tuple(batch_y.shape)} | "
                     f"x_mark {tuple(batch_x_mark.shape)} | y_mark {tuple(batch_y_mark.shape)}  "
                     f"(y/y_mark 第1维应等于 label_len+pred_len="
                     f"{tmdm_args.label_len + tmdm_args.pred_len})")
            optimizer.zero_grad()
            loss = runner._train_step(batch_x, batch_y, batch_x_mark, batch_y_mark)
            if torch.isnan(loss):
                print("[warn] loss 变成 NaN，跳过这个 batch")
                continue
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        runner.model.eval(); runner.cond_pred_model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader_epoch:
                batch_x, batch_y, batch_x_mark, batch_y_mark = unpack(batch, device)
                val_losses.append(runner._vali_step(batch_x, batch_y, batch_x_mark, batch_y_mark).item())
        val_loss = float(np.mean(val_losses))
        print(f"[epoch {epoch + 1}] train loss = {np.mean(train_losses):.6f} | val loss = {val_loss:.6f}")

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            torch.save({'model': runner.model.state_dict(),
                       'cond_pred_model': runner.cond_pred_model.state_dict()}, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"early stopping at epoch {epoch + 1}")
                break

    ckpt = torch.load(ckpt_path, map_location=device)
    runner.model.load_state_dict(ckpt['model'])
    runner.cond_pred_model.load_state_dict(ckpt['cond_pred_model'])

    @torch.no_grad()
    def export(loader, tag):
        runner.model.eval(); runner.cond_pred_model.eval()
        all_s, all_t = [], []
        for batch in tqdm(loader, desc=f"export {tag}"):
            batch_x, batch_y, batch_x_mark, batch_y_mark = unpack(batch, device)
            s, t = runner._sample_batch(batch_x, batch_y, batch_x_mark, batch_y_mark)
            all_s.append(s.numpy()); all_t.append(t.numpy())
        s = np.concatenate(all_s, axis=0)
        t = np.concatenate(all_t, axis=0)
        np.save(f"{out_prefix}_{tag}_samples.npy", s)
        np.save(f"{out_prefix}_{tag}_target.npy", t)
        print(f"{tag}: samples {s.shape}, target {t.shape}")

    export(val_loader_export, 'val')
    export(test_loader_export, 'test')