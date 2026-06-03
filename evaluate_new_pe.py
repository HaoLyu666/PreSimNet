from __future__ import print_function
import numpy as np

import loader2 as lo
from torch.utils.data import DataLoader, SubsetRandomSampler
import pandas as pd
from config import *
import model.model_MoE_gru_new_ablation as model
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import torch as t

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
loss_path = args['l_path'] + 'test_loss_test.csv'

if not os.path.isfile(loss_path):
    with open(loss_path, "w") as file:
        file.write("Header1\n")


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=3, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class Evaluate:
    def __init__(self):
        self.evaluate = True

    def main(self, name, val=True):
        Encoder = model.Encoder(args)
        sim = model.predictor(args)
        Encoder = Encoder.to(device)
        sim = sim

        # 加载模型权重
        checkpoint = t.load(args['path'] + '/epoch' + str(name) + '_e.tar')
        Encoder.load_state_dict(checkpoint['model_state_dict'])
        Encoder.eval()

        # 加载测试数据
        if val:
            t1 = lo.HighSimDataset('../data/val_data.npy', args['in_length'], args['out_length'])
        else:
            t1 = lo.HighSimDataset('../data/test_data.npy', args['in_length'], args['out_length'])

        testDataloader = DataLoader(t1, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_worker'],
                                    pin_memory=True, drop_last=True)

        lossVals_traj = t.zeros(args['out_length']).to(device)
        counts_traj = t.zeros(args['out_length']).to(device)

        # 初始化仿真评估指标
        lossVals_sim_gap = t.zeros(args['out_length']).to(device)
        lossVals_sim_vel = t.zeros(args['out_length']).to(device)
        counts_sim = t.zeros(args['out_length']).to(device)

        # 移除分类评估相关的初始化代码
        # correct = t.zeros(5).to(device)
        # total = t.zeros(5).to(device)
        # tp = t.zeros(5).to(device)
        # fp = t.zeros(5).to(device)
        # fn = t.zeros(5).to(device)

        with t.no_grad():
            for idx, data in enumerate(tqdm(testDataloader)):
                hist, nextv, fut, cf_type = data
                hist = hist.to(device)
                fut = fut.to(device)
                veh_state = hist[:, -1, [4, 2]]  # 间隙、速度
                nextv = nextv.to(device)
                # cf_type = cf_type.to(device)  # 消融：不再使用

                # 前向传播
                model_outputs = Encoder(hist)
                predictions = sim.forward(model_outputs, nextv, veh_state)

                # 提取预测结果
                direct_pred = predictions['direct_pred'].squeeze(-1)  # [B, out_length]
                dynamic_pred = predictions['dynamic_pred']  # [B, out_length, 2]

                # 1. 评估轨迹预测 (MSE)
                batch_size = direct_pred.shape[0]
                loss = (direct_pred - fut[:, :, 2]) ** 2
                lossVals_traj += loss.sum(dim=0)
                counts_traj += batch_size

                # 2. 评估仿真效果 (RMSE for Gap and Velocity)
                # 间隙误差
                gap_loss = (dynamic_pred[:, :, 0] - fut[:, :, 0]) ** 2
                lossVals_sim_gap += gap_loss.sum(dim=0)

                # 速度误差
                vel_loss = (dynamic_pred[:, :, 1] - fut[:, :, 1]) ** 2
                lossVals_sim_vel += vel_loss.sum(dim=0)

                counts_sim += batch_size

                # 3. 移除分类评估逻辑
                # cf_type_pred = model_outputs['cf_type_pred']
                # _, predicted = t.max(cf_type_pred, 1)
                # ...

        # 计算平均指标
        # 轨迹预测RMSE
        traj_rmse = t.sqrt(lossVals_traj / counts_traj)

        # 仿真RMSE
        sim_gap_rmse = t.sqrt(lossVals_sim_gap / counts_sim)
        sim_vel_rmse = t.sqrt(lossVals_sim_vel / counts_sim)

        print(f"Evaluation Results (Epoch {name}):")
        print("-" * 50)

        # 打印轨迹预测结果
        print("Trajectory Prediction RMSE (m):")
        print(f"0.5s: {traj_rmse[4]:.4f}")
        print(f"1.0s: {traj_rmse[9]:.4f}")
        print(f"1.5s: {traj_rmse[14]:.4f}")
        print(f"2.0s: {traj_rmse[19]:.4f}")
        print(f"Avg:  {traj_rmse.mean():.4f}")
        print("-" * 50)

        # 打印仿真结果
        print("Simulation Gap RMSE (m):")
        print(f"0.5s: {sim_gap_rmse[4]:.4f}")
        print(f"1.0s: {sim_gap_rmse[9]:.4f}")
        print(f"1.5s: {sim_gap_rmse[14]:.4f}")
        print(f"2.0s: {sim_gap_rmse[19]:.4f}")
        print(f"Avg:  {sim_gap_rmse.mean():.4f}")

        print("\nSimulation Velocity RMSE (m/s):")
        print(f"0.5s: {sim_vel_rmse[4]:.4f}")
        print(f"1.0s: {sim_vel_rmse[9]:.4f}")
        print(f"1.5s: {sim_vel_rmse[14]:.4f}")
        print(f"2.0s: {sim_vel_rmse[19]:.4f}")
        print(f"Avg:  {sim_vel_rmse.mean():.4f}")
        print("-" * 50)

        # 保存结果到CSV
        if not val:
            tf = pd.DataFrame([name], columns=['name'])

            # 轨迹预测结果
            traj_res = traj_rmse.tolist()
            traj_res.append(traj_rmse.mean().item())
            traj_df = pd.DataFrame([traj_res], columns=[f'traj_{i}' for i in range(20)] + ['traj_avg'])

            # 仿真结果
            sim_gap_res = sim_gap_rmse.tolist()
            sim_gap_res.append(sim_gap_rmse.mean().item())
            sim_gap_df = pd.DataFrame([sim_gap_res], columns=[f'sim_gap_{i}' for i in range(20)] + ['sim_gap_avg'])

            sim_vel_res = sim_vel_rmse.tolist()
            sim_vel_res.append(sim_vel_rmse.mean().item())
            sim_vel_df = pd.DataFrame([sim_vel_res], columns=[f'sim_vel_{i}' for i in range(20)] + ['sim_vel_avg'])

            # 合并所有结果 (移除分类结果)
            result_df = pd.concat([tf, traj_df, sim_gap_df, sim_vel_df], axis=1)

            # 追加到CSV文件
            if not os.path.isfile(loss_path) or os.stat(loss_path).st_size == 0:
                result_df.to_csv(loss_path, index=False)
            else:
                result_df.to_csv(loss_path, mode='a', header=False, index=False)