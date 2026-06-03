from __future__ import print_function
import numpy as np

import loader2 as lo
from torch.utils.data import DataLoader, SubsetRandomSampler
import pandas as pd
from config import *
import model.model_MoE_gru as model
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
loss_path = args['l_path'] + 'test_loss—test.csv'

if not os.path.isfile(loss_path):
    with open(loss_path, "w") as file:
        file.write("Header1\n")

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=3, verbose=False, delta=0,  trace_func=print):
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
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

class Evaluate():

    def __init__(self):
        self.op = 0
        self.drawImg = False
        self.scale = 0.3048
        self.prop = 1

    def dxMSETest(self, y_pred, y_gt, mask, dim):
        acc = t.zeros_like(mask)
        dx_pred = y_pred
        dx = y_gt
        out = t.pow(dx_pred - dx, 2)
        acc = out
        acc = acc * mask
        lossVal = t.pow(t.sum(acc, dim=0), 0.5)
        counts = t.pow(t.sum(mask, dim=0), 0.5)

        return lossVal, counts
        
    def MAPETest(self, y_pred, y_gt, mask, dim):
        epsilon = 1e-8  # 避免分母为零

        # 计算绝对百分比误差
        acc = t.abs(y_pred - y_gt) / (t.abs(y_gt) + epsilon) * 100

        # 将无效值的误差设置为 0，并将 NaN 替换为 0
        acc = acc * mask
        acc[acc != acc] = 0  # 将 NaN 替换为 0

        # 计算 MAPE
        lossVal = t.sum(acc, dim=0)

        # 计算有效值的数量
        counts = t.sum(mask, dim=0)

        return lossVal, counts

    def MAETest(self, y_pred, y_gt, mask, dim):
        # 计算绝对误差
        acc = t.abs(y_pred - y_gt)

        # 将无效值的误差设置为 0，并将 NaN 替换为 0
        acc = acc * mask
        acc[acc != acc] = 0  # 将 NaN 替换为 0

        # 计算 MAE
        lossVal = t.sum(acc, dim=0)

        # 计算有效值的数量
        counts = t.sum(mask, dim=0)

        return lossVal, counts


    def main(self, name, val):
        model_step = 1
        early_stopping = EarlyStopping(patience=4)
        args['train_flag'] = False

        c_path = args['path']

        # 初始化模型
        Encoder = model.Encoder(args)
        Koop = model.predictor(args)
        
        # 加载模型权重
        checkpoint = t.load(c_path + '/epoch' + name + '_e.tar')
        Encoder.load_state_dict(checkpoint['model_state_dict'])
    
        Encoder = Encoder.to(device)
        Encoder.eval()
        
        # 加载测试数据
        t2 = lo.HighSimDataset('../data/test_data.npy', args['in_length'], args['out_length'])
        valDataloader = DataLoader(t2, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_worker'],
                                   pin_memory=True, drop_last=True)

        # 初始化评估指标
        lossVals = t.zeros(args['out_length'], 2).to(device)  # 只评估间隙和速度
        counts = t.zeros(args['out_length'], 2).to(device)
        lossVals_P = t.zeros(args['out_length'], 2).to(device)
        counts_P = t.zeros(args['out_length'], 2).to(device)
        # 添加轨迹预测评估指标
        lossVals_traj = t.zeros(args['out_length'], 1).to(device)  # 轨迹预测RMSE
        counts_traj = t.zeros(args['out_length'], 1).to(device)

        all_time = 0
        
        # 初始化跟驰类型分类评估指标
        total_tp = t.zeros(args['cf_type']).to(device)
        total_fp = t.zeros(args['cf_type']).to(device)
        total_fn = t.zeros(args['cf_type']).to(device)
        
        val_batch_count = len(valDataloader)
        print("begin.................................", name)
        
        with t.no_grad():
            for idx, data in enumerate(tqdm(valDataloader)):
                hist, nextv, fut, cf_type = data
                hist = hist.to(device)
                fut = fut.to(device)
                veh_state = hist[:, -1, [4, 2]]  # 间隙、速度
                nextv = nextv.to(device)
                cf_type = cf_type.to(device)
                
                te = time.time()
                
                # 前向传播
                model_outputs = Encoder(hist)
                predictions = Koop.forward(model_outputs, nextv, veh_state)
                
                all_time += time.time() - te
                
                # 提取预测结果
                dynamic_pred = predictions['dynamic_pred']  # [B, out_length, 2]
                direct_pred = predictions['direct_pred']  # [B, out_length, 1] - 轨迹预测
                cf_type_pred = model_outputs['cf_type_pred']  # [B, cf_type]

                # 创建掩码
                mask = t.zeros_like(fut[:, :, :2])
                mask[:, :args['out_length'], :] = 1
                mask = mask.to(device)

                # 创建轨迹预测掩码
                traj_mask = t.zeros_like(fut[:, :, 2:3])
                traj_mask[:, :args['out_length'], :] = 1
                traj_mask = traj_mask.to(device)

                # 计算MSE和MAPE
                l, c = self.dxMSETest(dynamic_pred, fut[:, :, :2], mask, 2)
                lp, cp = self.MAPETest(dynamic_pred, fut[:, :, :2], mask, 2)

                # 计算轨迹预测RMSE
                l_traj, c_traj = self.dxMSETest(direct_pred, fut[:, :, 2:3], traj_mask, 1)

                lossVals += l.detach()
                counts += c.detach()
                lossVals_P += lp.detach()
                counts_P += cp.detach()
                lossVals_traj += l_traj.detach()
                counts_traj += c_traj.detach()

                # 计算跟驰类型分类指标
                _, predicted = t.max(cf_type_pred, dim=1)
                _, target = t.max(cf_type, dim=1)

                # 累积TP, FP, FN
                for i in range(args['cf_type']):
                    tp = ((predicted == i) & (target == i)).sum().float()
                    fp = ((predicted == i) & (target != i)).sum().float()
                    fn = ((predicted != i) & (target == i)).sum().float()

                    total_tp[i] += tp.detach()
                    total_fp[i] += fp.detach()
                    total_fn[i] += fn.detach()

                if idx == int(val_batch_count / 4) * model_step:
                    print('process:', model_step / 4)
                    model_step += 1

            # 打印评估结果
            print("MSE结果:")
            print(lossVals / counts)
            print("MAPE结果:")
            print(lossVals_P / counts_P)
            print("轨迹预测RMSE结果:")
            print(lossVals_traj / counts_traj)

            # 计算精确度、召回率和F1分数
            precision = total_tp / (total_tp + total_fp + 1e-6)
            recall = total_tp / (total_tp + total_fn + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

            for i in range(args['cf_type']):
                print(f"类别 {i} - 精确率: {precision[i]:.4f}, 召回率: {recall[i]:.4f}, F1: {f1[i]:.4f}")

            # 保存评估结果到CSV
            tf = pd.read_csv(loss_path)

            # 合并MSE和MAPE结果
            dx_loss = t.cat((lossVals / counts, lossVals_P / counts_P), dim=-1)
            loss_means = t.mean(dx_loss, dim=0, keepdim=True)
            dx_loss = t.cat((dx_loss, loss_means), dim=0)
            dx_loss = dx_loss.tolist()
            dx_loss = [["dx", "v", "dxp", "vp"]] + dx_loss + [["", "", "", ""]]
            ds_flattened = [row[0] for row in dx_loss] + [row[1] for row in dx_loss] + [row[2] for row in dx_loss]+[row[3] for row in dx_loss]

            # 轨迹预测RMSE结果
            traj_loss = lossVals_traj / counts_traj
            traj_loss_mean = t.mean(traj_loss, dim=0, keepdim=True)
            traj_loss = t.cat((traj_loss, traj_loss_mean), dim=0)
            traj_loss = traj_loss.tolist()
            traj_loss = [["traj"]] + traj_loss + [["", "", "", ""]]
            traj_flattened = [row[0] for row in traj_loss]

            # 合并分类指标
            class_loss = t.cat((precision.unsqueeze(1), recall.unsqueeze(1), f1.unsqueeze(1)), dim=-1)
            class_loss_means = t.mean(class_loss, dim=0, keepdim=True)
            class_loss = t.cat((class_loss, class_loss_means), dim=0)
            class_loss = class_loss.tolist()
            class_loss = [["precision", "recall", "f1"]] + class_loss + [["", "", "", ""]]
            class_flattened = [row[0] for row in class_loss] + [row[1] for row in class_loss] + [row[2] for row in class_loss]

            # 合并所有结果
            flattened = ds_flattened + traj_flattened + class_flattened
            test_loss_dx = pd.DataFrame({name + 'loss': flattened})

            tf_result = pd.concat([tf, test_loss_dx], axis=1)
            tf_result.to_csv(loss_path, index=False)
            
            print(f"平均推理时间: {all_time / len(valDataloader):.4f}秒/批次")


if __name__ == '__main__':
    names = '1'
    evaluate = Evaluate()
    evaluate.main(name=names, val=False)
    # for epoch in range(5, 20):
    #     evaluate.main(name=str(epoch + 1), val=False)
    # os.system("shutdown")
