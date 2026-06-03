import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
import loader2 as lo
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import os
from evaluate_new import Evaluate
from config import *
from DWA import *
import torch as t
import requests
import csv
import faulthandler
faulthandler.enable()
import model.model_MoE_gru_new_ablation as model
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR, CosineAnnealingLR
from torch import nn


def param_distribution_loss(acc_params, idm_params, cf_probs):
    """添加参数分布损失，使用更灵活的方式鼓励参数在合理范围内"""
    batch_size = cf_probs.shape[0]
    total_loss = 0.0
    
    # 定义参数范围
    param_ranges = {
        # ACC参数范围
        'acc': [
            (0.01, 2.0),   # kp: 比例增益
            (0.01, 2.0),   # kd: 微分增益
            (5.0, 40.0),   # v_des: 期望速度
            (0.5, 10.0),   # h0: 最小安全间距
            (0.0, 5.0),    # h1: 速度系数
            (0.01, 0.5)    # kv: 速度控制增益
        ],
        # IDM参数范围
        'idm': [
            (0.5, 10.0),   # s0: 最小安全间距
            (0.2, 5.0),    # T: 安全时距
            (0.2, 4.0),    # b: 舒适减速度
            (0.2, 5.0),    # a_max: 最大加速度
            (5.0, 50.0),   # v_max: 期望速度
            (2.0, 6.0)     # delta: 加速度指数
        ]
    }
    
    # 处理ACC参数
    for i in range(2):
        # 提取ACC参数
        params = acc_params[i].squeeze(1)  # [B, 6]
        acc_loss = 0
        
        # 计算每个参数的惩罚
        for j in range(6):
            param_value = params[:, j]
            min_val, max_val = param_ranges['acc'][j]
            # 对期望速度参数进行特殊处理
            scale = 10.0 if j == 2 else 1.0
            penalty = t.mean(t.relu(min_val - param_value) + t.relu(param_value - max_val)) / scale
            acc_loss += penalty
        
        # 加权累加损失
        acc_weight = cf_probs[:, i].mean()
        total_loss += acc_weight * acc_loss
    
    # 处理IDM参数
    for i in range(2):
        # 提取IDM参数
        params = idm_params[i].squeeze(1)  # [B, 6]
        idm_loss = 0
        
        # 计算每个参数的惩罚
        for j in range(6):
            param_value = params[:, j]
            min_val, max_val = param_ranges['idm'][j]
            # 对期望速度参数进行特殊处理
            scale = 10.0 if j == 4 else 1.0
            penalty = t.mean(t.relu(min_val - param_value) + t.relu(param_value - max_val)) / scale
            idm_loss += penalty
        
        # 加权累加损失
        idm_weight = cf_probs[:, i+2].mean()
        total_loss += idm_weight * idm_loss
    
    return total_loss


def main():
    args['train_flag'] = True
    evaluate = Evaluate()
    dwa = DWA(args)
    
    # 初始化模型
    Encoder = model.Encoder(args)
    sim = model.predictor(args)
    
    # 手动初始化参数预测头的偏置
    with t.no_grad():
        # 初始化所有专家的偏置
        for i, expert in enumerate(Encoder.parameter_head.all_experts):
            # last_layer = expert.layers[-1]
            # if isinstance(last_layer, nn.Linear):
            #     if i < 2:  # ACC专家
            #         # ACC参数: kp, kd, v_des, h0, h1, kv
            #         last_layer.bias.data[0] = 0.23  # kp
            #         last_layer.bias.data[1] = 0.07  # kd
            #         last_layer.bias.data[2] = 25.0  # v_des
            #         last_layer.bias.data[3] = 2.0   # h0
            #         last_layer.bias.data[4] = 1.6   # h1
            #         last_layer.bias.data[5] = 0.2   # kv
            #     else:  # IDM专家
            #         # IDM参数: s0, T, b, a_max, v_max, delta
            #         last_layer.bias.data[0] = 2.0   # s0
            #         last_layer.bias.data[1] = 1.6   # T
            #         last_layer.bias.data[2] = 1.67  # b
            #         last_layer.bias.data[3] = 1.5   # a_max
            #         last_layer.bias.data[4] = 25.0  # v_max
            #         last_layer.bias.data[5] = 4.0   # delta
            # MLP对象中找到最后一个Linear层
            if hasattr(expert, 'layers'):
                # 对于MLP对象，访问其layers属性
                for layer in reversed(expert.layers):
                    if isinstance(layer, nn.Linear):
                        last_layer = layer
                        break
            else:
                # 对于Sequential对象
                for layer in reversed(expert):
                    if isinstance(layer, nn.Linear):
                        last_layer = layer
                        break

            if i < 2:  # ACC专家
                # ACC参数: kp, kd, v_des, h0, h1, kv
                last_layer.bias.data[0] = 0.23  # kp
                last_layer.bias.data[1] = 0.07  # kd
                last_layer.bias.data[2] = 25.0  # v_des
                last_layer.bias.data[3] = 2.0   # h0
                last_layer.bias.data[4] = 1.6   # h1
                last_layer.bias.data[5] = 0.2   # kv
            else:  # IDM专家
                # IDM参数: s0, T, b, a_max, v_max, delta
                last_layer.bias.data[0] = 2.0   # s0
                last_layer.bias.data[1] = 1.6   # T
                last_layer.bias.data[2] = 1.67  # b
                last_layer.bias.data[3] = 1.5   # a_max
                last_layer.bias.data[4] = 25.0  # v_max
                last_layer.bias.data[5] = 4.0   # delta
    
    Encoder = Encoder.to(device)
    Encoder.train()

    # 加载训练数据
    t1 = lo.HighSimDataset('../data/train_data.npy', args['in_length'], args['out_length'])
    trainDataloader = DataLoader(t1, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_worker'],
                                 pin_memory=True, drop_last=True)
    
    # 设置优化器和学习率调度器
    optimizer_e = optim.Adam(Encoder.parameters(), lr=learning_rate)
    # scheduler_e = CosineAnnealingWarmRestarts(
    #     optimizer_e,
    #     T_0=len(trainDataloader) * 3,  # 每3个epoch重启一次
    #     T_mult=2  # 每次重启后周期翻倍
    # )
    scheduler_e = CosineAnnealingLR(optimizer_e, T_max=args['epoch'], last_epoch=-1)
    print(f"Total trainable parameters: {count_parameters(Encoder):,}")
    # 加载检查点（如果有）
    if args['last_epoch'] != 0:
        checkpoint = t.load(args['path'] + '/epoch' + str(args['last_epoch']) + '_e.tar')
        Encoder.load_state_dict(checkpoint['model_state_dict'])
        optimizer_e.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_e.load_state_dict(checkpoint['scheduler_state_dict'])


    # 训练循环
    for epoch in range(args['last_epoch'], args['epoch']):
        print("epoch:", epoch + 1, 'lr', optimizer_e.param_groups[0]['lr'])

        # 初始化损失计数器
        loss_direct_pos = 0  # 直接位置预测损失
        loss_dynamic_gap = 0  # 动力学模型间隙预测损失
        loss_dynamic_vel = 0  # 动力学模型速度预测损失
        loss_cf_type = 0  # 跟驰类型预测损失
        loss_param_dist = 0  # 参数分布损失
        
        model_step = 1
        batch_count = 0

        for idx, data in enumerate(tqdm(trainDataloader)):
            hist, nextv, fut, cf_type = data
            hist = hist.to(device)
            fut = fut.to(device)
            veh_state = hist[:, -1, [4, 2]]  # 间隙、速度
            nextv = nextv.to(device)
            cf_type = cf_type.to(device)
            
            # 前向传播
            model_outputs = Encoder(hist)
            predictions = sim.forward(model_outputs, nextv, veh_state)
            
            # 提取预测结果
            direct_pred = predictions['direct_pred']  # [B, out_length, 1]
            dynamic_pred = predictions['dynamic_pred']  # [B, out_length, 2]
            cf_type_pred = model_outputs['cf_type_pred']  # [B, cf_type]
            
            # 提取参数
            params_dict = model_outputs['params']
            acc_params = params_dict['acc_params']  # 两个ACC专家的参数
            idm_params = params_dict['idm_params']  # 两个IDM专家的参数
            cf_probs = params_dict['cf_probs']  # [B, cf_type]
            
            # 计算损失
            # 1. 直接位置预测损失
            direct_pos_loss = t.nn.MSELoss()(direct_pred.squeeze(-1), fut[:, :, 2])
            
            # 2. 动力学模型预测损失
            dynamic_gap_loss = t.nn.MSELoss()(dynamic_pred[:, :, 0], fut[:, :, 0])  # 间隙预测
            dynamic_vel_loss = t.nn.MSELoss()(dynamic_pred[:, :, 1], fut[:, :, 1])  # 速度预测
            
            # 3. 跟驰类型预测损失
            cf_type_loss = t.nn.CrossEntropyLoss()(cf_type_pred, cf_type)
            
            # 4. 参数分布损失
            param_dist_loss = param_distribution_loss(acc_params, idm_params, cf_probs)
            
            # 组合损失
            task_losses = {
                'direct_pos': direct_pos_loss,
                'dynamic_gap': dynamic_gap_loss,
                'dynamic_vel': dynamic_vel_loss,
                'cf_type': cf_type_loss,
                'param_dist': 0.01 * param_dist_loss
            }

            # 计算总损失
            total_loss = sum(task_losses.values())
            # 添加nan检查
            if t.isnan(total_loss).any():
                print("NaN")
            # 反向传播和优化
            optimizer_e.zero_grad()
            total_loss.backward()
            # 添加梯度裁剪
            t.nn.utils.clip_grad_norm_(Encoder.parameters(), max_norm=2.0)
            optimizer_e.step()

            
            # 累积损失
            loss_direct_pos += direct_pos_loss.item()
            loss_dynamic_gap += dynamic_gap_loss.item()
            loss_dynamic_vel += dynamic_vel_loss.item()
            loss_cf_type += cf_type_loss.item()
            loss_param_dist += param_dist_loss.item()
            
            batch_count += 1
            
            # 打印进度
            if idx == int(len(trainDataloader) / 4) * model_step:
                print(f'process: {model_step / 4}, loss: {total_loss.item():.4f}')
                model_step += 1
        scheduler_e.step()
        # 计算平均损失
        avg_losses = {
            'direct_pos': loss_direct_pos / batch_count,
            'dynamic_gap': loss_dynamic_gap / batch_count,
            'dynamic_vel': loss_dynamic_vel / batch_count,
            'cf_type': loss_cf_type / batch_count,
            'param_dist': loss_param_dist / batch_count
        }
        
        # 打印每个epoch的平均损失
        print(f"Epoch {epoch+1} Average Losses:")
        for loss_name, loss_value in avg_losses.items():
            print(f"{loss_name}: {loss_value:.4f}")
        
        # 保存检查点
        t.save({
            'epoch': epoch,
            'model_state_dict': Encoder.state_dict(),
            'optimizer_state_dict': optimizer_e.state_dict(),
            'scheduler_state_dict': scheduler_e.state_dict(),
            'loss': total_loss,
        }, args['path'] + '/epoch' + str(epoch + 1) + '_e.tar')
        
        # 每5个epoch进行一次评估
        if (epoch + 1) % 1 == 0:
            evaluate.main(name=str(epoch + 1), val=False)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    main()
