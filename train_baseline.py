import numpy as np
from torch.utils.data import DataLoader
import loader2 as lo
import torch.optim as optim
from tqdm import tqdm
import os
from evaluate_baseline import Evaluate
from config import *
import torch as t
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ExponentialLR, CosineAnnealingLR
from torch import nn
# import model.model_transformer_baseline as model
# import model.model_cslstm_baseline as model
# import model.model_bat_baseline as model
import model.model_hltp_baseline as model
import faulthandler
faulthandler.enable()

def main():
    args['train_flag'] = True
    evaluate = Evaluate()
    
    # 初始化模型
    baseline_model = model.Seq2SeqBaseline(args)
    predictor = model.predictor(args)
    
    baseline_model = baseline_model.to(device)
    baseline_model.train()

    # 加载训练数据
    t1 = lo.HighSimDataset('../data/train_data.npy', args['in_length'], args['out_length'])
    trainDataloader = DataLoader(t1, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_worker'],
                                 pin_memory=True, drop_last=True)
    
    # 设置优化器和学习率调度器
    optimizer = optim.Adam(baseline_model.parameters(), lr=learning_rate)
    # scheduler = CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=len(trainDataloader) * 3,  # 每3个epoch重启一次
    #     T_mult=2  # 每次重启后周期翻倍
    # )
    scheduler = CosineAnnealingLR(optimizer, T_max=args['epoch'], last_epoch=-1)
    # scheduler = ExponentialLR(optimizer, gamma=args['gamma'])
    # 假设你的模型叫 net
    print(f"Total trainable parameters: {count_parameters(baseline_model):,}")
    # 加载检查点（如果有）
    if args['last_epoch'] != 0:
        checkpoint = t.load(args['path'] + '/baseline_epoch' + str(args['last_epoch']) + '.tar')
        baseline_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # 训练循环
    for epoch in range(args['last_epoch'], args['epoch']):
        print("epoch:", epoch + 1, 'lr', optimizer.param_groups[0]['lr'])

        # 初始化损失计数器
        loss_position = 0  # 位置预测损失
        loss_cf_type = 0  # 跟驰类型预测损失
        
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
            model_outputs = baseline_model(hist)
            # predictions = predictor.forward(model_outputs, nextv, veh_state)
            
            # 提取预测结果
            position_pred = model_outputs['position_pred']  # [B, out_length, 1]
            cf_type_pred = model_outputs['cf_type_pred']  # [B, cf_type]
            
            # 计算损失
            # 1. 位置预测损失
            position_loss = t.nn.MSELoss()(position_pred.squeeze(-1), fut[:, :, 2])
            
            # 2. 跟驰类型预测损失
            cf_type_loss = t.nn.CrossEntropyLoss()(cf_type_pred, cf_type)
            
            # 组合损失
            total_loss = position_loss + cf_type_loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            # 添加梯度裁剪
            t.nn.utils.clip_grad_norm_(baseline_model.parameters(), max_norm=2.0)
            optimizer.step()

            
            # 累积损失
            loss_position += position_loss.item()
            loss_cf_type += cf_type_loss.item()
            
            batch_count += 1
            
            # 打印进度
            if idx == int(len(trainDataloader) / 4) * model_step:
                print(f'process: {model_step / 4}, loss: {total_loss.item():.4f}')
                model_step += 1
        scheduler.step()
        # 计算平均损失
        avg_losses = {
            'position': loss_position / batch_count,
            'cf_type': loss_cf_type / batch_count,
        }
        
        # 打印每个epoch的平均损失
        print(f"Epoch {epoch+1} Average Losses:")
        for loss_name, loss_value in avg_losses.items():
            print(f"{loss_name}: {loss_value:.4f}")
        
        # 保存检查点
        t.save({
            'epoch': epoch,
            'model_state_dict': baseline_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': total_loss,
        }, args['path'] + '/baseline_epoch' + str(epoch + 1) + '.tar')
        
        # 每个epoch进行一次评估
        if (epoch + 1) % 1 == 0:
            evaluate.main(name=str(epoch + 1), val=False)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == '__main__':
    main()