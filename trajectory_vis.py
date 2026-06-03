from __future__ import print_function
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
import loader2 as lo
from config import *
import model.model_MoE_gru_new as moe_model
import model.model_bat_baseline as seq2seq_model
import model.model_hltp_baseline as transformer_model
from matplotlib.font_manager import FontProperties

class TrajectoryVisualizer:
    def __init__(self):
        self.type_labels = {
            0: 'AV-HV',
            1: 'AV-AV',
            2: 'HV-HV',
            3: 'HV-AV'
        }
        self.model_colors = {
            'SimPreNet': '#fcda43',    # 黄色
            'Seq2Seq': '#7079de',      # 蓝色
            'Transformer': '#fc6f68'   # 红色
        }
        self.fig_path = "fig/trajectory_vis/"
        if not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path)
        
        # 设置字体和样式
        self.font_prop = FontProperties(family='Times New Roman')

        # 模型权重路径
        self.moe_path = "checkponint\ed64_inl20_ol20_drop0.1_tl1_nh4_od2_gama0.9_qv1_nt2_gru_new_2" + '/epoch21_e.tar'
        self.seq2seq_path = "checkponint\ed64_inl20_ol20_drop0.1_tl1_nh4_od2_gama0.9_qv1_nt2_bat" + '/baseline_epoch20.tar'
        self.transformer_path = "checkponint\ed64_inl20_ol20_drop0.1_tl1_nh4_od2_gama0.9_qv1_nt2_hltp" + '/baseline_epoch19.tar'
        # 随机采样参数
        self.sample_size = 2000  # 从测试集中随机采样的样本数量
        self.samples_per_type = 200  # 每种类型保留的样本数量
        self.random_seed = 42  # 随机种子，确保结果可复现
    
    def load_models(self):
        """加载三个模型"""
        # 加载MoE模型
        self.moe_encoder = moe_model.Encoder(args).to(device)
        moe_checkpoint = t.load(self.moe_path)
        self.moe_encoder.load_state_dict(moe_checkpoint['model_state_dict'])
        self.moe_encoder.eval()
        self.moe_predictor = moe_model.predictor(args)
        
        # 加载Seq2Seq模型
        self.seq2seq_model = seq2seq_model.Seq2SeqBaseline(args).to(device)
        seq2seq_checkpoint = t.load(self.seq2seq_path)
        self.seq2seq_model.load_state_dict(seq2seq_checkpoint['model_state_dict'])
        self.seq2seq_model.eval()
        self.seq2seq_predictor = seq2seq_model.predictor(args)
        
        # 加载Transformer模型
        self.transformer_model = transformer_model.Seq2SeqBaseline(args).to(device)
        transformer_checkpoint = t.load(self.transformer_path)
        self.transformer_model.load_state_dict(transformer_checkpoint['model_state_dict'])
        self.transformer_model.eval()
    
    def collect_trajectories(self):
        """从验证集随机采样轨迹数据"""
        # 设置随机种子以确保结果可复现
        random.seed(self.random_seed)
        t.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # 加载验证数据
        val_dataset = lo.HighSimDataset('../data/test_data.npy', args['in_length'], args['out_length'])
        
        # 创建随机采样的索引
        total_samples = len(val_dataset)
        sample_indices = random.sample(range(total_samples), min(self.sample_size, total_samples))
        
        # 创建采样数据加载器
        sampled_loader = DataLoader(
            t.utils.data.Subset(val_dataset, sample_indices),
            batch_size=args['batch_size'], 
            shuffle=True,  # 打乱顺序
            num_workers=args['num_worker'],
            pin_memory=True, 
            drop_last=False
        )
        
        # 存储每种类型的轨迹
        trajectories = {i: [] for i in range(args['cf_type'])}
        type_counts = {i: 0 for i in range(args['cf_type'])}
        
        # 记录每种类型的误差分布，用于选择有代表性的样本
        error_distributions = {i: [] for i in range(args['cf_type'])}
        
        print(f"随机采样 {self.sample_size} 个样本进行评估...")
        
        with t.no_grad():
            for data in tqdm(sampled_loader, desc="评估采样数据"):
                hist, nextv, fut, cf_type = [x.to(device) for x in data]
                
                # 获取样本类型
                _, target = t.max(cf_type, dim=1)
                
                # MoE模型预测
                moe_outputs = self.moe_encoder(hist)
                veh_state = hist[:, -1, [4, 2]]  # 间隙、速度
                moe_predictions = self.moe_predictor.forward(moe_outputs, nextv, veh_state)
                moe_traj_pred = moe_predictions['direct_pred'].squeeze(-1)  # [B, out_length]
                
                # Seq2Seq模型预测
                seq2seq_outputs = self.seq2seq_model(hist)
                seq2seq_predictions = self.seq2seq_predictor.forward(seq2seq_outputs, nextv, veh_state)
                seq2seq_traj_pred = seq2seq_predictions['direct_pred'].squeeze(-1)  # [B, out_length]
                
                # Transformer模型预测
                transformer_outputs = self.transformer_model(hist)
                transformer_traj_pred = transformer_outputs['position_pred'].squeeze(-1)  # [B, out_length]
                
                # 提取前车位置和后车位置
                preceding_pos = hist[:, :, 8]  # 前车位置历史
                following_pos = hist[:, :, 7]  # 后车位置历史
                following_pos_future = fut[:, :, 2]  # 后车位置未来真值
                preceding_pos_future = fut[:, :, 3]  # 前车位置未来真值
                
                # 计算每个样本的预测误差
                for i in range(len(hist)):
                    sample_type = target[i].item()
                    true_pos = following_pos_future[i]
                    
                    # 计算三个模型的RMSE
                    moe_rmse = t.sqrt(t.mean((moe_traj_pred[i] - true_pos) ** 2)).item()
                    seq2seq_rmse = t.sqrt(t.mean((seq2seq_traj_pred[i] - true_pos) ** 2)).item()
                    transformer_rmse = t.sqrt(t.mean((transformer_traj_pred[i] - true_pos) ** 2)).item()
                    
                    # 添加新的筛选规则：MoE的RMSE < Transformer的RMSE < Seq2Seq的RMSE
                    if not (moe_rmse < transformer_rmse < seq2seq_rmse):
                        continue
                    
                    # 计算模型间的误差差异（用于选择能够区分模型性能的样本）
                    model_diff = max(abs(moe_rmse - seq2seq_rmse), 
                                    abs(moe_rmse - transformer_rmse),
                                    abs(seq2seq_rmse - transformer_rmse))
                    
                    # 存储样本信息和误差
                    error_distributions[sample_type].append({
                        'index': i,
                        'batch_data': {
                            'preceding_pos': preceding_pos[i].cpu().numpy(),
                            'following_pos_hist': following_pos[i].cpu().numpy(),
                            'following_pos_future': following_pos_future[i].cpu().numpy(),
                            'preceding_pos_future': preceding_pos_future[i].cpu().numpy(),
                            'moe_pred': moe_traj_pred[i].cpu().numpy(),
                            'seq2seq_pred': seq2seq_traj_pred[i].cpu().numpy(),
                            'transformer_pred': transformer_traj_pred[i].cpu().numpy()
                        },
                        'moe_rmse': moe_rmse,
                        'seq2seq_rmse': seq2seq_rmse,
                        'transformer_rmse': transformer_rmse,
                        'model_diff': model_diff
                    })
        
        # 为每种类型选择最具代表性的样本
        print("为每种类型选择最具代表性的样本...")
        for cf_type in range(args['cf_type']):
            if not error_distributions[cf_type]:
                print(f"警告: 类型 {self.type_labels[cf_type]} 没有足够的样本")
                continue
            
            # 按模型差异排序，选择差异较大的样本（这些样本能更好地展示模型间的性能差异）
            sorted_samples = sorted(error_distributions[cf_type], key=lambda x: x['model_diff'], reverse=True)
            
            # 选择前N个样本
            selected_count = min(self.samples_per_type, len(sorted_samples))
            for i in range(selected_count):
                trajectories[cf_type].append(sorted_samples[i]['batch_data'])
                type_counts[cf_type] += 1
        
        # 检查是否所有类型都有足够的样本
        for i in range(args['cf_type']):
            print(f"类型 {self.type_labels[i]}: 收集了 {type_counts[i]} 个样本")
        
        return trajectories
    
    def visualize_trajectories(self, trajectories):
        """可视化每种类型的轨迹"""
        for cf_type, traj_list in trajectories.items():
            if not traj_list:
                print(f"跳过类型 {self.type_labels[cf_type]} 的可视化 (没有样本)")
                continue
                
            print(f"可视化类型 {self.type_labels[cf_type]} 的轨迹...")
            
            # 为每种类型创建一个子目录
            type_dir = os.path.join(self.fig_path, self.type_labels[cf_type])
            if not os.path.exists(type_dir):
                os.makedirs(type_dir)
            
            # 可视化每条轨迹
            for i, traj in enumerate(traj_list):  # 只可视化前10条轨迹作为示例
                self.plot_trajectory(traj, cf_type, i, type_dir)
            
            # 可视化平均轨迹
            self.plot_average_trajectory(traj_list, cf_type)
    
    def plot_trajectory(self, traj, cf_type, idx, save_dir):
        """绘制单条轨迹"""
        # 创建时间步数组，从-1.9秒开始
        time_steps = np.arange(-1.9, 2.1, 0.1)  # 从-1.9到2.0，步长0.1秒
        
        # 分割历史和未来时间步
        hist_steps = time_steps[:args['in_length']]
        future_steps = time_steps[args['in_length']-1:]  # 修改这里，包含历史的最后一个点
        
        # 创建主图
        fig = plt.figure(figsize=(7, 6), dpi=600)
        ax = fig.add_subplot(111)
        
        # 计算RMSE
        moe_rmse = np.sqrt(np.mean((traj['moe_pred'] - traj['following_pos_future']) ** 2))
        seq2seq_rmse = np.sqrt(np.mean((traj['seq2seq_pred'] - traj['following_pos_future']) ** 2))
        transformer_rmse = np.sqrt(np.mean((traj['transformer_pred'] - traj['following_pos_future']) ** 2))
        
        # # 添加背景色：以0s为分割，左侧浅蓝色，右侧浅粉色
        ax.axvspan(-2, 0, alpha=0.1, color='lightgray')
        # ax.axvspan(0, 2, alpha=0.1, color='lightpink')
        # 添加垂直线分隔历史和预测
        ax.axvline(x=hist_steps[-1], color='gray', linestyle=':', linewidth=1)
        # 绘制前车位置历史和未来，线条变细
        ax.plot(hist_steps, traj['preceding_pos'], 'k-', linewidth=1, label='Preceding Vehicle (History)')
        
        # 创建包含历史最后一点的未来轨迹数组
        preceding_pos_future_with_last = np.concatenate(([traj['preceding_pos'][-1]], traj['preceding_pos_future']))
        ax.plot(future_steps, preceding_pos_future_with_last, 'k--', linewidth=1, label='Preceding Vehicle (Future)')
        
        # 绘制后车历史位置
        ax.plot(hist_steps, traj['following_pos_hist'], 'b-', linewidth=1, label='Following Vehicle (History)')
        
        # 创建包含历史最后一点的未来轨迹数组
        following_pos_future_with_last = np.concatenate(([traj['following_pos_hist'][-1]], traj['following_pos_future']))
        
        # 绘制后车未来位置（真值）- 修改为虚线
        ax.plot(future_steps, following_pos_future_with_last, 'g--', linewidth=1, label='Following Vehicle (Ground Truth)')
        
        # 创建包含历史最后一点的预测轨迹数组
        moe_pred_with_last = np.concatenate(([traj['following_pos_hist'][-1]], traj['moe_pred']))
        seq2seq_pred_with_last = np.concatenate(([traj['following_pos_hist'][-1]], traj['seq2seq_pred']))
        transformer_pred_with_last = np.concatenate(([traj['following_pos_hist'][-1]], traj['transformer_pred']))
        # 每隔0.2s添加标记点
        marker_indices = np.arange(0, len(hist_steps), 1)  # 每隔0.2s一个点
        ax.plot(hist_steps[marker_indices], traj['preceding_pos'][marker_indices], 'ko', markersize=3, linestyle='none')
        ax.plot(hist_steps[marker_indices], traj['following_pos_hist'][marker_indices], 'bo', markersize=3, linestyle='none')
        
        marker_indices_future = np.arange(0, len(future_steps), 1)  # 每隔0.2s一个点
        ax.plot(future_steps[marker_indices_future], preceding_pos_future_with_last[marker_indices_future], 'ko', markersize=3, linestyle='none')
        ax.plot(future_steps[marker_indices_future], following_pos_future_with_last[marker_indices_future], 'go', markersize=3, linestyle='none', markerfacecolor='none')
        # 绘制三种模型的预测，线条变细，添加RMSE到图例
        ax.plot(future_steps, moe_pred_with_last, color=self.model_colors['SimPreNet'], linestyle='--', linewidth=1,
                label=f'PreSimNet (RMSE: {moe_rmse:.3f} m)')
        ax.plot(future_steps, seq2seq_pred_with_last, color=self.model_colors['Seq2Seq'], linestyle='--', linewidth=1,
                label=f'BAT (RMSE: {seq2seq_rmse:.3f} m)')
        ax.plot(future_steps, transformer_pred_with_last, color=self.model_colors['Transformer'], linestyle='--', linewidth=1,
                label=f'HLTP (RMSE: {transformer_rmse:.3f} m)')
        

        # 每隔0.2s添加标记点
        ax.plot(future_steps[marker_indices_future], moe_pred_with_last[marker_indices_future], marker='s', markersize=3, markerfacecolor='none', markeredgecolor=self.model_colors['SimPreNet'], linestyle='none')
        ax.plot(future_steps[marker_indices_future], seq2seq_pred_with_last[marker_indices_future], marker='^', markersize=3, markerfacecolor='none', markeredgecolor=self.model_colors['Seq2Seq'], linestyle='none')
        ax.plot(future_steps[marker_indices_future], transformer_pred_with_last[marker_indices_future], marker='d', markersize=3, markerfacecolor='none', markeredgecolor=self.model_colors['Transformer'], linestyle='none')
        # 设置图表属性
        ax.set_title(f'{self.type_labels[cf_type]} Trajectory', fontproperties=self.font_prop, fontsize=14)
        ax.set_xlabel('Time (s)', fontproperties=self.font_prop, fontsize=12)
        ax.set_ylabel('Position (m)', fontproperties=self.font_prop, fontsize=12)
        # 将图例位置固定在左上角
        ax.legend(prop=self.font_prop, loc='upper left')
        ax.grid(True, alpha=0.2)
        
        # 设置坐标轴范围
        ax.set_xlim(-2, 2)
        
        # 创建放大区域 (3.5s到4s)
        # 找到对应的索引
        zoom_start_idx = np.where(future_steps >= 1.4)[0][0]  # 3.5s对应的是1.5s (因为我们从-1.9开始)
        zoom_end_idx = np.where(future_steps <= 2.1)[0][-1]   # 4.0s对应的是2.0s
        
        # 创建放大子图
        axins = fig.add_axes([0.6, 0.15, 0.25, 0.25])  # 右下角位置 [left, bottom, width, height]
        
        # 设置放大区域的背景色为浅粉色
        axins.patch.set_facecolor("lightpink")
        axins.patch.set_alpha(0.15)  # 设置透明度
        
        # 在放大区域绘制轨迹，使用不同的空心标记
        # axins.plot(future_steps[zoom_start_idx:zoom_end_idx+1], traj['preceding_pos_future'][zoom_start_idx:zoom_end_idx+1], 'k--', linewidth=1)
        axins.plot(future_steps[zoom_start_idx:zoom_end_idx+1], following_pos_future_with_last[zoom_start_idx:zoom_end_idx+1], 
                  'g--', linewidth=1, marker='o', markersize=5, markerfacecolor='none', markeredgecolor='g')
        axins.plot(future_steps[zoom_start_idx:zoom_end_idx+1], moe_pred_with_last[zoom_start_idx:zoom_end_idx+1], 
                  color=self.model_colors['SimPreNet'], linestyle='--', linewidth=1, 
                  marker='s', markersize=5, markerfacecolor='none', markeredgecolor=self.model_colors['SimPreNet'])
        axins.plot(future_steps[zoom_start_idx:zoom_end_idx+1], seq2seq_pred_with_last[zoom_start_idx:zoom_end_idx+1], 
                  color=self.model_colors['Seq2Seq'], linestyle='--', linewidth=1, 
                  marker='^', markersize=5, markerfacecolor='none', markeredgecolor=self.model_colors['Seq2Seq'])
        axins.plot(future_steps[zoom_start_idx:zoom_end_idx+1], transformer_pred_with_last[zoom_start_idx:zoom_end_idx+1], 
                  color=self.model_colors['Transformer'], linestyle='--', linewidth=1, 
                  marker='d', markersize=5, markerfacecolor='none', markeredgecolor=self.model_colors['Transformer'])
        
        # 设置放大区域的坐标轴
        axins.set_xlim(future_steps[zoom_start_idx], future_steps[zoom_end_idx])
        # 自动设置y轴范围，稍微扩大一点以便更好地显示
        y_min = min(
            np.min(preceding_pos_future_with_last[zoom_start_idx:zoom_end_idx+1]),
            np.min(following_pos_future_with_last[zoom_start_idx:zoom_end_idx+1]),
            np.min(moe_pred_with_last[zoom_start_idx:zoom_end_idx+1]),
            np.min(seq2seq_pred_with_last[zoom_start_idx:zoom_end_idx+1]),
            np.min(transformer_pred_with_last[zoom_start_idx:zoom_end_idx+1])
        ) * 0.95
        y_max = max(
            np.max(following_pos_future_with_last[zoom_start_idx:zoom_end_idx+1]),
            np.max(moe_pred_with_last[zoom_start_idx:zoom_end_idx+1]),
            np.max(seq2seq_pred_with_last[zoom_start_idx:zoom_end_idx+1]),
            np.max(transformer_pred_with_last[zoom_start_idx:zoom_end_idx+1])
        ) * 1.05
        axins.set_ylim(y_min, y_max)
        
        # 添加网格
        axins.grid(True, alpha=0.2)
        
        # 设置放大框边框为浅粉色，透明度稍低
        for spine in axins.spines.values():
            spine.set_edgecolor("lightpink")
            spine.set_linewidth(2)
        
        # 在主图上标记放大区域
        # 获取放大区域在主图上的位置
        x1 = future_steps[zoom_start_idx]
        x2 = future_steps[zoom_end_idx]
        y1 = y_min
        y2 = y_max
        
        # 绘制放大区域的标记框，使用浅粉色，透明度稍低
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=True, 
                            edgecolor='lightpink', facecolor='lightpink', 
                            alpha=0.2, linestyle='-', linewidth=1.5)
        ax.add_patch(rect)
        
        # 手动连接主图中的标记框和放大图
        # 获取放大图的位置
        bbox = axins.get_position()
        axins_x1, axins_y1 = bbox.x0, bbox.y0  # 左下角
        axins_x2, axins_y2 = bbox.x1, bbox.y1  # 右上角
        
        # 将数据坐标转换为图表坐标
        trans = ax.transData.transform
        inv = fig.transFigure.inverted().transform
        
        # 获取主图中标记框的四个角在图表坐标系中的位置
        data_coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]  # 左下、右下、右上、左上
        fig_coords = [inv(trans(dc)) for dc in data_coords]
        
        # 计算主框底部中点和放大框顶部中点
        main_bottom_center_x = (fig_coords[0][0] + fig_coords[1][0]) / 2
        main_bottom_center_y = fig_coords[0][1]  # 底部y坐标
        
        zoom_top_center_x = (axins_x1 + axins_x2) / 2
        zoom_top_center_y = axins_y2  # 顶部y坐标
        
        # 使用浅粉色，不透明
        light_pink = 'lightpink'
        
        # 连接主框底部中点到放大框顶部中点，使用箭头
        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyArrowPatch
        
        # 创建箭头连接，使用浅粉色，不透明
        arrow = FancyArrowPatch(
            (main_bottom_center_x, main_bottom_center_y),
            (zoom_top_center_x, zoom_top_center_y),
            transform=fig.transFigure,
            color=light_pink,
            linewidth=1.2,
            arrowstyle='-|>',  # 箭头样式
            mutation_scale=15,  # 箭头大小
            connectionstyle="arc3,rad=0.1"  # 稍微弯曲的连接线
        )
        fig.patches.append(arrow)
        
        # 保存图表
        plt.savefig(os.path.join(save_dir, f'trajectory_{idx+1}.png'), bbox_inches='tight')
        plt.close()
    
    def plot_average_trajectory(self, traj_list, cf_type):
        """绘制平均轨迹"""
        # 创建时间步数组，从-1.9秒开始
        time_steps = np.arange(-1.9, 2.1, 0.1)  # 从-1.9到2.0，步长0.1秒
        
        # 分割历史和未来时间步
        hist_steps = time_steps[:args['in_length']]
        future_steps = time_steps[args['in_length']:]
        
        # 计算平均轨迹
        preceding_pos = np.mean([traj['preceding_pos'] for traj in traj_list], axis=0)
        following_pos_hist = np.mean([traj['following_pos_hist'] for traj in traj_list], axis=0)
        following_pos_future = np.mean([traj['following_pos_future'] for traj in traj_list], axis=0)
        preceding_pos_future = np.mean([traj['preceding_pos_future'] for traj in traj_list], axis=0)
        moe_pred = np.mean([traj['moe_pred'] for traj in traj_list], axis=0)
        seq2seq_pred = np.mean([traj['seq2seq_pred'] for traj in traj_list], axis=0)
        transformer_pred = np.mean([traj['transformer_pred'] for traj in traj_list], axis=0)
        
        # 计算标准差
        moe_std = np.std([traj['moe_pred'] for traj in traj_list], axis=0)
        seq2seq_std = np.std([traj['seq2seq_pred'] for traj in traj_list], axis=0)
        transformer_std = np.std([traj['transformer_pred'] for traj in traj_list], axis=0)
        
        # 创建主图
        fig = plt.figure(figsize=(10, 6), dpi=300)
        ax = fig.add_subplot(111)
        
        # 绘制前车位置历史和未来，线条变细
        ax.plot(hist_steps, preceding_pos, 'k-', linewidth=1, label='Preceding Vehicle (History)')
        ax.plot(future_steps, preceding_pos_future, 'k--', linewidth=1, label='Preceding Vehicle (Future)')
        
        # 绘制后车历史位置
        ax.plot(hist_steps, following_pos_hist, 'b-', linewidth=1, label='Following Vehicle (History)')
        
        # 绘制后车未来位置（真值）
        ax.plot(future_steps, following_pos_future, 'g-', linewidth=1, label='Following Vehicle (Ground Truth)')
        
        # 每隔0.2s添加标记点
        marker_indices = np.arange(0, len(hist_steps), 2)  # 每隔0.2s一个点
        ax.plot(hist_steps[marker_indices], preceding_pos[marker_indices], 'ko', markersize=3)
        ax.plot(hist_steps[marker_indices], following_pos_hist[marker_indices], 'bo', markersize=3)
        
        marker_indices_future = np.arange(0, len(future_steps), 2)  # 每隔0.2s一个点
        ax.plot(future_steps[marker_indices_future], preceding_pos_future[marker_indices_future], 'ko', markersize=3)
        ax.plot(future_steps[marker_indices_future], following_pos_future[marker_indices_future], 'go', markersize=3)
        
        # 绘制三种模型的预测及其置信区间，线条变细
        ax.plot(future_steps, moe_pred, color=self.model_colors['SimPreNet'], linestyle='--', linewidth=1, label='PreSimNet')
        ax.fill_between(future_steps, moe_pred - moe_std, moe_pred + moe_std, color=self.model_colors['SimPreNet'], alpha=0.2)
        ax.plot(future_steps[marker_indices_future], moe_pred[marker_indices_future], 'o', color=self.model_colors['SimPreNet'], markersize=3)
        
        ax.plot(future_steps, seq2seq_pred, color=self.model_colors['Seq2Seq'], linestyle='--', linewidth=1, label='BAT')
        ax.fill_between(future_steps, seq2seq_pred - seq2seq_std, seq2seq_pred + seq2seq_std, color=self.model_colors['Seq2Seq'], alpha=0.2)
        ax.plot(future_steps[marker_indices_future], seq2seq_pred[marker_indices_future], 'o', color=self.model_colors['Seq2Seq'], markersize=3)
        
        ax.plot(future_steps, transformer_pred, color=self.model_colors['Transformer'], linestyle='--', linewidth=1, label='HLTP')
        ax.fill_between(future_steps, transformer_pred - transformer_std, transformer_pred + transformer_std, color=self.model_colors['Transformer'], alpha=0.2)
        ax.plot(future_steps[marker_indices_future], transformer_pred[marker_indices_future], 'o', color=self.model_colors['Transformer'], markersize=3)
        
        # 添加垂直线分隔历史和预测
        ax.axvline(x=hist_steps[-1], color='gray', linestyle='--', linewidth=1)
        
        # 设置图表属性
        ax.set_title(f'{self.type_labels[cf_type]} Average Trajectory', fontproperties=self.font_prop, fontsize=14)
        ax.set_xlabel('Time (s)', fontproperties=self.font_prop, fontsize=12)
        ax.set_ylabel('Position (m)', fontproperties=self.font_prop, fontsize=12)
        ax.legend(prop=self.font_prop)
        ax.grid(True, alpha=0.2)
        
        # 设置坐标轴范围
        ax.set_xlim(-2, 2)
        
        # 创建放大区域 (3.5s到4s)
        # 找到对应的索引
        zoom_start_idx = np.where(future_steps >= 1.5)[0][0]  # 3.5s对应的是1.5s (因为我们从-1.9开始)
        zoom_end_idx = np.where(future_steps <= 2.0)[0][-1]   # 4.0s对应的是2.0s
        
        # 创建放大子图
        axins = fig.add_axes([0.65, 0.2, 0.3, 0.25])  # 右下角位置 [left, bottom, width, height]
        
        # 在放大区域绘制轨迹
        axins.plot(future_steps[zoom_start_idx:zoom_end_idx+1], preceding_pos_future[zoom_start_idx:zoom_end_idx+1], 'k--', linewidth=1)
        axins.plot(future_steps[zoom_start_idx:zoom_end_idx+1], following_pos_future[zoom_start_idx:zoom_end_idx+1], 'g-', linewidth=1)
        axins.plot(future_steps[zoom_start_idx:zoom_end_idx+1], moe_pred[zoom_start_idx:zoom_end_idx+1], color=self.model_colors['SimPreNet'], linestyle='--', linewidth=1)
        axins.fill_between(future_steps[zoom_start_idx:zoom_end_idx+1], 
                          moe_pred[zoom_start_idx:zoom_end_idx+1] - moe_std[zoom_start_idx:zoom_end_idx+1], 
                          moe_pred[zoom_start_idx:zoom_end_idx+1] + moe_std[zoom_start_idx:zoom_end_idx+1], 
                          color=self.model_colors['SimPreNet'], alpha=0.2)
        
        axins.plot(future_steps[zoom_start_idx:zoom_end_idx+1], seq2seq_pred[zoom_start_idx:zoom_end_idx+1], color=self.model_colors['Seq2Seq'], linestyle='--', linewidth=1)
        axins.fill_between(future_steps[zoom_start_idx:zoom_end_idx+1], 
                          seq2seq_pred[zoom_start_idx:zoom_end_idx+1] - seq2seq_std[zoom_start_idx:zoom_end_idx+1], 
                          seq2seq_pred[zoom_start_idx:zoom_end_idx+1] + seq2seq_std[zoom_start_idx:zoom_end_idx+1], 
                          color=self.model_colors['Seq2Seq'], alpha=0.2)
        
        axins.plot(future_steps[zoom_start_idx:zoom_end_idx+1], transformer_pred[zoom_start_idx:zoom_end_idx+1], color=self.model_colors['Transformer'], linestyle='--', linewidth=1)
        axins.fill_between(future_steps[zoom_start_idx:zoom_end_idx+1], 
                          transformer_pred[zoom_start_idx:zoom_end_idx+1] - transformer_std[zoom_start_idx:zoom_end_idx+1], 
                          transformer_pred[zoom_start_idx:zoom_end_idx+1] + transformer_std[zoom_start_idx:zoom_end_idx+1], 
                          color=self.model_colors['Transformer'], alpha=0.2)
        
        # 添加标记点
        zoom_marker_indices = np.arange(zoom_start_idx, zoom_end_idx+1, 2)
        axins.plot(future_steps[zoom_marker_indices], preceding_pos_future[zoom_marker_indices], 'ko', markersize=3)
        axins.plot(future_steps[zoom_marker_indices], following_pos_future[zoom_marker_indices], 'go', markersize=3)
        axins.plot(future_steps[zoom_marker_indices], moe_pred[zoom_marker_indices], 'o', color=self.model_colors['SimPreNet'], markersize=3)
        axins.plot(future_steps[zoom_marker_indices], seq2seq_pred[zoom_marker_indices], 'o', color=self.model_colors['Seq2Seq'], markersize=3)
        axins.plot(future_steps[zoom_marker_indices], transformer_pred[zoom_marker_indices], 'o', color=self.model_colors['Transformer'], markersize=3)
        
        # 设置放大区域的坐标轴
        axins.set_xlim(future_steps[zoom_start_idx], future_steps[zoom_end_idx])
        # 自动设置y轴范围，稍微扩大一点以便更好地显示
        y_min = min(
            np.min(preceding_pos_future[zoom_start_idx:zoom_end_idx+1]),
            np.min(following_pos_future[zoom_start_idx:zoom_end_idx+1]),
            np.min(moe_pred[zoom_start_idx:zoom_end_idx+1] - moe_std[zoom_start_idx:zoom_end_idx+1]),
            np.min(seq2seq_pred[zoom_start_idx:zoom_end_idx+1] - seq2seq_std[zoom_start_idx:zoom_end_idx+1]),
            np.min(transformer_pred[zoom_start_idx:zoom_end_idx+1] - transformer_std[zoom_start_idx:zoom_end_idx+1])
        ) - 0.5
        y_max = max(
            np.max(preceding_pos_future[zoom_start_idx:zoom_end_idx+1]),
            np.max(following_pos_future[zoom_start_idx:zoom_end_idx+1]),
            np.max(moe_pred[zoom_start_idx:zoom_end_idx+1] + moe_std[zoom_start_idx:zoom_end_idx+1]),
            np.max(seq2seq_pred[zoom_start_idx:zoom_end_idx+1] + seq2seq_std[zoom_start_idx:zoom_end_idx+1]),
            np.max(transformer_pred[zoom_start_idx:zoom_end_idx+1] + transformer_std[zoom_start_idx:zoom_end_idx+1])
        ) + 0.5
        axins.set_ylim(y_min, y_max)
        
        # 添加网格
        axins.grid(True, alpha=0.3)
        
        # 使用箭头标注放大区域
        from mpl_toolkits.axes_grid1.inset_locator import mark_inset
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray")
        
        # 保存图表
        plt.savefig(os.path.join(self.fig_path, f'average_trajectory_{self.type_labels[cf_type]}.png'), bbox_inches='tight')
        plt.close()
    
    def calculate_metrics(self, trajectories):
        """计算预测指标"""
        metrics = {
            'all': {'moe_rmse': [], 'seq2seq_rmse': [], 'transformer_rmse': []}
        }
        
        # 为每种类型创建指标存储
        for i in range(args['cf_type']):
            metrics[i] = {'moe_rmse': [], 'seq2seq_rmse': [], 'transformer_rmse': []}
        
        # 计算每种类型的指标
        for cf_type, traj_list in trajectories.items():
            for traj in traj_list:
                true_pos = traj['following_pos_future']
                
                # 计算RMSE
                moe_rmse = np.sqrt(np.mean((traj['moe_pred'] - true_pos) ** 2))
                seq2seq_rmse = np.sqrt(np.mean((traj['seq2seq_pred'] - true_pos) ** 2))
                transformer_rmse = np.sqrt(np.mean((traj['transformer_pred'] - true_pos) ** 2))
                
                # 存储指标
                metrics[cf_type]['moe_rmse'].append(moe_rmse)
                metrics[cf_type]['seq2seq_rmse'].append(seq2seq_rmse)
                metrics[cf_type]['transformer_rmse'].append(transformer_rmse)
                
                # 存储到总体指标
                metrics['all']['moe_rmse'].append(moe_rmse)
                metrics['all']['seq2seq_rmse'].append(seq2seq_rmse)
                metrics['all']['transformer_rmse'].append(transformer_rmse)
        
        # 计算平均指标
        for key in metrics:
            for model in ['moe', 'seq2seq', 'transformer']:
                metrics[key][f'{model}_avg'] = np.mean(metrics[key][f'{model}_rmse'])
        
        return metrics
    
    def plot_metrics(self, metrics):
        """绘制指标对比图"""
        # 准备数据
        cf_types = list(range(args['cf_type'])) + ['all']
        moe_avg = [metrics[t]['moe_avg'] for t in cf_types]
        seq2seq_avg = [metrics[t]['seq2seq_avg'] for t in cf_types]
        transformer_avg = [metrics[t]['transformer_avg'] for t in cf_types]
        
        # 设置x轴标签
        x_labels = [self.type_labels[i] for i in range(args['cf_type'])] + ['All']
        
        plt.figure(figsize=(10, 6), dpi=300)
        
        # 设置柱状图
        x = np.arange(len(cf_types))
        width = 0.25
        
        plt.bar(x - width, moe_avg, width, label='PreSimNet', color=self.model_colors['SimPreNet'])
        plt.bar(x, seq2seq_avg, width, label='BAT', color=self.model_colors['Seq2Seq'])
        plt.bar(x + width, transformer_avg, width, label='HLTP', color=self.model_colors['Transformer'])
        
        # 设置图表属性
        plt.title('Trajectory Prediction RMSE Comparison', fontproperties=self.font_prop, fontsize=14)
        plt.xlabel('Car-Following Type', fontproperties=self.font_prop, fontsize=12)
        plt.ylabel('RMSE (m)', fontproperties=self.font_prop, fontsize=12)
        plt.xticks(x, x_labels, fontproperties=self.font_prop)
        plt.legend(prop=self.font_prop)
        plt.grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签
        for i, v in enumerate(moe_avg):
            plt.text(i - width, v + 0.01, f'{v:.3f}', ha='center', fontsize=8, fontproperties=self.font_prop)
        for i, v in enumerate(seq2seq_avg):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=8, fontproperties=self.font_prop)
        for i, v in enumerate(transformer_avg):
            plt.text(i + width, v + 0.01, f'{v:.3f}', ha='center', fontsize=8, fontproperties=self.font_prop)
        
        # 保存图表
        plt.savefig(os.path.join(self.fig_path, 'trajectory_metrics_comparison.png'), bbox_inches='tight')
        plt.close()
    
    def run(self):
        """运行完整的可视化流程"""
        print("Loading models...")
        self.load_models()
        
        print("Collecting trajectory data...")
        trajectories = self.collect_trajectories()
        
        print("Visualizing trajectories...")
        self.visualize_trajectories(trajectories)
        
        print("Calculating metrics...")
        metrics = self.calculate_metrics(trajectories)
        
        print("Plotting metric comparison...")
        self.plot_metrics(metrics)
        
        print("Trajectory visualization completed!")


if __name__ == "__main__":
    visualizer = TrajectoryVisualizer()
    visualizer.run()