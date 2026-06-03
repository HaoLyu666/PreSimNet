from __future__ import print_function
import numpy as np
import torch
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
import scipy.stats as stats

class TrajectoryErrorAnalyzer:
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
        self.fig_path = "fig/error_analysis/"
        if not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path)
        
        # 设置字体和样式
        self.font_prop = FontProperties(family='Times New Roman')
        
        # 模型权重路径
        self.moe_path = "checkponint\ed64_inl20_ol20_drop0.1_tl1_nh4_od2_gama0.9_qv1_nt2_gru_new_2" + '/epoch21_e.tar'
        self.seq2seq_path = "checkponint\ed64_inl20_ol20_drop0.1_tl1_nh4_od2_gama0.8_qv1_nt2_bat" + '/baseline_epoch20.tar'
        self.transformer_path = "checkponint\ed64_inl20_ol20_drop0.1_tl2_nh4_od2_gama0.8_qv1_nt2_hltp" + '/baseline_epoch19.tar'
        
        # 置信区间设置
        self.confidence_level = 0.95  # 95%置信区间
    
    def load_models(self):
        """加载三个模型"""
        print("加载模型中...")
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
        print("模型加载完成")
    
    def evaluate_test_set(self):
        """评估测试集，为每种类型选择100个batch的数据"""
        print("开始评估测试集...")
        
        # 加载测试数据
        test_dataset = lo.HighSimDataset('../data/test_data.npy', args['in_length'], args['out_length'])
        test_loader = DataLoader(
            test_dataset,
            batch_size=args['batch_size'], 
            shuffle=False,  # 不打乱顺序
            num_workers=args['num_worker'],
            pin_memory=True, 
            drop_last=True
        )
        
        # 初始化存储每个时间步的误差
        # 结构: {类型: {模型: {时间步: [误差列表]}}}
        errors_by_timestep = {}
        for cf_type in range(args['cf_type']):
            errors_by_timestep[cf_type] = {
                'moe': {t: [] for t in range(args['out_length'])},
                'seq2seq': {t: [] for t in range(args['out_length'])},
                'transformer': {t: [] for t in range(args['out_length'])}
            }
        
        # 存储最终位置误差(FDE)
        final_errors = {}
        for cf_type in range(args['cf_type']):
            final_errors[cf_type] = {
                'moe': [],
                'seq2seq': [],
                'transformer': []
            }
        
        # 记录每种类型的样本数量
        type_counts = {i: 0 for i in range(args['cf_type'])}
        total_samples = 0
        
        with torch.no_grad():
            for data in tqdm(test_loader, desc="评估测试集"):
                hist, nextv, fut, cf_type = [x.to(device) for x in data]
                
                # 获取样本类型
                _, target = torch.max(cf_type, dim=1)
                
                # 筛选有效样本 - 检查是否有类型已经达到100个batch
                if all(count >= 100 * args['batch_size'] for count in type_counts.values()):
                    break
                
                # 检查当前batch中是否有我们需要的类型
                valid_types = [i for i in range(args['cf_type']) if type_counts[i] < 100 * args['batch_size']]
                if not any(t.item() in valid_types for t in target):
                    continue
                
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
                
                # 获取真实轨迹
                following_pos_future = fut[:, :, 2]  # 后车位置未来真值
                
                # 计算每个样本的误差
                for i in range(len(hist)):
                    sample_type = target[i].item()
                    
                    # 如果该类型已经收集了足够的样本，则跳过
                    if type_counts[sample_type] >= 100 * args['batch_size']:
                        continue
                    
                    true_pos = following_pos_future[i]
                    
                    # 计算每个时间步的误差
                    for t in range(args['out_length']):
                        # 计算绝对误差
                        moe_error = abs(moe_traj_pred[i, t] - true_pos[t]).item()
                        seq2seq_error = abs(seq2seq_traj_pred[i, t] - true_pos[t]).item()
                        transformer_error = abs(transformer_traj_pred[i, t] - true_pos[t]).item()
                        
                        # 存储误差
                        errors_by_timestep[sample_type]['moe'][t].append(moe_error)
                        errors_by_timestep[sample_type]['seq2seq'][t].append(seq2seq_error)
                        errors_by_timestep[sample_type]['transformer'][t].append(transformer_error)
                    
                    # 存储最终位置误差(FDE)
                    final_t = args['out_length'] - 1
                    final_errors[sample_type]['moe'].append(abs(moe_traj_pred[i, final_t] - true_pos[final_t]).item())
                    final_errors[sample_type]['seq2seq'].append(abs(seq2seq_traj_pred[i, final_t] - true_pos[final_t]).item())
                    final_errors[sample_type]['transformer'].append(abs(transformer_traj_pred[i, final_t] - true_pos[final_t]).item())
                    
                    type_counts[sample_type] += 1
                    total_samples += 1
            
        print(f"评估完成，共处理 {total_samples} 个样本")
        for i in range(args['cf_type']):
            print(f"类型 {self.type_labels[i]}: {type_counts[i]} 个样本")
        
        return errors_by_timestep, final_errors
    
    def calculate_metrics(self, errors_by_timestep, final_errors):
        """计算每个时间步的RMSE和置信区间"""
        print("计算评估指标...")
        
        # 初始化结果存储
        metrics = {}
        for cf_type in range(args['cf_type']):
            metrics[cf_type] = {
                'rmse': {
                    'moe': {'mean': [], 'lower': [], 'upper': []},
                    'seq2seq': {'mean': [], 'lower': [], 'upper': []},
                    'transformer': {'mean': [], 'lower': [], 'upper': []}
                },
                'fde': {
                    'moe': {'mean': 0, 'lower': 0, 'upper': 0},
                    'seq2seq': {'mean': 0, 'lower': 0, 'upper': 0},
                    'transformer': {'mean': 0, 'lower': 0, 'upper': 0}
                }
            }
        
        # 计算每个时间步的RMSE和置信区间
        for cf_type in range(args['cf_type']):
            for model in ['moe', 'seq2seq', 'transformer']:
                for t in range(args['out_length']):
                    errors = errors_by_timestep[cf_type][model][t]
                    if not errors:
                        continue
                    
                    # 计算RMSE
                    rmse = np.sqrt(np.mean(np.square(errors)))
                    
                    # 计算置信区间
                    # 使用bootstrap方法计算RMSE的置信区间
                    n_bootstrap = 1000
                    bootstrap_rmses = []
                    
                    for _ in range(n_bootstrap):
                        # 有放回地随机抽样
                        bootstrap_sample = np.random.choice(errors, size=len(errors), replace=True)
                        bootstrap_rmse = np.sqrt(np.mean(np.square(bootstrap_sample)))
                        bootstrap_rmses.append(bootstrap_rmse)
                    
                    # 计算置信区间
                    lower_bound = np.percentile(bootstrap_rmses, (1 - self.confidence_level) * 100 / 2)
                    upper_bound = np.percentile(bootstrap_rmses, 100 - (1 - self.confidence_level) * 100 / 2)
                    
                    # 存储结果
                    metrics[cf_type]['rmse'][model]['mean'].append(rmse)
                    metrics[cf_type]['rmse'][model]['lower'].append(lower_bound)
                    metrics[cf_type]['rmse'][model]['upper'].append(upper_bound)
                
                # 计算FDE及其置信区间
                final_model_errors = final_errors[cf_type][model]
                if final_model_errors:
                    fde_mean = np.mean(final_model_errors)
                    
                    # 使用bootstrap方法计算FDE的置信区间
                    n_bootstrap = 1000
                    bootstrap_fdes = []
                    
                    for _ in range(n_bootstrap):
                        bootstrap_sample = np.random.choice(final_model_errors, size=len(final_model_errors), replace=True)
                        bootstrap_fde = np.mean(bootstrap_sample)
                        bootstrap_fdes.append(bootstrap_fde)
                    
                    fde_lower = np.percentile(bootstrap_fdes, (1 - self.confidence_level) * 100 / 2)
                    fde_upper = np.percentile(bootstrap_fdes, 100 - (1 - self.confidence_level) * 100 / 2)
                    
                    metrics[cf_type]['fde'][model]['mean'] = fde_mean
                    metrics[cf_type]['fde'][model]['lower'] = fde_lower
                    metrics[cf_type]['fde'][model]['upper'] = fde_upper
        
        return metrics
    
    def plot_error_over_time(self, metrics):
        """绘制误差随时间变化的图表"""
        print("绘制误差随时间变化图表...")
        
        # 创建时间步数组（从0.1秒开始）
        time_steps = np.arange(1, args['out_length'] + 1) * 0.1
        
        # 为每种类型绘制RMSE随时间变化的图表
        for cf_type in range(args['cf_type']):
            plt.figure(figsize=(10, 6), dpi=300)
            
            # 绘制三种模型的RMSE曲线及置信区间
            for model, color in self.model_colors.items():
                model_key = model.lower() if model != 'SimPreNet' else 'moe'
                
                rmse_mean = metrics[cf_type]['rmse'][model_key]['mean']
                rmse_lower = metrics[cf_type]['rmse'][model_key]['lower']
                rmse_upper = metrics[cf_type]['rmse'][model_key]['upper']
                
                plt.plot(time_steps, rmse_mean, color=color, linewidth=2, label=f'{model}')
                plt.fill_between(time_steps, rmse_lower, rmse_upper, color=color, alpha=0.2)
            
            # 设置图表属性
            plt.title(f'{self.type_labels[cf_type]} 类型 RMSE 随时间变化', fontproperties=self.font_prop, fontsize=14)
            plt.xlabel('预测时间 (秒)', fontproperties=self.font_prop, fontsize=12)
            plt.ylabel('RMSE (米)', fontproperties=self.font_prop, fontsize=12)
            plt.legend(prop=self.font_prop)
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            plt.savefig(os.path.join(self.fig_path, f'rmse_over_time_{self.type_labels[cf_type]}.png'), bbox_inches='tight')
            plt.close()
        
        # 绘制所有类型的平均RMSE随时间变化的图表
        plt.figure(figsize=(10, 6), dpi=300)
        
        # 计算所有类型的平均RMSE
        all_type_metrics = {
            'rmse': {
                'moe': {'mean': np.zeros(args['out_length']), 'lower': np.zeros(args['out_length']), 'upper': np.zeros(args['out_length'])},
                'seq2seq': {'mean': np.zeros(args['out_length']), 'lower': np.zeros(args['out_length']), 'upper': np.zeros(args['out_length'])},
                'transformer': {'mean': np.zeros(args['out_length']), 'lower': np.zeros(args['out_length']), 'upper': np.zeros(args['out_length'])}
            }
        }
        
        # 计算所有类型的平均值
        for cf_type in range(args['cf_type']):
            for model in ['moe', 'seq2seq', 'transformer']:
                for metric_type in ['mean', 'lower', 'upper']:
                    all_type_metrics['rmse'][model][metric_type] += np.array(metrics[cf_type]['rmse'][model][metric_type]) / args['cf_type']
        
        # 绘制三种模型的平均RMSE曲线及置信区间
        for model, color in self.model_colors.items():
            model_key = model.lower() if model != 'SimPreNet' else 'moe'
            
            rmse_mean = all_type_metrics['rmse'][model_key]['mean']
            rmse_lower = all_type_metrics['rmse'][model_key]['lower']
            rmse_upper = all_type_metrics['rmse'][model_key]['upper']
            
            plt.plot(time_steps, rmse_mean, color=color, linewidth=2, label=f'{model}')
            plt.fill_between(time_steps, rmse_lower, rmse_upper, color=color, alpha=0.2)
        
        # 设置图表属性
        plt.title('所有类型平均 RMSE 随时间变化', fontproperties=self.font_prop, fontsize=14)
        plt.xlabel('预测时间 (秒)', fontproperties=self.font_prop, fontsize=12)
        plt.ylabel('RMSE (米)', fontproperties=self.font_prop, fontsize=12)
        plt.legend(prop=self.font_prop)
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        plt.savefig(os.path.join(self.fig_path, 'rmse_over_time_all_types.png'), bbox_inches='tight')
        plt.close()
    
    def plot_fde_comparison(self, metrics):
        """绘制FDE对比图表"""
        print("绘制FDE对比图表...")
        
        # 准备数据
        cf_types = list(range(args['cf_type'])) + ['all']
        
        # 计算所有类型的平均FDE
        all_type_fde = {
            'moe': {'mean': 0, 'lower': 0, 'upper': 0},
            'seq2seq': {'mean': 0, 'lower': 0, 'upper': 0},
            'transformer': {'mean': 0, 'lower': 0, 'upper': 0}
        }
        
        for cf_type in range(args['cf_type']):
            for model in ['moe', 'seq2seq', 'transformer']:
                for metric_type in ['mean', 'lower', 'upper']:
                    all_type_fde[model][metric_type] += metrics[cf_type]['fde'][model][metric_type] / args['cf_type']
        
        # 添加所有类型的平均值
        for cf_type in range(args['cf_type']):
            metrics[cf_type]['fde']['all_types'] = all_type_fde
        
        # 为每种类型绘制FDE柱状图
        for cf_type in range(args['cf_type']):
            plt.figure(figsize=(8, 6), dpi=300)
            
            # 准备数据
            models = ['SimPreNet', 'Seq2Seq', 'Transformer']
            model_keys = ['moe', 'seq2seq', 'transformer']
            fde_means = [metrics[cf_type]['fde'][key]['mean'] for key in model_keys]
            fde_errors = [
                [metrics[cf_type]['fde'][key]['mean'] - metrics[cf_type]['fde'][key]['lower'] for key in model_keys],
                [metrics[cf_type]['fde'][key]['upper'] - metrics[cf_type]['fde'][key]['mean'] for key in model_keys]
            ]
            
            # 绘制柱状图
            x = np.arange(len(models))
            width = 0.6
            
            bars = plt.bar(x, fde_means, width, label='FDE', 
                          color=[self.model_colors[model] for model in models])
            
            # 添加误差线
            plt.errorbar(x, fde_means, yerr=fde_errors, fmt='none', ecolor='black', capsize=5)
            
            # 在柱状图上添加数值标签
            for i, v in enumerate(fde_means):
                plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontproperties=self.font_prop)
            
            # 设置图表属性
            plt.title(f'{self.type_labels[cf_type]} 类型 FDE 对比', fontproperties=self.font_prop, fontsize=14)
            plt.xlabel('模型', fontproperties=self.font_prop, fontsize=12)
            plt.ylabel('FDE (米)', fontproperties=self.font_prop, fontsize=12)
            plt.xticks(x, models, fontproperties=self.font_prop)
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            plt.savefig(os.path.join(self.fig_path, f'fde_comparison_{self.type_labels[cf_type]}.png'), bbox_inches='tight')
            plt.close()
        
        # 绘制所有类型的平均FDE柱状图
        plt.figure(figsize=(8, 6), dpi=300)
        
        # 准备数据
        models = ['SimPreNet', 'Seq2Seq', 'Transformer']
        model_keys = ['moe', 'seq2seq', 'transformer']
        fde_means = [all_type_fde[key]['mean'] for key in model_keys]
        fde_errors = [
            [all_type_fde[key]['mean'] - all_type_fde[key]['lower'] for key in model_keys],
            [all_type_fde[key]['upper'] - all_type_fde[key]['mean'] for key in model_keys]
        ]
        
        # 绘制柱状图
        x = np.arange(len(models))
        width = 0.6
        
        bars = plt.bar(x, fde_means, width, label='FDE', 
                      color=[self.model_colors[model] for model in models])
        
        # 添加误差线
        plt.errorbar(x, fde_means, yerr=fde_errors, fmt='none', ecolor='black', capsize=5)
        
        # 在柱状图上添加数值标签
        for i, v in enumerate(fde_means):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontproperties=self.font_prop)
        
        # 设置图表属性
        plt.title('所有类型平均 FDE 对比', fontproperties=self.font_prop, fontsize=14)
        plt.xlabel('模型', fontproperties=self.font_prop, fontsize=12)
        plt.ylabel('FDE (米)', fontproperties=self.font_prop, fontsize=12)
        plt.xticks(x, models, fontproperties=self.font_prop)
        plt.grid(True, alpha=0.3)
        
        # 保存图表
        plt.savefig(os.path.join(self.fig_path, 'fde_comparison_all_types.png'), bbox_inches='tight')
        plt.close()
    
    def run(self):
        """运行完整的评估和可视化流程"""
        print("开始轨迹预测误差分析...")
        
        # 加载模型
        self.load_models()
        
        # 评估测试集
        errors_by_timestep, final_errors = self.evaluate_test_set()
        
        # 计算评估指标
        metrics = self.calculate_metrics(errors_by_timestep, final_errors)
        
        # 绘制RMSE随时间变化的图表
        self.plot_error_over_time(metrics)
        
        # 绘制FDE对比图表
        # self.plot_fde_comparison(metrics)
        
        print("轨迹预测误差分析完成!")


if __name__ == "__main__":
    analyzer = TrajectoryErrorAnalyzer()
    analyzer.run()