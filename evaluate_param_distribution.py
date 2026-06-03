from __future__ import print_function
import numpy as np
import torch as t
import loader2 as lo
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from config import *
import model.model_MoE_gru_new as model
import os
from tqdm import tqdm
import matplotlib as mpl
from scipy.stats import gaussian_kde, entropy
from scipy.spatial.distance import jensenshannon
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ParameterDistributionAnalyzer():
    def __init__(self):
        self.device = device
        self.fig_path = "fig/vis/param_distribution/"
        
        # 跟驰类型标签
        self.type_labels = {
            0: 'AV-HV',   # ACC Expert 1
            1: 'AV-AV',   # ACC Expert 2  
            2: 'HV-HV',   # IDM Expert 1
            3: 'HV-AV'    # IDM Expert 2
        }
        
        # ACC模型参数定义
        self.acc_param_names = [
            r'$k_p$ (Gap Error Gain, $s^{-2}$)',
            r'$k_d$ (Speed Difference Gain, $s^{-1}$)', 
            r'$v_{des}$ (Desired Speed, m/s)',
            r'$s_0$ (Min Safety Distance, m)',
            r'$T$ (Safe Time Headway, s)',
            r'$k_v$ (Speed Error Gain, $s^{-1}$)'
        ]
        
        # IDM模型参数定义
        self.idm_param_names = [
            r'$s_0$ (Min Safety Distance, m)',
            r'$T$ (Safe Time Headway, s)',
            r'$b$ (Comfortable Deceleration, $m/s^2$)',
            r'$a_{max}$ (Max Acceleration, $m/s^2$)',
            r'$v_{max}$ (Free Flow Speed, m/s)',
            r'$\delta$ (Acceleration Exponent)'
        ]
        
        if not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path)

    def collect_parameters(self, name):
        """收集验证集上所有样本的参数分布"""
        # 检查是否已存在npz文件
        npz_path = f'{self.fig_path}parameter_data.npz'
        if os.path.exists(npz_path):
            print("发现已存在的参数数据文件，直接加载...")
            data = np.load(npz_path, allow_pickle=True)
            
            acc_params = {
                0: data['acc_expert0'] if len(data['acc_expert0']) > 0 else None,
                1: data['acc_expert1'] if len(data['acc_expert1']) > 0 else None
            }
            idm_params = {
                0: data['idm_expert0'] if len(data['idm_expert0']) > 0 else None,
                1: data['idm_expert1'] if len(data['idm_expert1']) > 0 else None
            }
            
            print(f"加载完成:")
            for expert_idx in [0, 1]:
                acc_label = 'AV-HV' if expert_idx == 0 else 'AV-AV'
                idm_label = 'HV-HV' if expert_idx == 0 else 'HV-AV'
                print(f"  ACC {acc_label}: {len(acc_params[expert_idx]) if acc_params[expert_idx] is not None else 0} samples")
                print(f"  IDM {idm_label}: {len(idm_params[expert_idx]) if idm_params[expert_idx] is not None else 0} samples")
            
            return acc_params, idm_params
        
        print("未发现参数数据文件，开始重新收集...")
        args['train_flag'] = False
        
        # 初始化模型
        Encoder = model.Encoder(args).to(device)
        checkpoint = t.load('checkponint\ed64_inl20_ol20_drop0.1_tl1_nh4_od2_gama0.9_qv1_nt2_gru_new_2' + '/epoch' + name + '_e.tar')
        Encoder.load_state_dict(checkpoint['model_state_dict'])
        Encoder.eval()
        
        # 数据加载 - 使用测试集
        test_dataset = lo.HighSimDataset('../data/test_data.npy', args['in_length'], args['out_length'])
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], 
                                shuffle=False, num_workers=args['num_worker'],
                                pin_memory=True, drop_last=True)
        
        # 存储参数：每个跟驰类型对应一个专家
        # ACC: type 0,1 -> expert 0,1
        # IDM: type 2,3 -> expert 0,1  
        acc_params = {0: [], 1: []}  # AV-HV, AV-AV
        idm_params = {0: [], 1: []}  # HV-HV, HV-AV
        
        with t.no_grad():
            for data in tqdm(test_loader, desc="Collecting parameters"):
                hist, nextv, fut, cf_type = [x.to(device) for x in data]
                
                # 获取真实标签
                _, target = t.max(cf_type, dim=1)
                
                # 前向传播获取参数
                model_outputs = Encoder(hist)
                params = model_outputs['params']
                
                # 提取ACC和IDM参数
                acc_params_list = params['acc_params']  # [expert0, expert1]
                idm_params_list = params['idm_params']  # [expert0, expert1]
                
                # 根据真实标签收集对应专家的参数
                for i in range(target.shape[0]):
                    cf_label = target[i].item()
                    
                    if cf_label in [0, 1]:  # ACC类型
                        expert_idx = 0 if cf_label == 0 else 1  # AV-HV->expert0, AV-AV->expert1
                        param_values = acc_params_list[expert_idx][i, 0, :].cpu().numpy()  # [6]
                        acc_params[expert_idx].append(param_values)
                    
                    elif cf_label in [2, 3]:  # IDM类型
                        expert_idx = 0 if cf_label == 2 else 1  # HV-HV->expert0, HV-AV->expert1
                        param_values = idm_params_list[expert_idx][i, 0, :].cpu().numpy()  # [6]
                        idm_params[expert_idx].append(param_values)
        
        # 转换为numpy数组
        for expert_idx in [0, 1]:
            if acc_params[expert_idx]:
                acc_params[expert_idx] = np.array(acc_params[expert_idx])
            if idm_params[expert_idx]:
                idm_params[expert_idx] = np.array(idm_params[expert_idx])
        
        print(f"收集完成:")
        for expert_idx in [0, 1]:
            acc_label = 'AV-HV' if expert_idx == 0 else 'AV-AV'
            idm_label = 'HV-HV' if expert_idx == 0 else 'HV-AV'
            print(f"  ACC {acc_label}: {len(acc_params[expert_idx]) if acc_params[expert_idx] is not None else 0} samples")
            print(f"  IDM {idm_label}: {len(idm_params[expert_idx]) if idm_params[expert_idx] is not None else 0} samples")
        
        return acc_params, idm_params

    def calculate_divergences(self, data1, data2):
        """计算两个分布之间的KL散度和JS散度"""
        try:
            # 使用相同的支撑域
            x_min = min(np.min(data1), np.min(data2))
            x_max = max(np.max(data1), np.max(data2))
            x_vals = np.linspace(x_min, x_max, 1000)
            
            # 计算KDE
            kde1 = gaussian_kde(data1, bw_method=0.3)
            kde2 = gaussian_kde(data2, bw_method=0.3)
            
            # 计算概率密度
            p = kde1(x_vals)
            q = kde2(x_vals)
            
            # 归一化
            p = p / np.sum(p)
            q = q / np.sum(q)
            
            # 添加小的常数避免log(0)
            p = p + 1e-10
            q = q + 1e-10
            
            # 计算KL散度 (对称化)
            kl_div = 0.5 * (entropy(p, q) + entropy(q, p))
            
            # 计算JS散度
            js_div = jensenshannon(p, q)
            
            return kl_div, js_div
        except:
            return np.nan, np.nan

    def plot_parameter_distribution(self, acc_params, idm_params):
        """绘制参数分布图：2x6布局"""
        fig, axes = plt.subplots(2, 6, figsize=(24, 8), dpi=600)
        
        # 颜色设置 - ACC和IDM使用不同颜色方案
        acc_colors = ['#1f77b4', '#ff7f0e']  # 蓝色和橙色 for ACC
        idm_colors = ['#2ca02c', '#d62728']  # 绿色和红色 for IDM
        
        # 类别标签
        acc_labels = ['AV-HV', 'AV-AV']
        idm_labels = ['HV-HV', 'HV-AV']
        
        # 参数范围设置
        acc_ranges = [
            (0, 0.2),     # kp: 比例增益
            (0, 1.5),     # kd: 微分增益
            (0, 50.0),  # v_des: 期望速度
            (0, 10.0),    # h0: 最小安全间距
            (0.0, 6),     # T: 安全时距
            (0, 0.15)      # kv: 速度控制增益
        ]
        
        idm_ranges = [
            (0, 4),       # s0: 最小安全间距
            (0, 4.0),     # T: 安全时距
            (2, 6.0),     # b: 舒适减速度
            (0, 5.0),     # a_max: 最大加速度
            (0, 50.0),    # v_max: 期望速度
            (0, 7.0)    # delta: 加速度指数
        ]
        
        # 绘制IDM参数 (第一行)
        for param_idx in range(6):
            ax = axes[0, param_idx]
            
            # 检查数据是否存在
            data_exists = []
            for expert_idx in [0, 1]:
                if (idm_params[expert_idx] is not None and 
                    len(idm_params[expert_idx]) > 0):
                    data_exists.append(True)
                else:
                    data_exists.append(False)
            
            if any(data_exists):
                all_data = []
                valid_experts = []
                
                for expert_idx in [0, 1]:
                    if data_exists[expert_idx]:
                        param_data = idm_params[expert_idx][:, param_idx]
                        all_data.append(param_data)
                        valid_experts.append(expert_idx)
                        
                        # 计算自适应分箱数量
                        data_range = np.max(param_data) - np.min(param_data)
                        axis_range = idm_ranges[param_idx][1] - idm_ranges[param_idx][0]
                        # 基于范围比例调整分箱数，分箱粒度更细
                        bins = max(50, int(200 * data_range / axis_range))
                        
                        # 绘制直方图
                        ax.hist(param_data, bins=bins, alpha=0.3,
                               color=idm_colors[expert_idx], 
                               density=True)
                        
                        # 绘制KDE
                        kde = gaussian_kde(param_data, bw_method=0.3)
                        x_vals = np.linspace(np.min(param_data), np.max(param_data), 1000)
                        ax.plot(x_vals, kde(x_vals), color=idm_colors[expert_idx], linewidth=2)
                
                # 计算统计信息
                stats_text = ""
                for i, expert_idx in enumerate(valid_experts):
                    param_data = all_data[i]
                    mean_val = np.mean(param_data)
                    std_val = np.std(param_data)
                    stats_text += f"{idm_labels[expert_idx]} (n={len(param_data)}): μ={mean_val:.3f}, σ={std_val:.3f}\n"
                
                # 计算散度（如果有两个专家的数据）
                if len(valid_experts) == 2:
                    kl_div, js_div = self.calculate_divergences(all_data[0], all_data[1])
                    stats_text += f"KL Div: {kl_div:.3f}\nJS Div: {js_div:.3f}"
                
                # 添加统计信息 - 移到右上角
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                       bbox=dict(facecolor='white', alpha=0.8),
                       fontsize=8, verticalalignment='top', horizontalalignment='right')
            
            # 设置横坐标范围
            ax.set_xlim(idm_ranges[param_idx])
            ax.set_xlabel('Parameter Value', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.grid(True, alpha=0.3)
            # 标题移到下方并添加序号 - 不加粗
            ax.set_xlabel(f'{self.idm_param_names[param_idx]}', fontsize=10)
            ax.text(0.02, 0.98, f'({chr(97 + param_idx)})', transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top')
        # 绘制ACC参数 (第二行)
        for param_idx in range(6):
            ax = axes[1, param_idx]
            
            # 检查数据是否存在
            data_exists = []
            for expert_idx in [0, 1]:
                if (acc_params[expert_idx] is not None and 
                    len(acc_params[expert_idx]) > 0):
                    data_exists.append(True)
                else:
                    data_exists.append(False)
            
            if any(data_exists):
                all_data = []
                valid_experts = []
                
                for expert_idx in [0, 1]:
                    if data_exists[expert_idx]:
                        param_data = acc_params[expert_idx][:, param_idx]
                        all_data.append(param_data)
                        valid_experts.append(expert_idx)
                        
                        # 计算自适应分箱数量
                        data_range = np.max(param_data) - np.min(param_data)
                        axis_range = acc_ranges[param_idx][1] - acc_ranges[param_idx][0]
                        # 基于范围比例调整分箱数，分箱粒度更细
                        bins = max(50, int(200 * data_range / axis_range))
                        
                        # 绘制直方图
                        ax.hist(param_data, bins=bins, alpha=0.6, 
                               color=acc_colors[expert_idx], 
                               density=True)
                        
                        # 绘制KDE
                        kde = gaussian_kde(param_data, bw_method=0.3)
                        x_vals = np.linspace(np.min(param_data), np.max(param_data), 1000)
                        ax.plot(x_vals, kde(x_vals), color=acc_colors[expert_idx], linewidth=2)
                
                # 计算统计信息
                stats_text = ""
                for i, expert_idx in enumerate(valid_experts):
                    param_data = all_data[i]
                    mean_val = np.mean(param_data)
                    std_val = np.std(param_data)
                    stats_text += f"{acc_labels[expert_idx]} (n={len(param_data)}): μ={mean_val:.3f}, σ={std_val:.3f}\n"
                
                # 计算散度（如果有两个专家的数据）
                if len(valid_experts) == 2:
                    kl_div, js_div = self.calculate_divergences(all_data[0], all_data[1])
                    stats_text += f"KL Div: {kl_div:.3f}\nJS Div: {js_div:.3f}"
                
                # 添加统计信息 - 移到右上角
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                       bbox=dict(facecolor='white', alpha=0.8),
                       fontsize=8, verticalalignment='top', horizontalalignment='right')
            
            # 设置横坐标范围
            ax.set_xlim(acc_ranges[param_idx])
            # 设置纵坐标范围 - 为vdes参数(索引2)设置特定范围
            if param_idx == 2:  # vdes参数
                ax.set_ylim(0, 0.08)
            ax.set_xlabel('Parameter Value', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.grid(True, alpha=0.3)
            # 标题移到下方并添加序号 (g-l for ACC parameters) - 不加粗
            ax.set_xlabel(f'{self.acc_param_names[param_idx]}', fontsize=10)
            # 添加子图标签 (a), (b), (c), (d), (e)
            ax.text(0.02, 0.98, f'({chr(103 + param_idx)})', transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top')
        # 添加行标签和颜色图例
        # IDM Model标签和图例
        fig.text(0.01, 0.75, 'IDM Model', fontsize=14, 
                rotation=90, verticalalignment='center')
        # IDM颜色图例
        fig.text(0.025, 0.82, '■', fontsize=14, color=idm_colors[0], verticalalignment='center')
        fig.text(0.035, 0.82, 'HV-HV', fontsize=12, verticalalignment='center')
        fig.text(0.025, 0.68, '■', fontsize=14, color=idm_colors[1], verticalalignment='center')
        fig.text(0.035, 0.68, 'HV-AV', fontsize=12, verticalalignment='center')
        
        # ACC Model标签和图例
        fig.text(0.01, 0.25, 'ACC Model', fontsize=14, 
                rotation=90, verticalalignment='center')
        # ACC颜色图例
        fig.text(0.025, 0.32, '■', fontsize=14, color=acc_colors[0], verticalalignment='center')
        fig.text(0.035, 0.32, 'AV-HV', fontsize=12, verticalalignment='center')
        fig.text(0.025, 0.18, '■', fontsize=14, color=acc_colors[1], verticalalignment='center')
        fig.text(0.035, 0.18, 'AV-AV', fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.08)
        
        # 保存图片
        plt.savefig(f'{self.fig_path}parameter_distribution.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'{self.fig_path}parameter_distribution.pdf', dpi=600, bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存
        print(f"图片已保存到: {self.fig_path}parameter_distribution.png")
        
        # 保存参数数据
        np.savez_compressed(
            f'{self.fig_path}parameter_data.npz',
            acc_expert0=acc_params[0] if acc_params[0] is not None else np.array([]),
            acc_expert1=acc_params[1] if acc_params[1] is not None else np.array([]),
            idm_expert0=idm_params[0] if idm_params[0] is not None else np.array([]),
            idm_expert1=idm_params[1] if idm_params[1] is not None else np.array([]),
            acc_param_names=self.acc_param_names,
            idm_param_names=self.idm_param_names
        )

if __name__ == '__main__':
    analyzer = ParameterDistributionAnalyzer()
    acc_params, idm_params = analyzer.collect_parameters('21')  # 使用第21个epoch的模型
    analyzer.plot_parameter_distribution(acc_params, idm_params)