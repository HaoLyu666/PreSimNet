import numpy as np
import torch as t
import matplotlib.pyplot as plt
import seaborn as sns
from config import *
import model.model_MoE_gru_new as model
import os
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap

class MappingVisualizer:
    def __init__(self):
        self.device = device
        self.fig_path = "fig/vis/mapping/"
        # 参考evaluate_gap_vis.py中的类型顺序
        self.type_labels = {
            1: 'AV-AV',
            0: 'AV-HV', 
            3: 'HV-AV',
            2: 'HV-HV'
        }
        # 创建有序的类型列表用于绘图
        self.ordered_types = [1, 0, 3, 2]  # AV-AV, AV-HV, HV-AV, HV-HV
        
        # 创建自定义颜色映射（新的十六进制颜色）
        colors = [
            # (0x6a/255, 0x06/255, 0x24/255),   # #6a0624
            (0x86/255, 0x08/255, 0x24/255),   # #860824
            (0xb7/255, 0x1c/255, 0x2c/255),   # #b71c2c
            (0xfe/255, 0xab/255, 0x88/255),   # #feab88
            (0xfb/255, 0xd2/255, 0xbc/255),   # #bd2b2c (修正)
            (0xc7/255, 0xe0/255, 0xed/255),   # #c7e0ed
            (0x0f/255, 0xaf/255, 0xd2/255),   # #0fafd2 (修正)
            (0x32/255, 0x7d/255, 0xb7/255),   # #327db7
            (0x13/255, 0x4b/255, 0x87/255),   # #134b87
            # (0x05/255, 0x30/255, 0x61/255)    # #053061
        ]
        # colors = [
        #     # (0x6a/255, 0x06/255, 0x24/255),   # #6a0624
        #     # (0x86/255, 0x08/255, 0x24/255),   # #860824
        #     # (0xb7/255, 0x1c/255, 0x2c/255),   # #b71c2c
        #     (161 / 255, 212 / 255, 165 / 255),  # #feab88
        #     (193 / 255, 207 / 255, 155 / 255),  # #bd2b2c (修正)
        #     (225 / 255, 201 / 255, 146 / 255),  # #c7e0ed
        #     (245 / 255, 194 / 255, 141 / 255),  # #0fafd2 (修正)
        #     (241 / 255, 165 / 255, 151 / 255),  # #327db7
        #     (241 / 255, 88 / 255, 84 / 255),  # #134b87
        #     # (0x05/255, 0x30/255, 0x61/255)    # #053061
        # ]
        self.custom_cmap = LinearSegmentedColormap.from_list('custom', colors[::-1], N=256)
        
        if not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path)
    
    def load_model_and_extract_mapping(self, epoch_name='21'):
        """加载模型并提取mapping参数"""
        args['train_flag'] = False
        
        # 使用存在的checkpoint路径
        checkpoint_path = 'checkponint/ed64_inl20_ol20_drop0.1_tl1_nh4_od2_gama0.9_qv1_nt2_gru_new_2/epoch' + epoch_name + '_e.tar'
        
        # 初始化模型
        encoder = model.Encoder(args).to(self.device)
        checkpoint = t.load(checkpoint_path)
        encoder.load_state_dict(checkpoint['model_state_dict'])
        encoder.eval()
        
        # 提取mapping参数 [cf_type, out_length, in_length]
        mapping = encoder.mapping.data.cpu().numpy()
        print(f"Mapping shape: {mapping.shape}")
        
        return mapping
    
    def visualize_mapping(self, mapping, epoch_name='21'):
        """可视化mapping参数"""
        # 创建2x2的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=600)
        axes = axes.flatten()
        
        # 为每种跟驰类型创建热力图
        for idx, cf_type in enumerate(self.ordered_types):
            ax = axes[idx]
            
            # 获取当前类型的mapping矩阵 [out_length, in_length]
            type_mapping = mapping[cf_type]  # shape: [out_length, in_length]
            
            # 在未来轴(out_length)上进行softmax
            type_mapping_softmax = F.softmax(t.tensor(type_mapping), dim=1).numpy()
            
            
            
            # 创建热力图，使用自定义配色和网格效果
            im = ax.imshow(type_mapping_softmax,
                          cmap='RdBu_r',  # 使用自定义渐变配色
                          aspect='auto',
                          origin='lower',
                          interpolation='nearest')  # 使用nearest避免过度平滑，产生网格感
            
            # 设置坐标轴标签
            ax.set_xlabel('Historical Time Steps', fontsize=8)
            ax.set_ylabel('Future Time Steps', fontsize=8)
            
            # 设置刻度 - 显示每个时间步
            ax.set_xticks(range(0, args['in_length']))
            ax.set_xticklabels(range(-args['in_length']+1, 1), fontsize=6)  # 历史时间步从-19到0
            ax.set_yticks(range(0, args['out_length']))
            ax.set_yticklabels(range(args['out_length'], 0, -1), fontsize=6)  # 未来时间步从1到20
            
            # 添加网格线增强网格感
            ax.set_xticks(np.arange(-0.5, args['in_length'], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, args['out_length'], 1), minor=True)
            ax.grid(which='minor', color='white', linestyle='-', linewidth=0.1, alpha=0.3)
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax, shrink=1)
            # cbar.set_label('Attention Weight', fontsize=9)
            cbar.ax.tick_params(labelsize=8)
            
            # 标题放在下方
            label_map = {0: '(a) AV-AV', 1: '(b) AV-HV', 2: '(c) HV-AV', 3: '(d) HV-HV'}
            ax.text(0.5, -0.15, label_map[idx], transform=ax.transAxes,
                   fontsize=12, ha='center', fontweight='normal')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        # 保存图片
        try:
            plt.savefig(f'{self.fig_path}mapping_visualization2_epoch{epoch_name}.pdf',
                       dpi=300, bbox_inches='tight')
        except PermissionError:
            print(f"Warning: Could not save PDF file (permission denied)")
        
        plt.savefig(f'{self.fig_path}mapping_visualization2_epoch{epoch_name}.png',
                   dpi=600, bbox_inches='tight')
        plt.savefig(f'{self.fig_path}mapping_visualization2_epoch{epoch_name}.pdf',
                    dpi=600, bbox_inches='tight')
        plt.close()
        
        print(f"Mapping visualization saved to {self.fig_path}")
        
        # 保存mapping数据
        np.savez_compressed(
            f'{self.fig_path}mapping_data_epoch{epoch_name}.npz',
            mapping=mapping,
            type_labels=self.type_labels,
            ordered_types=self.ordered_types
        )
    
    def analyze_mapping_patterns(self, mapping):
        """分析mapping模式"""
        print("\n=== Mapping Pattern Analysis ===")
        
        for idx, cf_type in enumerate(self.ordered_types):
            type_label = self.type_labels[cf_type]
            type_mapping = mapping[cf_type]
            
            # 在未来轴上进行softmax
            type_mapping_softmax = F.softmax(t.tensor(type_mapping), dim=0).numpy()
            
            # 计算每个历史时间步的平均权重
            hist_weights = np.mean(type_mapping_softmax, axis=0)
            
            # 计算每个未来时间步的平均权重
            fut_weights = np.mean(type_mapping_softmax, axis=1)
            
            print(f"\n{type_label}:")
            print(f"  Historical attention focus (top 3 steps): {np.argsort(hist_weights)[-3:][::-1] + 1}")
            print(f"  Future prediction focus (top 3 steps): {np.argsort(fut_weights)[-3:][::-1] + 1}")
            print(f"  Max attention weight: {np.max(type_mapping_softmax):.4f}")
            print(f"  Min attention weight: {np.min(type_mapping_softmax):.4f}")

if __name__ == '__main__':
    visualizer = MappingVisualizer()
    
    # 加载模型并提取mapping
    mapping = visualizer.load_model_and_extract_mapping('21')  # 可以修改epoch名称
    
    # 可视化mapping
    visualizer.visualize_mapping(mapping, '21')
    
    # 分析mapping模式
    visualizer.analyze_mapping_patterns(mapping)
    
    print("\nMapping visualization completed!")