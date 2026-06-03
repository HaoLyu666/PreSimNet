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
from scipy.stats import gaussian_kde
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import gaussian_kde, entropy,  wasserstein_distance
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings('ignore')

class GapVelocityVisualEvaluate():
    def __init__(self):
        self.device = device
        self.fig_path = "fig/vis/gap_velocity_pred/"
        # 重新排序：AV-AV, AV-HV, HV-AV, HV-HV
        self.type_labels = {
            1: 'AV-AV',
            0: 'AV-HV', 
            3: 'HV-AV',
            2: 'HV-HV'
        }
        # 创建有序的类型列表用于绘图
        self.ordered_types = [1, 0, 3, 2]  # AV-AV, AV-HV, HV-AV, HV-HV
        if not os.path.exists(self.fig_path):
            os.makedirs(self.fig_path)

    def collect_samples(self, name):
        args['train_flag'] = False
        
        # 初始化模型
        Encoder = model.Encoder(args).to(device)
        checkpoint = t.load(args['path'] + '/epoch' + name + '_e.tar')
        Encoder.load_state_dict(checkpoint['model_state_dict'])
        Encoder.eval()
        Koop = model.predictor(args)
        
        # 数据加载
        test_dataset = lo.HighSimDataset('../data/test_data.npy', args['in_length'], args['out_length'])
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], 
                               shuffle=False, num_workers=args['num_worker'],
                               pin_memory=True, drop_last=True)
        
        # 存储每种类型的预测结果，包括gap和velocity
        predictions = {i: {'gap': [], 'true_gap': [], 'velocity': [], 'true_velocity': []} 
                      for i in range(args['cf_type'])}
        type_counts = {i: 0 for i in range(args['cf_type'])}
        
        with t.no_grad():
            for data in tqdm(test_loader):
                hist, nextv, fut, cf_type = [x.to(device) for x in data]
                
                _, target = t.max(cf_type, dim=1)
                model_outputs = Encoder(hist)
                veh_state = hist[:, -1, [4, 2]]  # 间隙、速度
                predictions_output = Koop.forward(model_outputs, nextv, veh_state)
                
                # 提取预测结果 - 使用参数预测的值
                dynamic_pred = predictions_output['dynamic_pred']  # [B, out_length, 2]
                
                # 筛选有效样本
                true_gap = fut[:, -1, 0]
                valid_mask = (true_gap <= 120)
                
                for i in range(args['cf_type']):
                    if type_counts[i] >= 100 * args['batch_size']:
                        continue
                        
                    type_mask = (target == i)
                    mask = type_mask & valid_mask
                    if not mask.any():
                        continue
                        
                    pred_gap = dynamic_pred[mask, -1, 0]
                    true_gap_i = fut[mask, -1, 0]
                    pred_velocity = dynamic_pred[mask, -1, 1]
                    true_velocity_i = fut[mask, -1, 1]
                    
                    predictions[i]['gap'].extend(pred_gap.cpu().numpy())
                    predictions[i]['true_gap'].extend(true_gap_i.cpu().numpy())
                    predictions[i]['velocity'].extend(pred_velocity.cpu().numpy())
                    predictions[i]['true_velocity'].extend(true_velocity_i.cpu().numpy())
                    
                    type_counts[i] += mask.sum().item()
                
                if all(count >= 100 * args['batch_size'] for count in type_counts.values()):
                    break
        
        # 检查是否所有类型都有足够的样本
        for i in range(args['cf_type']):
            if type_counts[i] < 100 * args['batch_size']:
                print(f"警告: 类型 {self.type_labels[i]} 的样本数量不足 ({type_counts[i]})")
        
        return predictions

    def _plot_single_scatter(self, ax, true, pred, title, xlabel, ylabel, metric_type='gap'):
        """在给定的ax上绘制单个散点图，包含密度图和密度条"""
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        xy = np.vstack([true, pred])
        z = gaussian_kde(xy)(xy)

        rmse = mean_squared_error(true, pred, squared=False)
        bias = np.mean(np.array(pred) - np.array(true))
        n = len(true)

        scatter = ax.scatter(true, pred, c=z, cmap='gist_ncar', s=8, alpha=0.7)

        slope, intercept, _, _, _ = linregress(true, pred)

        # 设置坐标轴范围
        if metric_type == 'gap':
            xlim = ylim = (0, 120)
            x_line = np.linspace(0, 120, 100)
        else:  # velocity
            xlim = ylim = (0, 40)
            x_line = np.linspace(0, 40, 100)

        ax.plot(x_line, slope * x_line + intercept, '--', color='aquamarine', linewidth=1)
        ax.plot(x_line, x_line, color='dimgray', linewidth=1)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')  # 确保散点图为正方形
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        # 移除标题，因为用户不需要
        # ax.set_title(title, fontsize=12, fontweight='bold')

        # 添加密度图
        divider = make_axes_locatable(ax)
        ax_histx = divider.append_axes("top", 1.2, pad=0.01, sharex=ax)
        ax_histy = divider.append_axes("right", 1.2, pad=0.01, sharey=ax)
        ax_cbar = divider.append_axes("right", 0.2, pad=0.3)
        
        # 画顶部核密度估计 (True的分布)
        kde_true = gaussian_kde(true)
        x_vals = np.linspace(xlim[0], xlim[1], 1000)
        ax_histx.plot(x_vals, kde_true(x_vals), color='deepskyblue', linewidth=1.5)
        ax_histx.fill_between(x_vals, 0, kde_true(x_vals), color='lightblue', alpha=0.4)
        ax_histx.axis('off')
        
        # 画右侧核密度估计 (Pred的分布)
        kde_pred = gaussian_kde(pred)
        y_vals = np.linspace(ylim[0], ylim[1], 1000)
        ax_histy.plot(kde_pred(y_vals), y_vals, color='red', linewidth=1.5)
        ax_histy.fill_betweenx(y_vals, 0, kde_pred(y_vals), color='lightcoral', alpha=0.4)
        ax_histy.axis('off')
        
        # 添加密度条
        cbar = plt.colorbar(scatter, cax=ax_cbar)
        cbar.ax.tick_params(labelsize=8)

        # 计算JS散度
        kde_true_js = gaussian_kde(true, bw_method=0.3)
        kde_pred_js = gaussian_kde(pred, bw_method=0.3)
        x_vals_js = np.linspace(min(min(true), min(pred)), max(max(true), max(pred)), 1000)
        p = kde_true_js(x_vals_js)
        q = kde_pred_js(x_vals_js)
        p += 1e-10
        q += 1e-10
        p /= p.sum()
        q /= q.sum()
        js_div = jensenshannon(p, q)

        # 添加统计信息
        unit = 'm' if metric_type == 'gap' else 'm/s'
        stats = f'n={n}\nRMSE={rmse:.3f}{unit}\nBias={bias:.3f}{unit}\nJS Div={js_div:.3f}'
        ax.text(0.05, 0.75, stats, transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8), fontsize=9)
        
        return scatter

    def visualize_results(self, predictions):
        # 创建2x4的子图布局
        fig, axes = plt.subplots(2, 4, figsize=(24, 10), dpi=600)
        # 移除总标题
        # fig.suptitle('Gap and Velocity Prediction Results', fontsize=16, fontweight='bold')
        
        # 按照指定顺序绘制：AV-AV, AV-HV, HV-AV, HV-HV
        for col_idx, cf_type in enumerate(self.ordered_types):
            type_label = self.type_labels[cf_type]
            
            # 上排：Gap预测
            self._plot_single_scatter(
                axes[0, col_idx],
                predictions[cf_type]['true_gap'],
                predictions[cf_type]['gap'],
                f'{type_label}',
                'Ground Truth Gap (m)',
                'Predicted Gap (m)',
                'gap'
            )
            
            # 下排：Velocity预测
            self._plot_single_scatter(
                axes[1, col_idx],
                predictions[cf_type]['true_velocity'],
                predictions[cf_type]['velocity'],
                f'{type_label}',
                'Ground Truth Velocity (m/s)',
                'Predicted Velocity (m/s)',
                'velocity'
            )
            
            # 在第二行底部添加标签，使用与图中其他部分一致的字体
            label_map = {0: '(a) AV-AV', 1: '(b) AV-HV', 2: '(c) HV-AV', 3: '(d) HV-HV'}
            axes[1, col_idx].text(0.5, -0.25, label_map[col_idx], transform=axes[1, col_idx].transAxes,
                                 fontsize=14, ha='center')
        
        # 移除行标签（Gap和Velocity）
        # axes[0, 0].text(-0.15, 0.5, 'Gap', transform=axes[0, 0].transAxes, 
        #                 fontsize=14, fontweight='bold', rotation=90, 
        #                 verticalalignment='center')
        # axes[1, 0].text(-0.15, 0.5, 'Velocity', transform=axes[1, 0].transAxes, 
        #                 fontsize=14, fontweight='bold', rotation=90, 
        #                 verticalalignment='center')
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.05, bottom=0.1)
        
        # 保存图片
        plt.savefig(f'{self.fig_path}gap_velocity_combined.pdf', dpi=600, bbox_inches='tight')
        plt.show()
        
        # 保存预测结果
        np.savez_compressed(
            f'{self.fig_path}gap_velocity_predictions.npz',
            **{self.type_labels[i]: np.array([
                predictions[i]['true_gap'],
                predictions[i]['gap'],
                predictions[i]['true_velocity'],
                predictions[i]['velocity']
            ]) for i in range(args['cf_type'])}
        )

if __name__ == '__main__':
    evaluator = GapVelocityVisualEvaluate()
    predictions = evaluator.collect_samples('21')  # 使用第21个epoch的模型，可以根据需要修改
    evaluator.visualize_results(predictions)