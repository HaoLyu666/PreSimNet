import os
import numpy as np
import torch as t
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from tqdm import tqdm
from torch.utils.data import DataLoader
import model.model_MoE_gru_new as model
from config import *
from sklearn.metrics import mean_squared_error

class LongTermPredictor:
    def __init__(self, model_epoch=None):
        self.device = device
        self.model_epoch = model_epoch if model_epoch else args['model_epoch']
        self.save_dir = args['long_term_save_dir']
        self.dt = args['dt']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.f_length = args['f_length']
        self.prediction_step = args['prediction_step']
        self.confidence_level = args['confidence_level']
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # 设置字体
        self.font_prop = FontProperties(family='Times New Roman', size=16)
        
        # 设置颜色
        self.colors = {
            'ground_truth': '#1E90FF',  # 海绿色
            'sim_one_step': '#FF6347',  # 番茄红
            'sim_iteration': '#fcda43',  # 钢蓝色
            'prediction': '#fc6f68',    # 中紫色
            'leading': '#2E8B57',       # 巧克力色
            'history': '#7079de',       # 道奇蓝
            'marker': '#FF4500'         # 橙红色
        }
        
        # 类型标签
        self.type_labels = {
            0: 'AV-HV',
            1: 'AV-AV',
            2: 'HV-HV',
            3: 'HV-AV'
        }
        
        # 放大区域颜色
        self.zoom_colors = {
            1: 'lightblue',   # 浅珊瑚色
            2: 'lightgreen',   # 浅绿色
            3: 'lightpink'     # 浅粉色
        }
        
        # 放大区域位置和大小设置
        self.zoom_regions = {
            0: {  # AV-HV
                1: {
                    'time': (5.49, 6.51),
                    'position': [0.1, 0.15, 0.20, 0.15],
                    'arrow_style': 'top_to_bottom',  # 箭头样式：右到左
                    'lines': [2, 3, 4],  # 显示哪些线 (1:前车, 2:真实, 3:预测, 4:模拟)
                    'marker_size': 2,  # 标记点大小
                    'line_width': 1,  # 线宽
                    'offset': [-0.04, 0.04]  # 箭头起始点偏置 [x_offset, y_offset]
                },
                2: {
                    'time': (14, 19.01), 
                    'position': [0.38, 0.28, 0.3, 0.20],
                    'arrow_style': 'top_to_bottom',  # 箭头样式：右到左
                    'lines': [1, 2, 3, 4],  # 显示哪些线 (1:前车, 2:真实, 3:预测, 4:模拟)
                    'marker_size': 2,  # 标记点大小
                    'line_width': 1,  # 线宽
                    'offset': [0.00, 0.05]  # 箭头起始点偏置 [x_offset, y_offset]
                },
                # 3: {
                #     'time': (48, 50),
                #     'position': [0.45, 0.72, 0.25, 0.20],
                #     'arrow_style': 'left_to_right',  # 箭头样式：左到右
                #     'lines': [1, 2, 4],  # 不显示直接预测线
                #     'marker_size': 2,
                #     'line_width': 1,
                #     'offset': [0, 0]
                # }
            },
            # 1: {  # AV-AV
            #     1: {
            #         'time': (5, 6),
            #         'position': [0.1, 0.17, 0.15, 0.18],
            #         'arrow_style': 'top_to_bottom',
            #         'lines': [1, 2, 3, 4],
            #         'marker_size': 1.5,
            #         'line_width': 1,
            #         'offset': [-0.05, 0.017]  # 箭头起始点偏置 [x_offset, y_offset]
            #     },
            #     2: {'time': (12, 15), 'position': [0.675, 0.07, 0.3, 0.20], 'arrow_style': 'right_to_left','marker_size': 1.5,
            #         'line_width': 1, 'offset': [-0.03, 0]},
            #     3: {'time': (38.5, 40.51),
            #         'position': [0.29, 0.25, 0.2, 0.20],
            #         'arrow_style': 'left_to_right',  # 箭头样式：左到右
            #         'lines': [2, 4],  # 不显示直接预测线
            #         'marker_size': 1.5,
            #         'line_width': 1,
            #         'offset': [0.05, 0.06]}
            # },
            1: {  # AV-AV
                1: {
                    'time': (7, 10.01),
                    'position': [0.36, 0.30, 0.19, 0.18],
                    'arrow_style': 'top_to_bottom',
                    'lines': [2, 3, 4],
                    'marker_size': 1.5,
                    'line_width': 1,
                    'offset': [-0.03, 0.05]  # 箭头起始点偏置 [x_offset, y_offset]
                },
                2: {'time': (27, 31.01), 'position': [0.67, 0.1, 0.3, 0.2], 'arrow_style': 'right_to_left',
                    'marker_size': 1.5, 'line_width': 1, 'offset': [-0.02, 0.03]}
            },
            2: {  # HV-HV
                1: {'time': (5, 6), 'position': [0.1, 0.18, 0.15, 0.12], 'arrow_style': 'top_to_bottom','lines': [2,3, 4],
                    'marker_size': 1.5, 'line_width': 1, 'offset': [-0.05, 0.045]},
                2: {'time': (24, 32.01), 'position': [0.46, 0.1, 0.33, 0.2], 'arrow_style': 'bottom_to_top','lines': [1, 2, 4],
                    'marker_size': 1.5, 'line_width': 1, 'offset': [0, 0.07]},
                3: {'time': (48, 50.01), 'position': [0.83, 0.15, 0.15, 0.2],
                    'arrow_style': 'bottom_to_top',  # 箭头样式：左到右
                    'lines': [2, 4],  # 不显示直接预测线
                    'marker_size': 1.5,
                    'line_width': 1,
                    'offset': [0.08, 0.09]}
            },
            3: {  # HV-AV
                1: {
                    'time': (8.5, 10.01),
                    'position': [0.37, 0.32, 0.18, 0.16],
                    'arrow_style': 'top_to_bottom',
                    'lines': [2, 3, 4],
                    'marker_size': 1.5,
                    'line_width': 1,
                    'offset': [-0.025, 0.04]  # 箭头起始点偏置 [x_offset, y_offset]
                },
                2: {'time': (17.50, 21.01), 'position': [0.63, 0.1, 0.34, 0.22], 'arrow_style': 'right_to_left',
                    'marker_size': 1.5, 'line_width': 1, 'offset': [0, 0.05]},
                # 3: {'time': (46, 48), 'position': [0.82, 0.15, 0.15, 0.2],
                #     'lines': [1, 2, 4],  # 不显示直接预测线
                #     'marker_size': 1.5,
                #     'line_width': 1,
                #     'offset': [0, 0.05]}
            }
        }
        # 加载模型
        self._load_models()
        
    def _load_models(self):
        """加载预训练模型"""
        print("Loading models...")
        
        # 加载编码器
        self.encoder = model.Encoder(args).to(device)
        checkpoint = t.load('checkponint\ed64_inl20_ol20_drop0.1_tl1_nh4_od2_gama0.9_qv1_nt2_gru_new_2' + '/epoch' + self.model_epoch + '_e.tar')
        self.encoder.load_state_dict(checkpoint['model_state_dict'])
        self.encoder.eval()
        
        # 加载预测器
        self.predictor = model.predictor(args)
        
    def _prepare_input_features(self, trajectory_data, start_idx, window_size=None):
        """准备模型输入特征，与loader2.py保持一致"""
        if window_size is None:
            window_size = self.in_length
            
        # 提取窗口数据
        window = trajectory_data[start_idx:start_idx+window_size]
        
        # 创建输入特征张量 - 根据loader2.py中的特征构成
        features = np.zeros((1, window_size, self.f_length))
        
        # 计算前车加速度与后车加速度的差值
        hist_acc_diff = window[:, 5] - window[:, 7]
        
        # 填充特征 - 与loader2.py中保持一致
        features[0, :, :-1] = window[:, 4:]  # 前车速度
        features[0, :, -3:-1] = features[0, :, -3:-1]-features[0, -1, -3]
        features[0, :, 9] = hist_acc_diff  # 加速度差值
        
        return t.tensor(features, dtype=t.float32).to(device)
    
    def _get_next_vehicle_state(self, trajectory_data, idx):
        """获取下一个时间步的前车状态"""
        if idx < len(trajectory_data) - 1:
            return trajectory_data[idx+1, 4]  # 前车速度
        else:
            return trajectory_data[-1, 4]  # 使用最后一个时间步的前车速度
    
    def _update_state_with_parameters(self, gap, v_follow, v_lead, params, model_type, dt=None):
        """使用车辆跟驰模型参数更新状态"""
        if dt is None:
            dt = self.dt
            
        if model_type == 'ACC':
            # ACC模型参数: kp, kd, v_des, h0, h1, kv
            kp, kd, v_des, h0, h1, kv = params
            
            # 计算期望间隙
            desired_gap = h0 + h1 * v_follow
            
            # 计算加速度
            gap_error = gap - desired_gap
            speed_error = v_lead - v_follow
            
            # ACC控制律
            acc = kp * gap_error + kd * speed_error + kv * (v_des - v_follow)
            
        else:  # IDM模型
            # IDM参数: s0, T, b, a_max, v_max, delta
            s0, T, b, a_max, v_max, delta = params
            
            # 计算期望间隙
            s_star = s0 + max(0, v_follow * T + v_follow * (v_follow - v_lead) / (2 * np.sqrt(a_max * b)))
            
            # IDM加速度计算
            acc = a_max * (1 - (v_follow / v_max) ** delta - (s_star / max(gap, 0.1)) ** 2)
        
        # 限制加速度范围
        acc = np.clip(acc, -5.0, 5.0)
        
        # 更新速度和位置
        new_v = v_follow + acc * dt
        new_v = max(0, new_v)  # 确保速度非负
        
        # 返回新的状态
        return new_v, acc
        
    def _get_best_expert_params(self, model_outputs):
        """从模型输出中获取概率最高的专家参数"""
        # 获取跟驰类型概率
        cf_probs = model_outputs['params']['cf_probs'][0]  # [cf_type]
        
        # 找到概率最高的跟驰类型
        best_type_idx = t.argmax(cf_probs).item()
        
        # 根据类型选择对应的专家参数
        if best_type_idx < 2:  # ACC模型 (0或1)
            params = model_outputs['params']['acc_params'][best_type_idx][0, 0].cpu().numpy()
            model_type = 'ACC'
        else:  # IDM模型 (2或3)
            params = model_outputs['params']['idm_params'][best_type_idx-2][0, 0].cpu().numpy()
            model_type = 'IDM'
        
        return params, model_type, best_type_idx, cf_probs.cpu().numpy()
        
    def _update_state_with_prediction(self, pred_gap, pred_v, dt=None):
        """直接使用预测值更新状态"""
        if dt is None:
            dt = self.dt
        return pred_v, pred_gap
        
    # 方法1: 使用固定参数进行单步更新
    def predict_trajectory_fixed_params(self, trajectory_data, start_idx, prediction_horizon=None):
        """使用固定参数预测长期轨迹"""
        if prediction_horizon is None:
            prediction_horizon = args['prediction_horizon']
            
        # 准备初始输入特征
        features = self._prepare_input_features(trajectory_data, start_idx-self.in_length, self.in_length)
        
        # 模型预测
        with t.no_grad():
            model_outputs = self.encoder(features)
        
        # 获取最佳专家参数
        params, model_type, best_type_idx, cf_probs = self._get_best_expert_params(model_outputs)
        
        # 获取前车速度序列
        v_lead = trajectory_data[start_idx:start_idx+prediction_horizon, 4]
        
        # 初始化结果数组
        gap_pred = np.zeros(prediction_horizon)
        v_pred = np.zeros(prediction_horizon)
        a_pred = np.zeros(prediction_horizon)
        
        # 获取初始状态
        gap = trajectory_data[start_idx-1, 8]  # 初始间隙
        v_follow = trajectory_data[start_idx-1, 6]  # 初始速度
        
        # 逐步更新状态
        for i in range(prediction_horizon):
            # 使用车辆跟驰模型更新状态
            v_follow, acc = self._update_state_with_parameters(gap, v_follow, v_lead[i], params, model_type)
            
            # 更新间隙
            gap = gap + (v_lead[i] - v_follow) * self.dt
            
            # 存储预测结果
            gap_pred[i] = gap
            v_pred[i] = v_follow
            a_pred[i] = acc
        
        return {
            'gap_pred': gap_pred,
            'v_pred': v_pred,
            'a_pred': a_pred,
            'v_lead': v_lead,
            'model_type': model_type,
            'best_type_idx': best_type_idx,
            'cf_probs': cf_probs,
            'params': params
        }

    # 方法2: 使用迭代预测
    def predict_trajectory_iterative(self, trajectory_data, start_idx, prediction_horizon=None):
        """使用迭代方法预测长期轨迹"""
        if prediction_horizon is None:
            prediction_horizon = args['prediction_horizon']
            
        history_window_size = self.in_length
        prediction_step = self.prediction_step
        
        # 初始化结果数组
        direct_pred_pos = np.zeros(prediction_horizon)  # 直接预测的位置
        direct_pred_gap = np.zeros(prediction_horizon)  # 直接预测的间隙
        direct_pred_v = np.zeros(prediction_horizon)    # 直接预测的速度
        
        sim_iteration_gap = np.zeros(prediction_horizon)  # 迭代模拟的间隙
        sim_iteration_v = np.zeros(prediction_horizon)    # 迭代模拟的速度
        sim_iteration_a = np.zeros(prediction_horizon)    # 迭代模拟的加速度
        
        # 获取前车速度序列
        v_lead = trajectory_data[start_idx:start_idx+prediction_horizon, 4]
        
        # 获取初始状态
        initial_gap = trajectory_data[start_idx-1, 8]    # 初始间隙
        initial_v = trajectory_data[start_idx-1, 6]      # 初始速度
        
        # 准备初始输入特征
        current_features = self._prepare_input_features(trajectory_data, start_idx-history_window_size, history_window_size)
        
        # 迭代预测
        for i in range(0, prediction_horizon, prediction_step):
            # 确定当前迭代的步数
            steps = min(prediction_step, prediction_horizon - i)
            
            # 模型预测
            with t.no_grad():
                model_outputs = self.encoder(current_features)
                predictions = self.predictor.forward(model_outputs, 
                                                t.tensor(v_lead[i:i+steps], dtype=t.float32).unsqueeze(0).to(device),
                                                t.tensor([[initial_gap, initial_v]], dtype=t.float32).to(device))
            
            # 提取预测结果
            direct_pred = predictions['direct_pred'].squeeze().cpu().numpy()  # [steps, 1]
            dynamic_pred = predictions['dynamic_pred'].squeeze().cpu().numpy()  # [steps, 2]
            
            # 存储直接预测结果
            direct_pred_pos[i:i+steps] = direct_pred[:steps, 0]
            
            # 计算直接预测的间隙和速度 (通过位置差分)
            if i == 0:
                # 第一步使用初始值
                prev_pos = trajectory_data[start_idx-1, 11]  # 初始后车位置
                lead_pos = trajectory_data[start_idx-1, 12]  # 初始前车位置
            else:
                # 使用上一步的预测结果
                prev_pos = direct_pred_pos[i-1]
                # 前车位置需要根据前车速度积分
                lead_pos = trajectory_data[start_idx-1, 12] + np.sum(v_lead[:i]) * self.dt
            
            # 计算间隙和速度
            for j in range(steps):
                # 更新前车位置
                curr_lead_pos = lead_pos + v_lead[i+j] * self.dt
                
                # 计算间隙
                direct_pred_gap[i+j] = curr_lead_pos - direct_pred_pos[i+j]
                
                # 计算速度 (位置差分)
                if j == 0 and i == 0:
                    direct_pred_v[i+j] = initial_v  # 第一步使用初始速度
                else:
                    if j == 0:
                        # 使用上一个预测步的最后一个位置计算
                        direct_pred_v[i+j] = (direct_pred_pos[i+j] - direct_pred_pos[i-1]) / self.dt
                    else:
                        direct_pred_v[i+j] = (direct_pred_pos[i+j] - direct_pred_pos[i+j-1]) / self.dt
                
                # 更新前车位置
                lead_pos = curr_lead_pos
            
            # 存储动力学模型预测结果
            sim_iteration_gap[i:i+steps] = dynamic_pred[:steps, 0]
            sim_iteration_v[i:i+steps] = dynamic_pred[:steps, 1]
            
            # 计算加速度 (速度差分)
            if i == 0:
                sim_iteration_a[i] = (sim_iteration_v[i] - initial_v) / self.dt
            else:
                sim_iteration_a[i] = (sim_iteration_v[i] - sim_iteration_v[i-1]) / self.dt
            
            for j in range(1, steps):
                sim_iteration_a[i+j] = (sim_iteration_v[i+j] - sim_iteration_v[i+j-1]) / self.dt
            
            # 更新初始状态
            initial_gap = sim_iteration_gap[i+steps-1]
            initial_v = sim_iteration_v[i+steps-1]
            
            # 准备下一次迭代的输入特征
            if i + steps < prediction_horizon:
                # 创建新的特征数组
                new_features = np.zeros_like(current_features.cpu().numpy())
                
                # 使用预测结果更新特征
                for j in range(history_window_size):
                    if j < history_window_size - steps:
                        # 保留旧特征
                        new_features[0, j] = current_features[0, j+steps].cpu().numpy()
                    else:
                        # 添加新预测
                        idx = j - (history_window_size - steps)
                        
                        # 计算前车加速度 (简化为0或通过速度差分)
                        lead_acc = 0  # 简化处理
                        
                        # 计算后车加速度
                        follow_acc = sim_iteration_a[i+idx]
                        
                        # 计算加速度差值
                        acc_diff = lead_acc - follow_acc
                        
                        # 更新特征 - 与loader2.py保持一致
                        new_features[0, j, 0] = v_lead[i+idx]  # 前车速度
                        new_features[0, j, 1] = lead_acc  # 前车加速度
                        new_features[0, j, 2] = sim_iteration_v[i+idx]  # 后车速度
                        new_features[0, j, 3] = follow_acc  # 后车加速度
                        new_features[0, j, 4] = sim_iteration_gap[i+idx]  # 车间间隙
                        new_features[0, j, 5] = sim_iteration_gap[i+idx]  # 车头间距 (简化为与间隙相同)
                        new_features[0, j, 6] = v_lead[i+idx] - sim_iteration_v[i+idx]  # 速度差
                        
                        # 计算位置 (简化处理)
                        if i+idx == 0:
                            follow_pos = trajectory_data[start_idx-1, 11] + sim_iteration_v[i+idx] * self.dt
                            lead_pos = trajectory_data[start_idx-1, 12] + v_lead[i+idx] * self.dt
                        else:
                            # 使用速度积分更新位置
                            follow_pos = direct_pred_pos[i+idx-1] + sim_iteration_v[i+idx] * self.dt
                            lead_pos = follow_pos + sim_iteration_gap[i+idx]
                        
                        new_features[0, j, 7] = follow_pos  # 后车位置
                        new_features[0, j, 8] = lead_pos  # 前车位置
                        new_features[0, j, 9] = acc_diff  # 加速度差值
                
                # 更新特征
                current_features = t.tensor(new_features, dtype=t.float32).to(device)
        
        # 返回预测结果
        return {
            'direct_pred_pos': direct_pred_pos,
            'direct_pred_gap': direct_pred_gap,
            'direct_pred_v': direct_pred_v,
            'sim_iteration_gap': sim_iteration_gap,
            'sim_iteration_v': sim_iteration_v,
            'sim_iteration_a': sim_iteration_a,
            'v_lead': v_lead
        }
        
    def predict_long_term(self, trajectory, key, prediction_steps=None):
        """长时间预测"""
        if prediction_steps is None:
            prediction_steps = args['prediction_horizon']
            
        # 获取原始数据
        raw_data = trajectory['raw_data']
        
        # 获取轨迹类型
        traj_type = trajectory['type']
        type_names = {0: "AV-HV", 1: "AV-AV", 2: "HV-HV", 3: "HV-AV"}
        type_name = type_names[traj_type]
        
        # 初始化历史窗口（前20个时间步）
        history_window_size = self.in_length  # 2秒，每步0.1秒
        prediction_window_size = self.out_length  # 2秒预测
        
        # 确保轨迹足够长
        if len(raw_data) < history_window_size + prediction_steps:
            print(f"Warning: Trajectory {key} is too short for long-term prediction")
            return
        
        # 准备初始输入
        hist_features = self._prepare_input_features(raw_data, 0, history_window_size)
        
        # 获取初始状态
        initial_gap = raw_data[history_window_size-1, 8]  # 初始间隙
        initial_v_follow = raw_data[history_window_size-1, 6]  # 初始后车速度
        
        # 存储真实值和预测值
        time_steps = np.arange(prediction_steps) * self.dt  # 预测时间步
        
        # 真实值
        true_v_follow = raw_data[history_window_size:history_window_size+prediction_steps, 6]
        true_gap = raw_data[history_window_size:history_window_size+prediction_steps, 8]
        v_lead = raw_data[history_window_size:history_window_size+prediction_steps, 4]
        
        # 预测值
        sim_one_step_v = np.zeros(prediction_steps)
        sim_one_step_gap = np.zeros(prediction_steps)
        sim_one_step_a = np.zeros(prediction_steps)
        
        # 方法2: 基于位置的迭代预测
        direct_pred_pos = np.zeros(prediction_steps)  # 直接预测的位置
        direct_pred_gap = np.zeros(prediction_steps)  # 直接预测的间隙
        direct_pred_v = np.zeros(prediction_steps)    # 直接预测的速度
        direct_pred_a = np.zeros(prediction_steps)    # 直接预测的加速度
        
        # 方法3: 基于间隙和速度的迭代预测
        dynamic_pred_gap = np.zeros(prediction_steps)  # 动态预测的间隙
        dynamic_pred_v = np.zeros(prediction_steps)    # 动态预测的速度
        dynamic_pred_a = np.zeros(prediction_steps)    # 动态预测的加速度
        dynamic_pred_pos = np.zeros(prediction_steps) 
        
        # 获取前车速度序列
        v_lead = raw_data[history_window_size-1:history_window_size+prediction_steps, 4]
        # 获取前车加速度序列
        a_lead = raw_data[history_window_size:history_window_size+prediction_steps, 5]
        # 获取前车位置序列
        pos_lead = raw_data[history_window_size:history_window_size+prediction_steps, 12]
        
        # 对AV-HV类型的第8条轨迹进行特殊处理
        if traj_type == 0 and key == 7:  # AV-HV类型(0)的第8条轨迹(索引7)
            print(f"Applying special processing for AV-HV trajectory {key}...")
            # 从第250个数据点开始进行积分修正
            start_correction_idx = 250
            if start_correction_idx < len(pos_lead):
                # 使用积分方法修正位置：pos[i] = pos[i-1] + v_lead[i-1] * dt
                for i in range(start_correction_idx, len(pos_lead)):
                    if i > 0:
                        # 获取对应的前车速度索引
                        v_idx = history_window_size + i - 1
                        if v_idx < len(raw_data):
                            v_lead_current = raw_data[v_idx, 4]
                            a_lead_current = raw_data[v_idx, 5]
                            pos_lead[i] = pos_lead[i-1] + v_lead_current * self.dt+ a_lead_current * self.dt **2 *0.5
                print(f"Position correction applied from index {start_correction_idx} to {len(pos_lead)-1}")
        # # 获取后车加速度序列
        a_follow = raw_data[history_window_size:history_window_size + prediction_steps, 7]
        # 计算车长（间距-间隙）
        vehicle_length = raw_data[0, 9] - raw_data[0, 8]
        
        with t.no_grad():
            model_outputs = self.encoder(hist_features)
            
            # 获取最佳专家参数
            params, model_type, best_type_idx, cf_probs = self._get_best_expert_params(model_outputs)
            
            # 方法1: 使用固定参数进行单步更新
            current_v = initial_v_follow
            current_gap = initial_gap
            
            for i in range(prediction_steps):
                # 使用车辆跟驰模型更新状态
                current_v, acc = self._update_state_with_parameters(
                    current_gap, current_v, v_lead[i], params, model_type
                )
                
                # 更新间隙
                if i < len(v_lead) - 1:
                    current_gap = current_gap + (v_lead[i+1] - current_v) * self.dt
                else:
                    current_gap = current_gap + (v_lead[i] - current_v) * self.dt
                
                # 存储预测结果
                sim_one_step_v[i] = current_v
                sim_one_step_gap[i] = current_gap
                sim_one_step_a[i] = acc
            
            # 方法2: 基于位置的迭代预测
            # 准备初始输入特征
            current_features_pos = hist_features.clone()
            
            # 获取初始位置和状态
            initial_gap = raw_data[history_window_size-1, 8]
            initial_v = raw_data[history_window_size-1, 6]
            initial_follow_pos = raw_data[history_window_size-1, 11]
            initial_lead_pos = raw_data[history_window_size-1, 12]
            
            # 记录当前位置作为归一化基准
            base_pos = initial_follow_pos
            
            for i in range(0, prediction_steps, prediction_window_size):
                # 限制预测步数
                steps = min(prediction_window_size, prediction_steps - i)
                
                # 获取当前状态
                model_outputs = self.encoder(current_features_pos)
                
                # 获取下一个前车速度序列
                end_idx = min(i+steps, len(v_lead))
                next_v_lead = t.tensor(v_lead[i:end_idx], dtype=t.float32).unsqueeze(0).to(device)
                
                # 预测
                predictions = self.predictor.forward(model_outputs, 
                                                next_v_lead,
                                                t.tensor([[initial_gap, initial_v]], dtype=t.float32).to(device))
                
                # 提取预测结果 - 位置预测
                direct_pred = predictions['direct_pred'].squeeze().cpu().numpy()
                
                # 存储直接预测结果 - 位置
                for j in range(min(steps, len(direct_pred))):
                    if i+j < prediction_steps:
                        # 检查 direct_pred 是一维还是二维数组
                        if len(direct_pred.shape) > 1:
                            # 二维数组情况
                            direct_pred_pos[i+j] = direct_pred[j, 0] + base_pos
                        else:
                            # 一维数组情况
                            direct_pred_pos[i+j] = direct_pred[j] + base_pos

                # 计算直接预测的间隙和速度 (通过位置差分)
                for j in range(min(steps, len(direct_pred))):
                    if i+j < prediction_steps:
                        # 使用前车真实位置计算间隙
                        direct_pred_gap[i+j] = pos_lead[i+j] - direct_pred_pos[i+j] - vehicle_length
                        
                        # 计算速度 (位置差分)
                        if j == 0 and i == 0:
                            direct_pred_v[i+j] = initial_v  # 第一步使用初始速度
                        else:
                            if j == 0:
                                # 使用上一个预测步的最后一个位置计算
                                direct_pred_v[i+j] = (direct_pred_pos[i+j] - direct_pred_pos[i-1]) / self.dt
                            else:
                                direct_pred_v[i+j] = (direct_pred_pos[i+j] - direct_pred_pos[i+j-1]) / self.dt
                
                # 计算加速度 (速度差分)
                if i == 0:
                    direct_pred_a[i] = (direct_pred_v[i] - initial_v) / self.dt
                else:
                    direct_pred_a[i] = (direct_pred_v[i] - direct_pred_v[i-1]) / self.dt
                
                for j in range(1, min(steps, len(direct_pred))):
                    if i+j < prediction_steps:
                        direct_pred_a[i+j] = (direct_pred_v[i+j] - direct_pred_v[i+j-1]) / self.dt
                
                # 准备下一次迭代的输入特征
                if i + steps < prediction_steps:
                    # 创建新的特征数组
                    new_features = np.zeros_like(current_features_pos.cpu().numpy())
                    
                    # 使用预测结果更新特征
                    for j in range(history_window_size):
                        if j < history_window_size - steps:
                            # 保留旧特征
                            new_features[0, j] = current_features_pos[0, j+steps].cpu().numpy()
                        else:
                            # 添加新预测
                            idx = j - (history_window_size - steps)
                            
                            # 计算前车加速度 (简化为0或通过速度差分)
                            if i+idx > 0 and i+idx < prediction_steps:
                                lead_acc =a_lead[i+idx]
                            else:
                                lead_acc = 0  # 第一个时间步或超出范围时简化处理
                            if i+idx > 0 and i+idx < prediction_steps:
                                follow_acc =a_follow[i+idx]
                            else:
                                follow_acc = 0  # 第一个时间步或超出范围时简化处理
                            # 计算后车加速度
                            # follow_acc = direct_pred_a[i+idx]
                            
                            # 计算加速度差值
                            acc_diff = lead_acc - follow_acc
                            
                            # 更新特征 - 与loader2.py保持一致
                            lead_v_idx = min(i+idx, len(v_lead)-1)
                            new_features[0, j, 0] = v_lead[lead_v_idx]  # 前车速度
                            new_features[0, j, 1] = lead_acc  # 前车加速度
                            new_features[0, j, 2] = direct_pred_v[i+idx]  # 后车速度
                            new_features[0, j, 3] = follow_acc  # 后车加速度
                            new_features[0, j, 4] = direct_pred_gap[i+idx]  # 车间间隙
                            new_features[0, j, 5] = direct_pred_gap[i+idx] + vehicle_length  # 车头间距
                            new_features[0, j, 6] = v_lead[lead_v_idx] - direct_pred_v[i+idx]  # 速度差
                            
                            # 位置需要归一化处理 - 以当前窗口第一个位置为基准
                            # if j == history_window_size - steps:
                            #     # 更新归一化基准
                            #     base_pos = direct_pred_pos[i+idx]
                            
                            # 归一化后的位置
                            new_features[0, j, 7] = direct_pred_pos[i+idx] - base_pos  # 后车位置 (归一化)
                            new_features[0, j, 8] = pos_lead[i+idx] - base_pos  # 前车位置 (归一化，使用真实值)
                            new_features[0, j, 9] = acc_diff  # 加速度差值
                    base_pos = direct_pred[-1] + base_pos
                    # 更新特征
                    new_features[..., -3:-1] = new_features[..., -3:-1] - new_features[
                        ..., -1, -3]
                    current_features_pos = t.tensor(new_features, dtype=t.float32).to(device)

                    # 更新初始状态
                    initial_gap = direct_pred_gap[i+steps-1]
                    initial_v = direct_pred_v[i+steps-1]
            
            # 方法3: 基于间隙和速度的迭代预测
            # 准备初始输入特征
            current_features_dyn = hist_features.clone()
            
            # 获取初始位置和状态
            initial_gap = raw_data[history_window_size-1, 8]
            initial_v = raw_data[history_window_size-1, 6]
            initial_follow_pos = raw_data[history_window_size-1, 11]
            
            # 记录当前位置作为归一化基准
            base_pos = initial_follow_pos
            
            for i in range(0, prediction_steps, prediction_window_size):
                # 限制预测步数
                steps = min(prediction_window_size, prediction_steps - i)
                
                # 获取当前状态
                model_outputs = self.encoder(current_features_dyn)
                
                # 获取下一个前车速度序列
                end_idx = min(i+steps, len(v_lead))
                next_v_lead = t.tensor(v_lead[i:end_idx], dtype=t.float32).unsqueeze(0).to(device)

                # 预测
                predictions = self.predictor.forward(model_outputs, 
                                                next_v_lead,
                                                t.tensor([[initial_gap, initial_v]], dtype=t.float32).to(device))
                
                # 提取预测结果 - 间隙和速度预测
                dynamic_pred = predictions['dynamic_pred'].squeeze().cpu().numpy()
                
                # 存储动态预测结果
                for j in range(min(steps, len(dynamic_pred))):
                    if i+j < prediction_steps:

                        dynamic_pred_gap[i+j] = dynamic_pred[j, 0]
                        dynamic_pred_v[i+j] = dynamic_pred[j, 1]

                
                # 计算加速度 (速度差分)
                if i == 0:
                    dynamic_pred_a[i] = (dynamic_pred_v[i] - initial_v) / self.dt
                else:
                    dynamic_pred_a[i] = (dynamic_pred_v[i] - dynamic_pred_v[i-1]) / self.dt
                
                for j in range(1, min(steps, len(dynamic_pred))):
                    if i+j < prediction_steps:
                        dynamic_pred_a[i+j] = (dynamic_pred_v[i+j] - dynamic_pred_v[i+j-1]) / self.dt

                dynamic_pred_pos[i:i+min(steps, len(dynamic_pred))] = pos_lead[i:i+min(steps, len(dynamic_pred))] - dynamic_pred_gap[i:i+min(steps, len(dynamic_pred))] - vehicle_length

                
                # 准备下一次迭代的输入特征
                if i + steps < prediction_steps:
                    # 创建新的特征数组
                    new_features = np.zeros_like(current_features_dyn.cpu().numpy())
                    
                    # 使用预测结果更新特征
                    for j in range(history_window_size):
                        if j < history_window_size - steps:
                            # 保留旧特征
                            new_features[0, j] = current_features_dyn[0, j+steps].cpu().numpy()
                        else:
                            # 添加新预测
                            idx = j - (history_window_size - steps)

                            # 计算前车加速度 (简化为0或通过速度差分)
                            if i + idx > 0 and i + idx < prediction_steps:
                                lead_acc = a_lead[i + idx]
                            else:
                                lead_acc = 0  # 第一个时间步或超出范围时简化处理
                            if i+idx > 0 and i+idx < prediction_steps:
                                follow_acc =a_follow[i+idx]
                            else:
                                follow_acc = 0  # 第一个时间步或超出范围时简化处理
                            # 计算后车加速度
                            # follow_acc = dynamic_pred_a[i+idx]
                            
                            # 计算加速度差值
                            acc_diff = lead_acc - follow_acc
                            
                            # 更新特征 - 与loader2.py保持一致
                            new_features[0, j, 0] = v_lead[i+idx]  # 前车速度
                            new_features[0, j, 1] = lead_acc  # 前车加速度
                            new_features[0, j, 2] = dynamic_pred_v[i+idx]  # 后车速度
                            new_features[0, j, 3] = follow_acc  # 后车加速度
                            new_features[0, j, 4] = dynamic_pred_gap[i+idx]  # 车间间隙
                            new_features[0, j, 5] = dynamic_pred_gap[i+idx] + vehicle_length  # 车头间距
                            new_features[0, j, 6] = v_lead[i+idx+1] - dynamic_pred_v[i+idx]  # 速度差
                            
                            # 位置需要归一化处理 - 以当前窗口第一个位置为基准
                            if j == history_window_size - steps:
                                # 更新归一化基准
                                base_pos = 0
                            
                            # 归一化后的位置
                            new_features[0, j, 7] = dynamic_pred_pos[i+idx] - base_pos  # 后车位置 (归一化)
                            new_features[0, j, 8] = pos_lead[i+idx] - base_pos  # 前车位置 (归一化)
                            new_features[0, j, 9] = acc_diff  # 加速度差值
                    base_pos = pos_lead[i+steps-1] - dynamic_pred[-1, 0]
                    new_features[..., -3:-1] = new_features[..., -3:-1] - new_features[
                        ..., -1, -3]
                    # 更新特征
                    current_features_dyn = t.tensor(new_features, dtype=t.float32).to(device)

                    # 更新初始状态
                    initial_gap = dynamic_pred_gap[i+steps-1]
                    initial_v = dynamic_pred_v[i+steps-1]
        
        # # 计算RMSE
        # sim_one_step_rmse = np.sqrt(mean_squared_error(true_v_follow, sim_one_step_v))
        # direct_pred_rmse = np.sqrt(mean_squared_error(true_v_follow, direct_pred_v))
        # dynamic_pred_rmse = np.sqrt(mean_squared_error(true_v_follow, dynamic_pred_v))
        #
        # # 创建图形
        # fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        #
        # # 绘制真实值和预测值
        # ax.plot(time_steps, true_v_follow, self.colors['ground_truth'], linewidth=1.5, label='Ground Truth')
        # ax.plot(time_steps, v_lead[1:], self.colors['leading'], linewidth=1.5, label='Leading Vehicle')
        # ax.plot(time_steps, sim_one_step_v, 'g', linewidth=1.5,
        #         label=f'Simulation (Iteration) RMSE: {dynamic_pred_rmse:.3f} m/s')
        # ax.plot(time_steps, direct_pred_v, 'lightblue', linewidth=1.5,
        #         label=f'Position-based Prediction RMSE: {direct_pred_rmse:.3f} m/s', alpha=0.2)
        # # ax.plot(time_steps, dynamic_pred_v, 'g', linewidth=1.5,
        # #         label=f'Simulation (Iteration) RMSE: {dynamic_pred_rmse:.3f} m/s')
        #
        # # 计算速度图的y轴范围
        # v_min = min(np.min(true_v_follow), np.min(v_lead[1:]),
        #            np.min(sim_one_step_v))
        # v_max = max(np.max(true_v_follow), np.max(v_lead[1:]),
        #            np.max(sim_one_step_v))
        #
        # # 添加一些边距
        # v_margin = (v_max - v_min) * 0.1
        # ax.set_ylim(v_min - v_margin, v_max + v_margin)
        #
        # # 设置图表属性
        # ax.set_title(f'{type_name} Long-term Velocity Prediction', fontproperties=self.font_prop, fontsize=16)
        # ax.set_xlabel('Time (s)', fontproperties=self.font_prop, fontsize=14)
        # ax.set_ylabel('Velocity (m/s)', fontproperties=self.font_prop, fontsize=14)
        # ax.legend(prop=self.font_prop, loc='best')
        # ax.grid(True, linestyle='--', alpha=0.7)
        #
        # # 保存图像
        # plt.tight_layout()
        # plt.savefig(os.path.join(self.save_dir, f'{key}_velocity.png'), bbox_inches='tight')
        # plt.close()
        #
        # # 创建间隙图
        # fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        #
        # # 计算间隙RMSE
        # gap_sim_one_step_rmse = np.sqrt(mean_squared_error(true_gap, sim_one_step_gap))
        # gap_direct_pred_rmse = np.sqrt(mean_squared_error(true_gap, direct_pred_gap))
        # gap_dynamic_pred_rmse = np.sqrt(mean_squared_error(true_gap, dynamic_pred_gap))
        #
        # # 绘制间隙
        # ax.plot(time_steps, true_gap, self.colors['ground_truth'], linewidth=1.5, label='Ground Truth')
        # # ax.plot(time_steps, sim_one_step_gap, self.colors['sim_one_step'], linewidth=1.5,
        # #         label=f'Simulation (one step) RMSE: {gap_sim_one_step_rmse:.3f} m')
        # ax.plot(time_steps, direct_pred_gap, 'skyblue', linewidth=1.5,
        #         label=f'Prediction RMSE: {gap_direct_pred_rmse:.3f} m')
        # ax.plot(time_steps, dynamic_pred_gap, 'g', linewidth=1.5,
        #         label=f'Simulation RMSE: {gap_dynamic_pred_rmse:.3f} m')
        #
        # # 计算间隙图的y轴范围
        # gap_min = min(np.min(true_gap),
        #              np.min(dynamic_pred_gap))
        # gap_max = max(np.max(true_gap),
        #              np.max(dynamic_pred_gap))
        #
        # # 添加一些边距
        # gap_margin = (gap_max - gap_min) * 0.1
        # ax.set_ylim(gap_min - gap_margin, gap_max + gap_margin)
        #
        # # 设置图表属性
        # ax.set_title(f'{type_name} Long-term Gap Prediction', fontproperties=self.font_prop, fontsize=16)
        # ax.set_xlabel('Time (s)', fontproperties=self.font_prop, fontsize=14)
        # ax.set_ylabel('Gap (m)', fontproperties=self.font_prop, fontsize=14)
        # ax.legend(prop=self.font_prop, loc='best')
        # ax.grid(True, linestyle='--', alpha=0.7)
        #
        # # 保存图像
        # plt.tight_layout()
        # plt.savefig(os.path.join(self.save_dir, f'{key}_gap.png'), bbox_inches='tight')
        # plt.close()
        #
        # 创建位置图
        # 创建位置图
        # 创建放大区域函数
        def create_zoom_region(region_id, zoom_time_range, position, arrow_style='top_to_bottom',
                              lines=[1, 2, 3, 4], marker_size=3, line_width=0.5, offset=[0, 0]):
            """
            创建放大区域
            
            参数:
            region_id: 区域ID (1, 2, 3)
            zoom_time_range: 时间范围 (start_time, end_time)
            position: 放大框位置 [left, bottom, width, height]
            arrow_style: 箭头样式 ('bottom_to_top', 'right_to_left', 'left_to_right')
            lines: 要显示的线列表 [1, 2, 3, 4] (1:前车, 2:真实, 3:预测, 4:模拟)
            marker_size: 标记点大小
            line_width: 线宽
            """
            # 获取颜色
            zoom_color = self.zoom_colors[region_id]
            
            # 找到对应的索引
            zoom_start_time, zoom_end_time = zoom_time_range
            
            # 找到最接近这些时间的索引
            if zoom_start_time < 2:
                # 如果是历史数据区域
                zoom_start_idx = np.where(hist_time_steps >= zoom_start_time)[0][0]
                zoom_end_idx = np.where(hist_time_steps <= zoom_end_time)[0][-1]
                time_steps_to_use = hist_time_steps
                
                # 准备数据
                data_to_plot = {
                    'lead': hist_pos_lead[zoom_start_idx:zoom_end_idx+1] if 1 in lines else None,
                    'follow': hist_pos_follow[zoom_start_idx:zoom_end_idx+1] if 2 in lines else None,
                    'ground_truth': None,
                    'prediction': None,
                    'simulation': None
                }
            else:
                # 如果是预测数据区域
                zoom_start_idx = np.where(future_time_steps >= zoom_start_time)[0][0]
                zoom_end_idx = np.where(future_time_steps <= zoom_end_time)[0][-1]
                time_steps_to_use = future_time_steps
                
                # 准备数据
                data_to_plot = {
                    'lead': (pos_lead - base_pos)[zoom_start_idx:zoom_end_idx+1] if 1 in lines else None,
                    'follow': None,
                    'ground_truth': true_pos_follow[zoom_start_idx:zoom_end_idx+1] if 2 in lines else None,
                    'prediction': (direct_pred_pos - base_pos)[zoom_start_idx:zoom_end_idx+1] if 3 in lines else None,
                    'simulation': (dynamic_pred_pos - base_pos)[zoom_start_idx:zoom_end_idx+1] if 4 in lines else None
                }
            
            # 创建放大子图
            axins = fig.add_axes(position)  # [left, bottom, width, height]
            
            # 设置放大区域的背景色
            axins.patch.set_facecolor(zoom_color)
            axins.patch.set_alpha(0.15)  # 设置透明度
            
            # 在放大区域绘制轨迹
            if data_to_plot['lead'] is not None:
                axins.plot(time_steps_to_use[zoom_start_idx:zoom_end_idx+1], 
                        data_to_plot['lead'], 
                        color=self.colors['leading'], linewidth=line_width)
            
            if data_to_plot['follow'] is not None:
                axins.plot(time_steps_to_use[zoom_start_idx:zoom_end_idx+1], 
                        data_to_plot['follow'], 
                        color=self.colors['history'], linewidth=line_width)
            
            if data_to_plot['ground_truth'] is not None:
                axins.plot(time_steps_to_use[zoom_start_idx:zoom_end_idx+1], 
                        data_to_plot['ground_truth'], 
                        color=self.colors['ground_truth'], linewidth=line_width, 
                        marker='o', markersize=marker_size, markerfacecolor='none', 
                        markeredgecolor=self.colors['ground_truth'])
            
            if data_to_plot['prediction'] is not None:
                axins.plot(time_steps_to_use[zoom_start_idx:zoom_end_idx+1], 
                        data_to_plot['prediction'], 
                        color=self.colors['prediction'], linewidth=line_width, 
                        marker='s', markersize=marker_size, markerfacecolor='none', 
                        markeredgecolor=self.colors['prediction'])
            
            if data_to_plot['simulation'] is not None:
                axins.plot(time_steps_to_use[zoom_start_idx:zoom_end_idx+1], 
                        data_to_plot['simulation'], 
                        color=self.colors['sim_iteration'], linewidth=line_width, 
                        marker='^', markersize=marker_size, markerfacecolor='none', 
                        markeredgecolor=self.colors['sim_iteration'])
            
            # 设置放大区域的坐标轴
            axins.set_xlim(time_steps_to_use[zoom_start_idx], time_steps_to_use[zoom_end_idx])
            
            # 自动设置y轴范围，稍微扩大一点以便更好地显示
            y_values = []
            for data in data_to_plot.values():
                if data is not None:
                    y_values.append(data)
            
            if y_values:
                y_min = min([np.min(y) for y in y_values]) * 0.99
                y_max = max([np.max(y) for y in y_values]) * 1.01
                
                axins.set_ylim(y_min, y_max)
            else:
                # 如果没有数据，设置默认范围
                y_min = 0
                y_max = 100
            
            # 添加网格
            axins.grid(True, alpha=0.2)
            
            # 设置坐标轴刻度字体大小
            axins.tick_params(axis='both', which='major', labelsize=14)
            
            # 设置放大框边框
            for spine in axins.spines.values():
                spine.set_edgecolor(zoom_color)
                spine.set_linewidth(1)
            
            # 在主图上标记放大区域
            # 获取放大区域在主图上的位置
            x1 = time_steps_to_use[zoom_start_idx]
            x2 = time_steps_to_use[zoom_end_idx]
            y1 = y_min
            y2 = y_max
            
            # 绘制放大区域的标记框
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=True, 
                                edgecolor=zoom_color, facecolor=zoom_color, 
                                alpha=0.2, linestyle='-', linewidth=1.5)
            ax.add_patch(rect)
            
            # 连接主图中的标记框和放大图
            # 获取放大图的位置
            bbox = axins.get_position()
            axins_x1, axins_y1 = bbox.x0, bbox.y0  # 左下角
            axins_x2, axins_y2 = bbox.x1, bbox.y1  # 右上角
            
            # 将数据坐标转换为图表坐标
            # 修复坐标转换偏移问题：使用直接的组合变换
            
            # 使用组合变换：数据坐标 -> 图形坐标
            transform = ax.transData + fig.transFigure.inverted()
            
            # 直接计算矩形框的关键点
            center_x_data = (x1 + x2) / 2
            bottom_center_data = (center_x_data, y1)
            top_center_data = (center_x_data, y2)
            left_center_data = (x1, (y1 + y2) / 2)
            right_center_data = (x2, (y1 + y2) / 2)
            
            # 转换关键点到图形坐标
            bottom_center_fig = transform.transform(bottom_center_data)
            top_center_fig = transform.transform(top_center_data)
            left_center_fig = transform.transform(left_center_data)
            right_center_fig = transform.transform(right_center_data)
            
            # 获取主图中标记框的四个角在数据坐标系中的位置（用于其他计算）
            data_coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]  # 左下、右下、右上、左上
            fig_coords = [transform.transform(dc) for dc in data_coords]
            
            # 根据箭头样式设置连接点
            from matplotlib.patches import FancyArrowPatch
            
            if arrow_style == 'bottom_to_top':
                # 主框底部中点到放大框顶部中点
                main_point_x, main_point_y = bottom_center_fig
                
                zoom_point_x = axins_x1 + (axins_x2 - axins_x1) / 2
                zoom_point_y = axins_y2  # 顶部y坐标
                
            elif arrow_style == 'right_to_left':
                # 主框右边中点到放大框左边中点
                main_point_x, main_point_y = right_center_fig
                
                zoom_point_x = axins_x1  # 左边x坐标
                zoom_point_y = (axins_y1 + axins_y2) / 2  # 左边中点y坐标
                
            elif arrow_style == 'left_to_right':
                # 主框左边中点到放大框右边中点
                main_point_x, main_point_y = left_center_fig
                
                zoom_point_x = axins_x2  # 右边x坐标
                zoom_point_y = (axins_y1 + axins_y2) / 2  # 右边中点y坐标
            
            else:  # 默认 top_to_bottom
                # 使用精确计算的底部中心点
                main_point_x, main_point_y = top_center_fig
                
                zoom_point_x = axins_x1 + (axins_x2 - axins_x1) / 2
                zoom_point_y = axins_y1  
            
            # 获取偏置参数
            offset_x, offset_y = offset[0], offset[1]
            
            # 创建箭头连接，应用偏置
            arrow = FancyArrowPatch(
                (main_point_x + offset_x, main_point_y + offset_y),
                (zoom_point_x, zoom_point_y),
                transform=fig.transFigure,
                color=zoom_color,
                linewidth=1.2,
                arrowstyle='-|>',  # 箭头样式
                mutation_scale=15,  # 箭头大小
                connectionstyle="arc3,rad=0.1"  # 稍微弯曲的连接线
            )
            fig.patches.append(arrow)
            
            return y_min, y_max
        
        
        # 创建包含两个子图的图形：上面是位置图，下面是速度图
        fig, (ax_pos, ax_vel) = plt.subplots(2, 1, figsize=(15, 10), dpi=600)
        
        # 位置图在上面
        ax = ax_pos
        
        base_pos = raw_data[0, 11]
        # 获取真实后车位置
        true_pos_follow = raw_data[history_window_size:history_window_size+prediction_steps, 11] - base_pos
        # 获取历史数据
        hist_pos_follow = raw_data[:history_window_size, 11] - base_pos  # 历史后车位置
        hist_pos_lead = raw_data[:history_window_size, 12] - base_pos   # 历史前车位置
        
        # 设置不同类型的y轴最大值
        y_max_values = {
            0: 1800,
            1: 700,
            2: 1400,
            3: 500
        }
        
        # 创建完整时间轴（包括历史和预测）
        full_time_steps = np.arange(self.dt, (history_window_size + prediction_steps+1) * self.dt, self.dt)
        hist_time_steps = full_time_steps[:history_window_size]
        future_time_steps = full_time_steps[history_window_size:history_window_size+prediction_steps]
        
        # 添加背景色：0-2s区域用灰色填充
        ax.axvspan(0, 2, alpha=0.2, color='lightgray')
        
        # 计算位置RMSE
        pos_direct_pred_rmse = np.sqrt(mean_squared_error(true_pos_follow, direct_pred_pos - base_pos))
        pos_dynamic_pred_rmse = np.sqrt(mean_squared_error(true_pos_follow, dynamic_pred_pos - base_pos))
        
        # 绘制历史轨迹
        ax.plot(hist_time_steps, hist_pos_lead, color=self.colors['leading'], linewidth=1.5, label='Preceding Vehicle')
        ax.plot(hist_time_steps, hist_pos_follow, color=self.colors['history'], linewidth=1.5, label='Following Vehicle (History)')
        
        # 绘制位置
        ax.plot(future_time_steps, true_pos_follow, color=self.colors['ground_truth'], linewidth=1.5, label='Following Vehicle (Ground Truth)')
        ax.plot(future_time_steps, pos_lead - base_pos, color=self.colors['leading'], linewidth=1.5)
        # 根据轨迹类型和时间条件设置直接预测位置线的透明度
        # if traj_type == 1:  # AV-AV类型
        #     # 找到33秒的索引
        #     time_33s_idx = np.where(future_time_steps >= 33.5)[0]
        #     if len(time_33s_idx) > 0:
        #         # 分两段绘制：33秒前正常透明度，33秒后透明度0.2
        #         split_idx = time_33s_idx[0]
        #         # 33秒前的部分
        #         ax.plot(future_time_steps[:split_idx], (direct_pred_pos - base_pos)[:split_idx],
        #                 color=self.colors['prediction'], linewidth=1.5,
        #                 label=f'Prediction RMSE: {pos_direct_pred_rmse:.3f} m')
        #         # 33秒后的部分（透明度0.2）
        #         ax.plot(future_time_steps[split_idx-1:], (direct_pred_pos - base_pos)[split_idx-1:],
        #                 color=self.colors['prediction'], linewidth=1.5, alpha=0.2)
        if traj_type == 2:  # AV-AV类型
            # 找到33秒的索引
            time_33s_idx = np.where(future_time_steps >= 22)[0]
            if len(time_33s_idx) > 0:
                # 分两段绘制：33秒前正常透明度，33秒后透明度0.2
                split_idx = time_33s_idx[0]
                # 33秒前的部分
                ax.plot(future_time_steps[:split_idx], (direct_pred_pos - base_pos)[:split_idx],
                        color=self.colors['prediction'], linewidth=1.5,
                        label=f'Prediction RMSE: {pos_direct_pred_rmse:.3f} m')
                # 33秒后的部分（透明度0.2）
                ax.plot(future_time_steps[split_idx-1:], (direct_pred_pos - base_pos)[split_idx-1:],
                        color=self.colors['prediction'], linewidth=1.5, alpha=0.2)
        elif traj_type == 3:  # AV-AV类型
            # 找到33秒的索引
            time_33s_idx = np.where(future_time_steps >= 31)[0]
            if len(time_33s_idx) > 0:
                # 分两段绘制：33秒前正常透明度，33秒后透明度0.2
                split_idx = time_33s_idx[0]
                # 33秒前的部分
                ax.plot(future_time_steps[:split_idx], (direct_pred_pos - base_pos)[:split_idx],
                        color=self.colors['prediction'], linewidth=1.5,
                        label=f'Prediction RMSE: {pos_direct_pred_rmse:.3f} m')
                # 33秒后的部分（透明度0.2）
                ax.plot(future_time_steps[split_idx-1:], (direct_pred_pos - base_pos)[split_idx-1:],
                        color=self.colors['prediction'], linewidth=1.5, alpha=0.2)

        else:
            # 其他类型正常绘制
            ax.plot(future_time_steps, direct_pred_pos - base_pos, color=self.colors['prediction'], linewidth=1.5,
                    label=f'Prediction RMSE: {pos_direct_pred_rmse:.3f} m')
        ax.plot(future_time_steps, dynamic_pred_pos - base_pos, color=self.colors['sim_iteration'], linewidth=1.5,
                label=f'Simulation RMSE: {pos_dynamic_pred_rmse:.3f} m')
        
        # 在2s处添加圆形标记
        ax.plot(2, hist_pos_follow[-1], 'o', color=self.colors['history'], markersize=3)
        ax.plot(2, hist_pos_lead[-1], 'o', color=self.colors['leading'], markersize=3)
        
        # 基于仿真值、前后车真值确定位置图的纵坐标范围
        all_position_values = np.concatenate([
            hist_pos_follow, hist_pos_lead, true_pos_follow, 
            pos_lead - base_pos, dynamic_pred_pos - base_pos
        ])
        pos_y_min = max(0, np.min(all_position_values) * 0.9)  # 最小值不低于0
        pos_y_max = np.max(all_position_values) * 1.05  # 稍微扩大上限
        
        # 设置坐标轴范围
        ax.set_xlim(0, 50)
        ax.set_ylim(pos_y_min, pos_y_max)
        
        # 在图例中添加点标记
        # 创建自定义图例句柄
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=self.colors['leading'], lw=1.5, label='Preceding Vehicle'),
            Line2D([0], [0], color=self.colors['history'], lw=1.5, label='Following Vehicle (History)'),
            Line2D([0], [0], color=self.colors['ground_truth'], lw=1.5, marker='o', 
                   markerfacecolor='none', markeredgecolor=self.colors['ground_truth'], 
                   label='Following Vehicle (Ground Truth)'),
            Line2D([0], [0], color=self.colors['prediction'], lw=1.5, marker='s', 
                   markerfacecolor='none', markeredgecolor=self.colors['prediction'], 
                   label=f'Predicted Value (RMSE: {pos_direct_pred_rmse:.3f} m)'),
            Line2D([0], [0], color=self.colors['sim_iteration'], lw=1.5, marker='^', 
                   markerfacecolor='none', markeredgecolor=self.colors['sim_iteration'], 
                   label=f'Simulated Value (RMSE: {pos_dynamic_pred_rmse:.3f} m)')
        ]
        
        # 使用自定义图例
        ax.legend(handles=legend_elements, prop=self.font_prop, loc='upper left')
        
        # 设置图表属性
        # ax.set_title(f'{type_name}', fontproperties=self.font_prop, fontsize=16)
        ax.set_xlabel('Time (s)', fontproperties=self.font_prop, fontsize=18)
        ax.set_ylabel('Position (m)', fontproperties=self.font_prop, fontsize=18)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 设置坐标轴刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # 暂时注释掉位置图放大区域绘制功能
        # 创建放大区域 (2s到4s)
        # # 找到对应的索引
        # zoom_start_time = 3.49
        # zoom_end_time = 4.5
        # 
        # # 找到最接近这些时间的索引
        # zoom_start_idx = np.where(future_time_steps >= zoom_start_time)[0][0]
        # zoom_end_idx = np.where(future_time_steps <= zoom_end_time)[0][-1]
        # 
        # # 创建放大子图
        # axins = fig.add_axes([0.12, 0.4, 0.15, 0.25])  # 右下角位置 [left, bottom, width, height]
        # 
        # # 设置放大区域的背景色为浅粉色
        # axins.patch.set_facecolor("lightblue")
        # axins.patch.set_alpha(0.15)  # 设置透明度
        # 
        # # 在放大区域绘制轨迹
        # axins.plot(future_time_steps[zoom_start_idx:zoom_end_idx+1], 
        #         (pos_lead - base_pos)[zoom_start_idx:zoom_end_idx+1], 
        #         color=self.colors['leading'], linewidth=1)
        # 
        # axins.plot(future_time_steps[zoom_start_idx:zoom_end_idx+1], 
        #         true_pos_follow[zoom_start_idx:zoom_end_idx+1], 
        #         color=self.colors['ground_truth'], linewidth=1,
        #         marker='o', markersize=3, markerfacecolor='none', markeredgecolor=self.colors['ground_truth'])
        # 
        # axins.plot(future_time_steps[zoom_start_idx:zoom_end_idx+1], 
        #         (direct_pred_pos - base_pos)[zoom_start_idx:zoom_end_idx+1], 
        #         color=self.colors['prediction'], linewidth=1,
        #         marker='s', markersize=3, markerfacecolor='none', markeredgecolor=self.colors['prediction'], alpha=0.7)
        # 
        # axins.plot(future_time_steps[zoom_start_idx:zoom_end_idx+1], 
        #         (dynamic_pred_pos - base_pos)[zoom_start_idx:zoom_end_idx+1], 
        #         color=self.colors['sim_iteration'], linewidth=1,
        #         marker='^', markersize=3, markerfacecolor='none', markeredgecolor=self.colors['sim_iteration'], alpha=0.6)
        # 
        # # 设置放大区域的坐标轴
        # axins.set_xlim(future_time_steps[zoom_start_idx], future_time_steps[zoom_end_idx])
        # 
        # # 自动设置y轴范围，稍微扩大一点以便更好地显示
        # y_values = [
        #     true_pos_follow[zoom_start_idx:zoom_end_idx+1],
        #     (pos_lead - base_pos)[zoom_start_idx:zoom_end_idx+1],
        #     (direct_pred_pos - base_pos)[zoom_start_idx:zoom_end_idx+1],
        #     (dynamic_pred_pos - base_pos)[zoom_start_idx:zoom_end_idx+1]
        # ]
        # 
        # y_min = min([np.min(y) for y in y_values]) * 0.99
        # y_max = max([np.max(y) for y in y_values]) * 1.01
        # 
        # axins.set_ylim(y_min, y_max)
        # 
        # # 添加网格
        # axins.grid(True, alpha=0.2)
        # 
        # # 设置放大框边框为浅粉色
        # for spine in axins.spines.values():
        #     spine.set_edgecolor("lightblue")
        #     spine.set_linewidth(1)
        # 
        # # 在主图上标记放大区域
        # # 获取放大区域在主图上的位置
        # x1 = future_time_steps[zoom_start_idx]
        # x2 = future_time_steps[zoom_end_idx]
        # y1 = y_min
        # y2 = y_max
        # 
        # # 绘制放大区域的标记框
        # rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=True, 
        #                     edgecolor='lightblue', facecolor='lightblue',
        #                     alpha=0.2, linestyle='-', linewidth=1)
        # ax.add_patch(rect)
        # 
        # # 连接主图中的标记框和放大图
        # # 获取放大图的位置
        # bbox = axins.get_position()
        # axins_x1, axins_y1 = bbox.x0, bbox.y0  # 左下角
        # axins_x2, axins_y2 = bbox.x1, bbox.y1  # 右上角
        # 
        # # 将数据坐标转换为图表坐标
        # trans = ax.transData.transform
        # inv = fig.transFigure.inverted().transform
        # 
        # # 获取主图中标记框的四个角在图表坐标系中的位置
        # data_coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]  # 左下、右下、右上、左上
        # fig_coords = [inv(trans(dc)) for dc in data_coords]
        # 
        # # 计算主框底部中点和放大框顶部中点
        # main_bottom_center_x = fig_coords[0][0] - 0.04
        # main_bottom_center_y = fig_coords[2][1]  # 底部y坐标
        # 
        # zoom_top_center_x = (axins_x1 + axins_x2) / 2
        # zoom_top_center_y = axins_y1 - 0.04  # 顶部y坐标
        # 
        # # 使用浅粉色
        # light_pink = 'lightblue'
        # 
        # # 连接主框底部中点到放大框顶部中点，使用箭头
        # from matplotlib.patches import FancyArrowPatch
        # 
        # # 创建箭头连接
        # arrow = FancyArrowPatch(
        #     (main_bottom_center_x, main_bottom_center_y),
        #     (zoom_top_center_x, zoom_top_center_y),
        #     transform=fig.transFigure,
        #     color=light_pink,
        #     linewidth=1.2,
        #     arrowstyle='-|>',  # 箭头样式
        #     mutation_scale=15,  # 箭头大小
        #     connectionstyle="arc3,rad=0.1"  # 稍微弯曲的连接线
        # )
        # fig.patches.append(arrow)
        
        # 为当前轨迹类型创建所有放大区域
        # ==================== 绘制放大区域 ====================
        # 为当前轨迹类型绘制所有配置的放大区域
        # 每个放大区域包含指定时间范围内的详细轨迹对比
        for region_id, region_info in self.zoom_regions[traj_type].items():
            # 调整位置图放大区域的位置（向上移动以适应双子图布局）
            pos_position = region_info['position'].copy()
            pos_position[1] = pos_position[1] + 0.5  # 向上移动0.5个单位
            
            # 调用放大区域绘图函数
            # 参数说明：
            # - region_id: 放大区域编号
            # - region_info['time']: 放大的时间范围
            # - pos_position: 放大框在图中的位置
            # - arrow_style: 连接线箭头样式
            # - lines: 要显示的轨迹线类型
            # - marker_size: 标记点大小
            # - line_width: 线条宽度
            create_zoom_region(
                region_id, 
                region_info['time'], 
                pos_position,
                arrow_style=region_info.get('arrow_style', ''),
                lines=region_info.get('lines', [1, 2, 3, 4]),
                marker_size=region_info.get('marker_size', 3),
                line_width=region_info.get('line_width', 0.5),
                offset=region_info.get('offset', [0, 0])
            )



        # ==================== 速度图绘制 ====================
        # 获取真实速度数据
        true_v_follow = raw_data[history_window_size:history_window_size+prediction_steps, 6]
        hist_v_follow = raw_data[:history_window_size, 6]  # 历史后车速度
        hist_v_lead = raw_data[:history_window_size, 4]   # 历史前车速度
        
        # 计算速度RMSE
        vel_direct_pred_rmse = np.sqrt(mean_squared_error(true_v_follow, direct_pred_v))
        vel_dynamic_pred_rmse = np.sqrt(mean_squared_error(true_v_follow, dynamic_pred_v))
        
        # 基于真实值确定速度图的纵坐标范围
        all_true_velocities = np.concatenate([hist_v_follow, hist_v_lead, true_v_follow, v_lead, dynamic_pred_v])
        vel_y_min = max(0, np.min(all_true_velocities) * 0.92)  # 最小值不低于0
        vel_y_max = np.max(all_true_velocities) * 1.05  # 稍微扩大上限
        
        # 添加背景色：0-2s区域用灰色填充
        ax_vel.axvspan(0, 2, alpha=0.2, color='lightgray')
        
        # 绘制历史速度轨迹
        ax_vel.plot(hist_time_steps, hist_v_lead, color=self.colors['leading'], linewidth=1.5, label='Preceding Vehicle')
        ax_vel.plot(hist_time_steps, hist_v_follow, color=self.colors['history'], linewidth=1.5, label='Following Vehicle (History)')
        
        # 绘制预测速度
        ax_vel.plot(future_time_steps, v_lead[1:], color=self.colors['leading'], linewidth=1.5)
        ax_vel.plot(future_time_steps, true_v_follow, color=self.colors['ground_truth'], linewidth=1.5, label='Following Vehicle (Ground Truth)')
        ax_vel.plot(future_time_steps, direct_pred_v, color=self.colors['prediction'], linewidth=1.5, alpha=0.2,
                    label=f'Prediction RMSE: {vel_direct_pred_rmse:.3f} m/s')
        ax_vel.plot(future_time_steps, dynamic_pred_v, color=self.colors['sim_iteration'], linewidth=1.5,
                    label=f'Simulation RMSE: {vel_dynamic_pred_rmse:.3f} m/s')
        
        # 在2s处添加圆形标记
        ax_vel.plot(2, hist_v_follow[-1], 'o', color=self.colors['history'], markersize=3)
        ax_vel.plot(2, hist_v_lead[-1], 'o', color=self.colors['leading'], markersize=3)
        
        # 设置速度图坐标轴范围
        ax_vel.set_xlim(0, 50)
        ax_vel.set_ylim(vel_y_min, vel_y_max)
        
        # 创建速度图的自定义图例句柄（移除marker）
        vel_legend_elements = [
            Line2D([0], [0], color=self.colors['leading'], lw=1.5, label='Preceding Vehicle'),
            Line2D([0], [0], color=self.colors['history'], lw=1.5, label='Following Vehicle (History)'),
            Line2D([0], [0], color=self.colors['ground_truth'], lw=1.5, 
                   label='Following Vehicle (Ground Truth)'),
            Line2D([0], [0], color=self.colors['prediction'], lw=1.5, 
                   label=f'Predicted Value (RMSE: {vel_direct_pred_rmse:.3f} m/s)'),
            Line2D([0], [0], color=self.colors['sim_iteration'], lw=1.5, 
                   label=f'Simulated Value (RMSE: {vel_dynamic_pred_rmse:.3f} m/s)')
        ]
        
        # 使用自定义图例，AV-HV类型放在左上角，AV-AV类型放在左下角，其他类型放在右上角
        if traj_type == 0:  # AV-HV
            legend_loc = 'upper left'
        elif traj_type == 1:  # AV-AV
            legend_loc = 'lower left'
        else:  # HV-AV, HV-HV
            legend_loc = 'upper right'
        ax_vel.legend(handles=vel_legend_elements, prop=self.font_prop, loc=legend_loc)
        
        # 设置速度图表属性
        ax_vel.set_xlabel('Time (s)', fontproperties=self.font_prop, fontsize=18)
        ax_vel.set_ylabel('Velocity (m/s)', fontproperties=self.font_prop, fontsize=18)
        ax_vel.grid(True, linestyle='--', alpha=0.7)
        
        # 设置坐标轴刻度字体大小
        ax_vel.tick_params(axis='both', which='major', labelsize=16)
        
        # 为速度图创建放大区域函数
        def create_velocity_zoom_region(region_id, zoom_time_range, position, arrow_style='top_to_bottom',
                                      lines=[1, 2, 3, 4], marker_size=3, line_width=0.5):
            """
            为速度图创建放大区域
            """
            # 获取颜色
            zoom_color = self.zoom_colors[region_id]
            
            # 找到对应的索引
            zoom_start_time, zoom_end_time = zoom_time_range
            
            # 找到最接近这些时间的索引
            if zoom_start_time < 2:
                # 如果是历史数据区域
                zoom_start_idx = np.where(hist_time_steps >= zoom_start_time)[0][0]
                zoom_end_idx = np.where(hist_time_steps <= zoom_end_time)[0][-1]
                time_steps_to_use = hist_time_steps
                
                # 准备速度数据
                data_to_plot = {
                    'lead': hist_v_lead[zoom_start_idx:zoom_end_idx+1] if 1 in lines else None,
                    'follow': hist_v_follow[zoom_start_idx:zoom_end_idx+1] if 2 in lines else None,
                    'ground_truth': None,
                    'prediction': None,
                    'simulation': None
                }
            else:
                # 如果是预测数据区域
                zoom_start_idx = np.where(future_time_steps >= zoom_start_time)[0][0]
                zoom_end_idx = np.where(future_time_steps <= zoom_end_time)[0][-1]
                time_steps_to_use = future_time_steps
                
                # 准备速度数据
                data_to_plot = {
                    'lead': v_lead[zoom_start_idx:zoom_end_idx+1] if 1 in lines else None,
                    'follow': None,
                    'ground_truth': true_v_follow[zoom_start_idx:zoom_end_idx+1] if 2 in lines else None,
                    'prediction': direct_pred_v[zoom_start_idx:zoom_end_idx+1] if 3 in lines else None,
                    'simulation': dynamic_pred_v[zoom_start_idx:zoom_end_idx+1] if 4 in lines else None
                }
            
            # 创建放大子图
            axins_vel = fig.add_axes(position)  # [left, bottom, width, height]
            
            # 设置放大区域的背景色
            axins_vel.patch.set_facecolor(zoom_color)
            axins_vel.patch.set_alpha(0.15)  # 设置透明度
            
            # 在放大区域绘制速度轨迹
            if data_to_plot['lead'] is not None:
                axins_vel.plot(time_steps_to_use[zoom_start_idx:zoom_end_idx+1], 
                        data_to_plot['lead'], 
                        color=self.colors['leading'], linewidth=line_width)
            
            if data_to_plot['follow'] is not None:
                axins_vel.plot(time_steps_to_use[zoom_start_idx:zoom_end_idx+1], 
                        data_to_plot['follow'], 
                        color=self.colors['history'], linewidth=line_width)
            
            if data_to_plot['ground_truth'] is not None:
                axins_vel.plot(time_steps_to_use[zoom_start_idx:zoom_end_idx+1], 
                        data_to_plot['ground_truth'], 
                        color=self.colors['ground_truth'], linewidth=line_width, 
                        marker='o', markersize=marker_size, markerfacecolor='none', 
                        markeredgecolor=self.colors['ground_truth'])
            
            if data_to_plot['prediction'] is not None:
                axins_vel.plot(time_steps_to_use[zoom_start_idx:zoom_end_idx+1], 
                        data_to_plot['prediction'], 
                        color=self.colors['prediction'], linewidth=line_width, 
                        marker='s', markersize=marker_size, markerfacecolor='none', 
                        markeredgecolor=self.colors['prediction'])
            
            if data_to_plot['simulation'] is not None:
                axins_vel.plot(time_steps_to_use[zoom_start_idx:zoom_end_idx+1], 
                        data_to_plot['simulation'], 
                        color=self.colors['sim_iteration'], linewidth=line_width, 
                        marker='^', markersize=marker_size, markerfacecolor='none', 
                        markeredgecolor=self.colors['sim_iteration'])
            
            # 设置放大区域的坐标轴
            axins_vel.set_xlim(time_steps_to_use[zoom_start_idx], time_steps_to_use[zoom_end_idx])
            
            # 自动设置y轴范围，稍微扩大一点以便更好地显示
            y_values = []
            for data in data_to_plot.values():
                if data is not None:
                    y_values.append(data)
            
            if y_values:
                y_min = min([np.min(y) for y in y_values]) * 0.99
                y_max = max([np.max(y) for y in y_values]) * 1.01
                
                axins_vel.set_ylim(y_min, y_max)
            else:
                # 如果没有数据，设置默认范围
                y_min = 0
                y_max = 30
            
            # 添加网格
            axins_vel.grid(True, alpha=0.2)
            
            # 设置坐标轴刻度字体大小
            axins_vel.tick_params(axis='both', which='major', labelsize=14)
            
            # 设置放大框边框
            for spine in axins_vel.spines.values():
                spine.set_edgecolor(zoom_color)
                spine.set_linewidth(1)
            
            # 在速度图上标记放大区域
            # 获取放大区域在主图上的位置
            x1 = time_steps_to_use[zoom_start_idx]
            x2 = time_steps_to_use[zoom_end_idx]
            y1 = y_min
            y2 = y_max
            
            # 绘制放大区域的标记框
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=True, 
                                edgecolor=zoom_color, facecolor=zoom_color, 
                                alpha=0.2, linestyle='-', linewidth=1.5)
            ax_vel.add_patch(rect)
            
            # 连接主图中的标记框和放大图
            # 获取放大图的位置
            bbox = axins_vel.get_position()
            axins_x1, axins_y1 = bbox.x0, bbox.y0  # 左下角
            axins_x2, axins_y2 = bbox.x1, bbox.y1  # 右上角
            
            # 将数据坐标转换为图表坐标
            trans = ax_vel.transData.transform
            inv = fig.transFigure.inverted().transform
            
            # 获取主图中标记框的四个角在图表坐标系中的位置
            data_coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]  # 左下、右下、右上、左上
            fig_coords = [inv(trans(dc)) for dc in data_coords]
            
            # 根据箭头样式设置连接点
            from matplotlib.patches import FancyArrowPatch
            
            if arrow_style == 'bottom_to_top':
                # 主框底部中点到放大框顶部中点
                main_point_x = (fig_coords[0][0] + fig_coords[1][0]) / 2
                main_point_y = fig_coords[0][1]  # 底部y坐标
                
                zoom_point_x = (axins_x1 + axins_x2) / 2
                zoom_point_y = axins_y2  # 顶部y坐标
                
            elif arrow_style == 'right_to_left':
                # 主框右边中点到放大框左边中点
                main_point_x = fig_coords[1][0]  # 右边x坐标
                main_point_y = (fig_coords[1][1] + fig_coords[3][1]) / 2  # 右边中点y坐标
                
                zoom_point_x = axins_x1  # 左边x坐标
                zoom_point_y = (axins_y1 + axins_y2) / 2  # 左边中点y坐标
                
            elif arrow_style == 'left_to_right':
                # 主框左边中点到放大框右边中点
                main_point_x = fig_coords[0][0]+0.04  # 左边x坐标
                main_point_y =  fig_coords[2][1]  # 左边中点y坐标
                
                zoom_point_x = axins_x2  # 右边x坐标
                zoom_point_y = (axins_y1 + axins_y2) / 2  # 右边中点y坐标
            
            else:  # 默认 bottom_to_top
                main_point_x = (fig_coords[0][0] + fig_coords[1][0]) / 2
                main_point_y = fig_coords[3][1]
                
                zoom_point_x = (axins_x1 + axins_x2) / 2
                zoom_point_y = axins_y1
            
            # 创建箭头连接
            arrow = FancyArrowPatch(
                (main_point_x, main_point_y),
                (zoom_point_x, zoom_point_y),
                transform=fig.transFigure,
                color=zoom_color,
                linewidth=1.2,
                arrowstyle='-|>',  # 箭头样式
                mutation_scale=15,  # 箭头大小
                connectionstyle="arc3,rad=0.1"  # 稍微弯曲的连接线
            )
            fig.patches.append(arrow)
            
            return y_min, y_max
        
        # 暂时注释掉速度图放大区域绘制功能
        # 为速度图创建放大区域 (3.49s到4.5s)
        # zoom_start_time = 3.49
        # zoom_end_time = 4.5
        # 
        # # 找到最接近这些时间的索引
        # zoom_start_idx = np.where(future_time_steps >= zoom_start_time)[0][0]
        # zoom_end_idx = np.where(future_time_steps <= zoom_end_time)[0][-1]
        # 
        # # 创建速度图的放大子图
        # axins_vel = fig.add_axes([0.12, 0.15, 0.15, 0.15])  # 调整位置以适应速度图
        # 
        # # 设置放大区域的背景色为浅蓝色
        # axins_vel.patch.set_facecolor("lightblue")
        # axins_vel.patch.set_alpha(0.15)  # 设置透明度
        # 
        # # 在放大区域绘制速度轨迹
        # axins_vel.plot(future_time_steps[zoom_start_idx:zoom_end_idx+1], 
        #         v_lead[zoom_start_idx:zoom_end_idx+1], 
        #         color=self.colors['leading'], linewidth=1)
        # 
        # axins_vel.plot(future_time_steps[zoom_start_idx:zoom_end_idx+1], 
        #         true_v_follow[zoom_start_idx:zoom_end_idx+1], 
        #         color=self.colors['ground_truth'], linewidth=1,
        #         marker='o', markersize=3, markerfacecolor='none', markeredgecolor=self.colors['ground_truth'])
        # 
        # axins_vel.plot(future_time_steps[zoom_start_idx:zoom_end_idx+1], 
        #         direct_pred_v[zoom_start_idx:zoom_end_idx+1], 
        #         color=self.colors['prediction'], linewidth=1,
        #         marker='s', markersize=3, markerfacecolor='none', markeredgecolor=self.colors['prediction'], alpha=0.7)
        # 
        # axins_vel.plot(future_time_steps[zoom_start_idx:zoom_end_idx+1], 
        #         dynamic_pred_v[zoom_start_idx:zoom_end_idx+1], 
        #         color=self.colors['sim_iteration'], linewidth=1,
        #         marker='^', markersize=3, markerfacecolor='none', markeredgecolor=self.colors['sim_iteration'], alpha=0.6)
        # 
        # # 设置速度放大区域的坐标轴
        # axins_vel.set_xlim(future_time_steps[zoom_start_idx], future_time_steps[zoom_end_idx])
        # 
        # # 自动设置y轴范围，稍微扩大一点以便更好地显示
        # vel_y_values = [
        #     true_v_follow[zoom_start_idx:zoom_end_idx+1],
        #     v_lead[zoom_start_idx:zoom_end_idx+1],
        #     direct_pred_v[zoom_start_idx:zoom_end_idx+1],
        #     dynamic_pred_v[zoom_start_idx:zoom_end_idx+1]
        # ]
        # 
        # vel_y_min = min([np.min(y) for y in vel_y_values]) * 0.99
        # vel_y_max = max([np.max(y) for y in vel_y_values]) * 1.01
        # 
        # axins_vel.set_ylim(vel_y_min, vel_y_max)
        # 
        # # 添加网格
        # axins_vel.grid(True, alpha=0.2)
        # 
        # # 设置放大框边框为浅蓝色
        # for spine in axins_vel.spines.values():
        #     spine.set_edgecolor("lightblue")
        #     spine.set_linewidth(1)
        # 
        # # 在速度图上标记放大区域
        # # 获取放大区域在主图上的位置
        # vel_x1 = future_time_steps[zoom_start_idx]
        # vel_x2 = future_time_steps[zoom_end_idx]
        # vel_y1 = vel_y_min
        # vel_y2 = vel_y_max
        # 
        # # 绘制放大区域的标记框
        # vel_rect = plt.Rectangle((vel_x1, vel_y1), vel_x2-vel_x1, vel_y2-vel_y1, fill=True, 
        #                     edgecolor='lightblue', facecolor='lightblue',
        #                     alpha=0.2, linestyle='-', linewidth=1)
        # ax_vel.add_patch(vel_rect)
        # 
        # # 连接速度图中的标记框和放大图
        # # 获取放大图的位置
        # vel_bbox = axins_vel.get_position()
        # vel_axins_x1, vel_axins_y1 = vel_bbox.x0, vel_bbox.y0  # 左下角
        # vel_axins_x2, vel_axins_y2 = vel_bbox.x1, vel_bbox.y1  # 右上角
        # 
        # # 将数据坐标转换为图表坐标
        # vel_trans = ax_vel.transData.transform
        # vel_inv = fig.transFigure.inverted().transform
        # 
        # # 获取主图中标记框的四个角在图表坐标系中的位置
        # vel_data_coords = [(vel_x1, vel_y1), (vel_x2, vel_y1), (vel_x2, vel_y2), (vel_x1, vel_y2)]  # 左下、右下、右上、左上
        # vel_fig_coords = [vel_inv(vel_trans(dc)) for dc in vel_data_coords]
        # 
        # # 计算主框底部中点和放大框顶部中点
        # vel_main_bottom_center_x = vel_fig_coords[0][0] - 0.04
        # vel_main_bottom_center_y = vel_fig_coords[2][1]  # 底部y坐标
        # 
        # vel_zoom_top_center_x = (vel_axins_x1 + vel_axins_x2) / 2
        # vel_zoom_top_center_y = vel_axins_y1 - 0.04  # 顶部y坐标
        # 
        # # 创建箭头连接
        # vel_arrow = FancyArrowPatch(
        #     (vel_main_bottom_center_x, vel_main_bottom_center_y),
        #     (vel_zoom_top_center_x, vel_zoom_top_center_y),
        #     transform=fig.transFigure,
        #     color='lightblue',
        #     linewidth=1.2,
        #     arrowstyle='-|>',  # 箭头样式
        #     mutation_scale=15,  # 箭头大小
        #     connectionstyle="arc3,rad=0.1"  # 稍微弯曲的连接线
        # )
        # fig.patches.append(vel_arrow)
        
        # ==================== 绘制速度图放大区域 ====================
        # 为当前轨迹类型的速度图创建所有配置的放大区域
        # 每个放大区域显示指定时间范围内的速度变化细节
        # 暂时注释掉速度图放大区域绘制功能
        # for region_id, region_info in self.zoom_regions[traj_type].items():
        #     # 调整速度图放大区域的位置（向下移动）
        #     vel_position = region_info['position'].copy()
        #     vel_position[1] = vel_position[1] - 0.5  # 向下移动0.5个单位
        #     
        #     # 调用速度图放大区域绘图函数
        #     # 参数说明：
        #     # - region_id: 放大区域编号
        #     # - region_info['time']: 放大的时间范围
        #     # - vel_position: 放大框在速度图中的位置
        #     # - arrow_style: 连接线箭头样式
        #     # - lines: 要显示的轨迹线类型
        #     # - marker_size: 标记点大小
        #     # - line_width: 线条宽度
        #     create_velocity_zoom_region(
        #         region_id, 
        #         region_info['time'], 
        #         vel_position,
        #         arrow_style=region_info.get('arrow_style', ''),
        #         lines=region_info.get('lines', [1, 2, 3, 4]),
        #         marker_size=region_info.get('marker_size', 3),
        #         line_width=region_info.get('line_width', 0.5)
        #     )
        
        # ==================== 保存图像到对应类型文件夹 ====================
        # 根据轨迹类型创建对应的子文件夹
        type_name = self.type_labels[traj_type]
        type_save_dir = os.path.join(self.save_dir)
        if not os.path.exists(type_save_dir):
            os.makedirs(type_save_dir)
        
        # 保存图像到对应类型的文件夹中
        plt.tight_layout()
        plt.savefig(os.path.join(type_save_dir, f'{type_name}.png'), bbox_inches='tight')
        plt.close()
        
        return {
            'velocity': {
                'true': true_v_follow,
                'lead': v_lead,
                'direct_pred': direct_pred_v,
                'dynamic_pred': dynamic_pred_v,
                'rmse': {
                    'direct_pred': vel_direct_pred_rmse,
                    'dynamic_pred': vel_dynamic_pred_rmse
                }
            },
            'position': {
                'true': true_pos_follow,
                'lead': pos_lead,
                'direct_pred': direct_pred_pos,
                'dynamic_pred': dynamic_pred_pos,
                'rmse': {
                    'direct_pred': pos_direct_pred_rmse,
                    'dynamic_pred': pos_dynamic_pred_rmse
                }
            }
        }
    def evaluate_trajectories(self, trajectories, num_trajectories=10):
        """评估多条轨迹的长期预测性能"""
        results = {i: [] for i in range(4)}
        
        # 对每种类型评估num_trajectories条轨迹
        for traj_type in range(4):
            # 获取该类型的所有轨迹
            type_trajectories = [(k, v) for k, v in trajectories.items() if v['type'] == traj_type]
            
            # 选择前num_trajectories条轨迹
            selected_trajectories = type_trajectories[:num_trajectories]
            
            print(f"评估类型{traj_type}的{len(selected_trajectories)}条轨迹...")
            
            for key, trajectory in tqdm(selected_trajectories):
                # 长时间预测
                result = self.predict_long_term(trajectory, key)
                if result:
                    results[traj_type].append((key, result))
        
        return results

    def run_evaluation(self, num_trajectories=30):
        """评估多条轨迹"""
        # 加载轨迹数据
        print("Loading trajectory data...")
        trajectory_file = "trajectory_samples/extracted_specific_trajectories.npy"
        if not os.path.exists(trajectory_file):
            print(f"Error: File not found {trajectory_file}")
            return
        
        trajectories = np.load(trajectory_file, allow_pickle=True).item()
        
        # 为每种类型创建子目录
        type_dirs = {}
        for i in range(4):
            type_name = {0: "AV-HV", 1: "AV-AV", 2: "HV-HV", 3: "HV-AV"}[i]
            type_dir = os.path.join(self.save_dir, type_name)
            if not os.path.exists(type_dir):
                os.makedirs(type_dir)
            type_dirs[i] = type_dir
        results = self.evaluate_trajectories(trajectories, num_trajectories)
        # # 存储评估结果
        # results = {i: [] for i in range(4)}
        
        # # 对每种类型评估num_trajectories条轨迹
        # for traj_type in range(4):
        #     # 获取该类型的所有轨迹
        #     type_trajectories = [(k, v) for k, v in trajectories.items() if v['type'] == traj_type]
            
        #     # 选择前num_trajectories条轨迹
        #     selected_trajectories = type_trajectories[:num_trajectories]
            
        #     print(f"Evaluating {len(selected_trajectories)} trajectories of type {traj_type}...")
            
        #     for key, trajectory in tqdm(selected_trajectories):
        #         # 长时间预测
        #         result = self.predict_long_term(trajectory, key)
        #         if result:
        #             results[traj_type].append((key, result))
        
        # # 保存评估结果
        np.save(os.path.join(self.save_dir, "evaluation_results_2.npy"), results)
        
        # # 打印统计信息
        # print("\nEvaluation complete!")
        # for traj_type in range(4):
        #     type_name = {0: "AV-HV", 1: "AV-AV", 2: "HV-HV", 3: "HV-AV"}[traj_type]
        #     if results[traj_type]:
        #         # 计算平均RMSE
        #         v_sim_one_step_rmse = np.mean([r[1]['velocity']['rmse']['sim_one_step'] for r in results[traj_type]])
        #         v_sim_iteration_rmse = np.mean([r[1]['velocity']['rmse']['sim_iteration'] for r in results[traj_type]])
        #         v_prediction_rmse = np.mean([r[1]['velocity']['rmse']['prediction'] for r in results[traj_type]])
        #
        #         gap_sim_one_step_rmse = np.mean([r[1]['gap']['rmse']['sim_one_step'] for r in results[traj_type]])
        #         gap_sim_iteration_rmse = np.mean([r[1]['gap']['rmse']['sim_iteration'] for r in results[traj_type]])
        #         gap_prediction_rmse = np.mean([r[1]['gap']['rmse']['prediction'] for r in results[traj_type]])
        #
        #         print(f"\nType {type_name} Average RMSE:")
        #         print(f"  Velocity:")
        #         print(f"    Simulation (one step): {v_sim_one_step_rmse:.3f} m/s")
        #         print(f"    Simulation (iteration): {v_sim_iteration_rmse:.3f} m/s")
        #         print(f"    Prediction: {v_prediction_rmse:.3f} m/s")
        #
        #         print(f"  Gap:")
        #         print(f"    Simulation (one step): {gap_sim_one_step_rmse:.3f} m")
        #         print(f"    Simulation (iteration): {gap_sim_iteration_rmse:.3f} m")
        #         print(f"    Prediction: {gap_prediction_rmse:.3f} m")

if __name__ == "__main__":
    predictor = LongTermPredictor(model_epoch='21')
    predictor.run_evaluation(num_trajectories=100)
