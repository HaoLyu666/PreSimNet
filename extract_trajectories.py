import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

def check_trajectory_quality(trajectory):
    """
    检查整个轨迹的质量
    """
    # 将 DataFrame 转换为 NumPy 数组
    traj_values = trajectory.values
    
    # 检查各项条件
    col_5 = traj_values[:, 4]  # 速度条件
    col_7 = traj_values[:, 6]  # 加速度条件
    col_8 = traj_values[:, 7]  # 加加速度条件
    col_9 = traj_values[:, 8]  # 间距条件
    col_11 = np.abs(traj_values[:, 10])  # 速度差条件
    
    # 一步动力学校验
    con = np.abs(traj_values[:-1, 8] + traj_values[:-1, 10] * 0.1 - traj_values[1:, 8])
    
    # 计算各条件的违反比例
    ratio_col_5 = np.sum(col_5 < 1) / len(col_5)
    ratio_col_7 = np.sum(col_7 < 1) / len(col_7)
    ratio_col_8 = np.sum(np.abs(col_8) > 5) / len(col_8)
    ratio_col_9 = np.sum(col_9 < 2) / len(col_9)
    ratio_con = np.sum(con > 0.6) / len(con)
    ratio_col_11 = np.sum(col_11 > 10) / len(col_11)
    
    # 如果任何条件的违反比例超过1%，则认为轨迹质量不佳
    if (ratio_col_5 > 0.01 or ratio_col_7 > 0.01 or ratio_col_8 > 0.01 or 
        ratio_col_9 > 0.01 or ratio_con > 0.01 or ratio_col_11 > 0.01):
        return False
    
    return True

def process_group(group, window_size=40):
    """
    使用滑动窗口对轨迹数据进行切片
    """
    # 将分组按照时间索引排序
    group_sorted = group.sort_values(by='Time_Index')

    # 将 DataFrame 转换为 NumPy 数组
    group_sorted_values = group_sorted.values

    num_points = len(group_sorted_values)

    # 使用 NumPy 的切片操作来提取窗口
    data_samples = []
    # if int(group_sorted_values[0, 3]) == 2 or int(group_sorted_values[0, 3]) == 3:
    #     stride = 2
    # else:
    stride = 1
    for i in range(0, num_points - window_size + 1, stride):
        window = group_sorted_values[i:i + window_size]
        data_samples.append(window)

    return np.array(data_samples) if data_samples else np.array([])

def main():
    # 设置随机种子以确保结果可复现
    random.seed(42)
    np.random.seed(42)
    
    # 加载数据
    print("加载数据...")
    data_path = "D:\博\科研\数据\carfollowing4type/all_data_pos.csv"
    if not os.path.exists(data_path):
        print(f"错误：找不到文件 {data_path}")
        return
    
    merged_df = pd.read_csv(data_path)
    
    # 按照 'Dataset_ID' 和 'Trajectory_ID' 分组
    grouped = merged_df.groupby(['Dataset_ID', 'Trajectory_ID'])
    
    # 将分组转换为列表并打乱顺序
    print("将分组转换为列表并打乱顺序...")
    group_list = []
    for (dataset_id, trajectory_id), group in grouped:
        # 获取轨迹类型
        traj_type = group['Type'].iloc[0]
        # 检查轨迹长度是否大于300个数据点（30秒）
        if len(group) >= 500:
            group_list.append((dataset_id, trajectory_id, traj_type, group))
    
    # 打乱分组顺序
    random.shuffle(group_list)
    
    # 用于存储每种类型的轨迹
    trajectories_by_type = {0: [], 1: [], 2: [], 3: []}
    
    # 用于存储最终选择的轨迹
    selected_trajectories = {}
    
    print("筛选符合条件的轨迹...")
    # 遍历打乱后的分组列表
    for dataset_id, trajectory_id, traj_type, group in tqdm(group_list):
        # 如果该类型的轨迹已经收集了100条，则跳过
        if len(trajectories_by_type[traj_type]) >= 100:
            continue
        
        # 检查轨迹质量
        if not check_trajectory_quality(group):
            continue
        
        # 将符合条件的轨迹添加到对应类型的列表中
        trajectories_by_type[traj_type].append((dataset_id, trajectory_id, group))
    
    # 检查是否每种类型都有足够的轨迹
    for traj_type, trajectories in trajectories_by_type.items():
        if len(trajectories) < 5:
            print(f"警告：类型 {traj_type} 只找到 {len(trajectories)} 条符合条件的轨迹")
    
    print("处理选定的轨迹...")
    # 处理选定的轨迹
    for traj_type, trajectories in trajectories_by_type.items():
        for i, (dataset_id, trajectory_id, group) in enumerate(trajectories):
            # 使用滑动窗口处理轨迹
            windows = process_group(group, window_size=40)
            
            if len(windows) > 100:
                # 获取原始数据
                raw_data = group.sort_values(by='Time_Index').values.copy()
                
                # 对AV-HV类型的第8条轨迹进行特殊处理
                if traj_type == 0 and i == 7:  # AV-HV类型(0)的第8条轨迹(索引7)
                    print(f"对AV-HV类型第8条轨迹应用二阶差分修正...")
                    dt = 0.1  # 时间步长
                    start_correction_idx = 250  # 从第250个数据点开始修正
                    
                    if start_correction_idx < len(raw_data):
                        # 使用二阶差分方法修正前车位置
                        # 公式: pos[i] = pos[i-1] + v[i-1]*dt + 0.5*a[i-1]*dt^2
                        for idx in range(start_correction_idx, len(raw_data)):
                            if idx > 0:
                                # 获取前一时刻的速度和加速度
                                v_prev = raw_data[idx-1, 4]  # 前车速度(索引4)
                                a_prev = raw_data[idx-1, 5]  # 前车加速度(索引5)
                                
                                # 使用二阶差分公式计算新位置
                                raw_data[idx, 12] = (raw_data[idx-1, 12] + 
                                                    v_prev * dt + 
                                                    0.5 * a_prev * dt * dt)
                        raw_data=raw_data[20:]
                        print(f"位置修正已应用，从索引{start_correction_idx}到{len(raw_data)-1}")
                
                # 存储轨迹信息和窗口数据
                key = f"type_{traj_type}_traj_{i+1}"
                selected_trajectories[key] = {
                    'dataset_id': dataset_id,
                    'trajectory_id': trajectory_id,
                    'type': traj_type,
                    'length': len(group),
                    'windows': windows,
                    'raw_data': raw_data
                }
    
    # 创建保存目录
    save_dir = "trajectory_samples"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 保存选定的轨迹
    print("保存选定的轨迹...")
    np.save(os.path.join(save_dir, "selected_trajectories_2.npy"), selected_trajectories)
    
    # 打印统计信息
    print("\n轨迹选择完成！")
    print(f"总共选择了 {len(selected_trajectories)} 条轨迹")
    
    for traj_type in range(4):
        count = sum(1 for key in selected_trajectories if f"type_{traj_type}" in key)
        print(f"类型 {traj_type}: {count} 条轨迹")
    
    # 打印每条轨迹的窗口数量
    print("\n每条轨迹的窗口数量:")
    for key, traj in selected_trajectories.items():
        print(f"{key}: {len(traj['windows'])} 个窗口")

if __name__ == "__main__":
    main()