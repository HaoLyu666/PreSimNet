import numpy as np
import os

def extract_specific_trajectories():
    """
    从selected_trajectories_2.npy中提取特定的轨迹
    提取traj_type 0-3中编号分别为8、50、79、65的轨迹
    """
    # 加载原始数据
    data_path = "trajectory_samples/selected_trajectories_2.npy"
    if not os.path.exists(data_path):
        print(f"错误：找不到文件 {data_path}")
        return
    
    print("加载原始轨迹数据...")
    all_trajectories = np.load(data_path, allow_pickle=True).item()
    
    # 定义要提取的轨迹
    target_trajectories = {
        'type_0_traj_8': 0,   # AV-HV类型第8条轨迹
        'type_1_traj_12': 1,  # AV-AV类型第50条轨迹
        'type_2_traj_79': 2,  # HV-AV类型第79条轨迹
        'type_3_traj_65': 3   # HV-HV类型第65条轨迹
    }
    
    # 提取指定的轨迹
    extracted_trajectories = {}
    
    print("提取指定轨迹...")
    for key, traj_type in target_trajectories.items():
        if key in all_trajectories:
            extracted_trajectories[key] = all_trajectories[key]
            print(f"成功提取: {key} (类型 {traj_type})")
            
            # 打印轨迹信息
            traj_info = all_trajectories[key]
            print(f"  - Dataset ID: {traj_info['dataset_id']}")
            print(f"  - Trajectory ID: {traj_info['trajectory_id']}")
            print(f"  - 轨迹长度: {traj_info['length']}")
            print(f"  - 窗口数量: {len(traj_info['windows'])}")
            print(f"  - 原始数据形状: {traj_info['raw_data'].shape}")
        else:
            print(f"警告：未找到轨迹 {key}")
    
    if not extracted_trajectories:
        print("错误：没有找到任何指定的轨迹")
        return
    
    # 保存提取的轨迹
    save_path = "trajectory_samples/extracted_specific_trajectories.npy"
    print(f"\n保存提取的轨迹到: {save_path}")
    np.save(save_path, extracted_trajectories)
    
    print(f"\n提取完成！")
    print(f"总共提取了 {len(extracted_trajectories)} 条轨迹:")
    for key in extracted_trajectories.keys():
        print(f"  - {key}")
    
    # 验证保存的文件
    print("\n验证保存的文件...")
    try:
        loaded_data = np.load(save_path, allow_pickle=True).item()
        print(f"验证成功：文件包含 {len(loaded_data)} 条轨迹")
        for key in loaded_data.keys():
            print(f"  - {key}: 窗口数量 {len(loaded_data[key]['windows'])}")
    except Exception as e:
        print(f"验证失败: {e}")

if __name__ == "__main__":
    extract_specific_trajectories()