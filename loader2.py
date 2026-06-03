from __future__ import print_function, division
from torch.utils.data import Dataset
import scipy.io as scp
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

'''
0 数据集编号
1 轨迹编号
2 时间戳
3 跟驰类型
4 前车速度
5 前车加速度
6 后车速度
7 后车加速度
8 车间间隙
9 车头间距
10 速度差
11 后车位置
12 前车位置
'''


class HighSimDataset(Dataset):
    def __init__(self, npy_file, hist_length=20, fut_length=20):
        """
        Args:
            npy_file (str): Path to the .npy file containing the data.
            window_size (int): Total length of each trajectory sample.
            hist_length (int): Length of historical data in each sample.
        """
        self.npy_file = npy_file
        self.window_size = hist_length + fut_length
        self.hist_length = hist_length
        self.data = np.load(npy_file)  # Use memory mapping to save memory



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window = self.data[idx]
        hist_acc_diff = window[:, 5] - window[:, 7]
        hist = window[:self.hist_length, 4:]
        hist = np.column_stack((hist, hist_acc_diff[:self.hist_length]))  # Add hist_acc_diff as a new column
        # Future data
        next_v = window[self.hist_length - 1:, 4]  # Column 4: 前车速度
        fut = window[self.hist_length:, [8, 6, 11, 12]]  # Columns 8 and 6: 车间间隙, 后车速度
        cf_type = np.zeros(4, dtype=np.int32)  # Using np.int32 to save memory
        cf_type[int(window[0, 3])] = 1  # One-hot encoding for cf_type

        return torch.tensor(hist, dtype=torch.float32), torch.tensor(next_v, dtype=torch.float32), torch.tensor(fut,
                                                                                                                dtype=torch.float32), torch.tensor(
            cf_type, dtype=torch.float32)


