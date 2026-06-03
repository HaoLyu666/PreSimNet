from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


DATA_COLUMNS = {
    0: "dataset_id",
    1: "trajectory_id",
    2: "timestamp",
    3: "interaction_type",
    4: "lead_vehicle_velocity",
    5: "lead_vehicle_acceleration",
    6: "following_vehicle_velocity",
    7: "following_vehicle_acceleration",
    8: "gap",
    9: "headway",
    10: "relative_velocity",
    11: "following_vehicle_position",
    12: "lead_vehicle_position",
}


class HighSimDataset(Dataset):
    def __init__(self, npy_file, hist_length=20, fut_length=20):
        self.npy_file = npy_file
        self.window_size = hist_length + fut_length
        self.hist_length = hist_length
        self.data = np.load(npy_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window = self.data[idx]
        hist_acc_diff = window[:, 5] - window[:, 7]
        hist = window[: self.hist_length, 4:]
        hist = np.column_stack((hist, hist_acc_diff[: self.hist_length]))

        next_v = window[self.hist_length - 1 :, 4]
        fut = window[self.hist_length :, [8, 6, 11, 12]]

        cf_type = np.zeros(4, dtype=np.int32)
        cf_type[int(window[0, 3])] = 1

        return (
            torch.tensor(hist, dtype=torch.float32),
            torch.tensor(next_v, dtype=torch.float32),
            torch.tensor(fut, dtype=torch.float32),
            torch.tensor(cf_type, dtype=torch.float32),
        )
