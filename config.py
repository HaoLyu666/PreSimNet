import os
import random

import numpy as np
import torch as t


seed = 72
random.seed(30)
np.random.seed(seed)
t.manual_seed(seed)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
learning_rate = 0.0005
dataset = "highsim"

setting = "presimnet_default"

args = {
    "train_flag": False,
    "gamma": 0.9,
    "num_worker": 0,
    "device": device,
    "encoder_size": 64,
    "n_head": 4,
    "in_length": 20,
    "out_length": 20,
    "para_length": 1,
    "f_length": 10,
    "batch_size": 512,
    "dropout": 0.1,
    "relu": 0.1,
    "epoch": 21,
    "transformer_layer": 1,
    "out_dim": 2,
    "num_task": 2,
    "ksv_flag": False,
    "num_mc": 0,
    "veh_num": 6,
    "time_step": 0.1,
    "query_var_flag": True,
    "np": 20,
    "sim_step": 5550,
    "cf_type": 4,
    "last_epoch": 0,
    "prediction_horizon": 480,
    "prediction_step": 20,
    "model_epoch": "21",
    "dt": 0.1,
    "confidence_level": 0.95,
    "path": os.path.join("checkpoints", setting),
    "f_path": os.path.join("outputs", "figures"),
    "l_path": os.path.join("results", setting),
    "long_term_save_dir": os.path.join("outputs", "long_term_predictions"),
}

for directory_key in ("path", "f_path", "l_path", "long_term_save_dir"):
    os.makedirs(args[directory_key], exist_ok=True)
