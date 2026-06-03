import random
import numpy as np
import torch as t
import os
import random


# args = {'path': 'checkponint/true_new_64/', 'f_path': 'fig/small_10_seed_30/64_10//0.2_0/100/'}
# args = {'f_path': 'fig/small_10_seed_30/64_10//0.2_0/100/'}
args = {'f_path': 'fig/gen/vis/'}

# ---------------------------------------------------------------------------
# 参数设置
seed = 72
random.seed(30)
np.random.seed(seed)
t.manual_seed(seed)
t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
learning_rate = 0.0005
dataset = "highsim"  # highd ngsim
args['train_flag'] = False
args['gamma'] = 0.7  # 学习率衰减
args['num_worker'] = 0
args['device'] = device
args['encoder_size'] = 64
args['n_head'] = 4
args['in_length'] = 20
args['out_length'] = 20
args['para_length'] = 1
args['f_length'] = 10
args['batch_size'] = 512
args['dropout'] = 0.1
args['relu'] = 0.1
args['epoch'] = 21
args['transformer_layer'] = 1
args['out_dim'] = 2
args['num_task'] = 2
args["ksv_flag"] = False
args["num_mc"] = 0
args['veh_num'] = 6
args['time_step'] = 0.1
args['query_var_flag'] = True
args['np'] = 20
args['sim_step'] = 5550
args['cf_type'] = 4
args['last_epoch'] = 0

# 长期预测相关参数
args['prediction_horizon'] = 480 # 长期预测的时间步数（48秒）
args['prediction_step'] = 20      # 每次预测的步长
args['model_epoch'] = '21'        # 使用的模型epoch
args['dt'] = 0.1                  # 时间步长
args['confidence_level'] = 0.95   # 置信区间
args['long_term_save_dir'] = 'long_term_predictions_2'  # 长期预测结果保存目录
# -------------------------------------------------------------------------

# gru_new_2-param-hist
setting = 'ed{}_inl{}_ol{}_drop{}_tl{}_nh{}_od{}_gama{}_qv{}_nt{}_gru_new-onlysim_task/'.format(
            args['encoder_size'],
            args['in_length'],
            args['out_length'],
            args['dropout'],
            args['transformer_layer'],
            args['n_head'],
            args['out_dim'],
            args['gamma'],
            int(args['query_var_flag']), args['num_task'])
c_path = os.path.join("checkponint", setting)
l_path = os.path.join("result", setting)

args['path'] = c_path
args['l_path'] = l_path
if not os.path.exists(args['path']):
    os.makedirs(args['path'])
if not os.path.exists(args['f_path']):
    os.makedirs(args['f_path'])
if not os.path.exists(args['l_path']):
    os.makedirs(args['l_path'])
if not os.path.exists(args['long_term_save_dir']):
    os.makedirs(args['long_term_save_dir'])