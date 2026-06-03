import math

import torch
import torch as t
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from module.module import BERTTimeEmbedding, MLP
from module.mamba import Mamba, MambaConfig
from torch import Tensor
from typing import Optional


class WeightedAggregation(nn.Module):
    """针对单查询优化的加权聚合机制

    比标准注意力更简单高效，特别适合query_len=1的场景
    """

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.dim = dim

        # 学习权重生成器
        self.weight_generator = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 4, 1)
        )

        # 值变换
        self.value_transform = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)

    def forward(self, query, key_value, mask=None):
        """
        Args:
            query: [B, 1, dim] - 单个查询
            key_value: [B, seq_len, dim]
            mask: [B, seq_len]
        """
        B, seq_len, dim = key_value.shape

        # 基于查询生成每个位置的权重
        query_expanded = query.expand(-1, seq_len, -1)  # [B, seq_len, dim]

        # 计算相似度权重
        similarity = F.cosine_similarity(query_expanded, key_value, dim=-1)  # [B, seq_len]

        # 学习的权重调整
        learned_weights = self.weight_generator(key_value).squeeze(-1)  # [B, seq_len]

        # 组合权重
        combined_weights = similarity + learned_weights  # [B, seq_len]

        # 应用掩码
        if mask is not None:
            combined_weights = combined_weights.masked_fill(mask == 0, float('-inf'))

        # Softmax归一化
        attn_weights = F.softmax(combined_weights, dim=-1)  # [B, seq_len]

        # 值变换
        values = self.value_transform(key_value)  # [B, seq_len, dim]

        # 加权聚合
        output = t.sum(attn_weights.unsqueeze(-1) * values, dim=1, keepdim=True)  # [B, 1, dim]

        return self.output_proj(output)


class SwiGLU(nn.Module):
    """SwiGLU激活函数，支持dropout"""

    def __init__(self, dim_in, dim_out, bias=True, dropout=0.0):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_out, bias=bias)
        self.w2 = nn.Linear(dim_in, dim_out, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(F.silu(self.w1(x)) * self.w2(x))


class GLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GLU, self).__init__()
        # 线性变换权重
        self.linear = nn.Linear(input_dim, hidden_dim)
        # 门控机制的权重
        self.gate = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # 计算线性变换
        linear_output = self.linear(x)
        # 计算门控输出
        gate_output = t.sigmoid(self.gate(x))
        # 逐元素相乘
        return linear_output * gate_output


# 轨迹预测头 - 使用GRU替代Transformer
class TrajectoryPredictionHead(nn.Module):
    def __init__(self, args, encoder_size, in_length, out_length, cf_type):
        super().__init__()
        self.encoder_size = encoder_size
        self.in_length = in_length
        self.out_length = out_length
        # self.cf_type = cf_type  # Removed for ablation
        self.dropout = args['dropout']
        self.num_layers = args['transformer_layer']
        self.n_head = args['n_head']

        # 恒速恒加速模型的投影层
        self.cv_ca_proj = nn.Linear(2, encoder_size)

        # 特征融合层 - Removed fusion layer as cf_type is removed
        # self.fusion_layer = nn.Linear(encoder_size + cf_type, encoder_size)

        # 初始查询向量
        self.query_embed = nn.Parameter(t.randn(1, 1, encoder_size))
        nn.init.xavier_uniform_(self.query_embed)

        # GRU解码器
        self.gru = nn.GRU(
            input_size=encoder_size,
            hidden_size=encoder_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=encoder_size,
            num_heads=self.n_head,
            dropout=self.dropout,
            batch_first=True
        )

        # 注意力输出融合
        self.attn_combine = nn.Linear(encoder_size * 2, encoder_size)

        # 添加FFN层
        self.ffn = nn.Sequential(
            nn.Linear(encoder_size, encoder_size * 4),
            nn.SiLU(),
            nn.Linear(encoder_size * 4, encoder_size),
            nn.Dropout(self.dropout)
        )

        # 添加FFN后的层归一化
        self.ffn_layer_norm = nn.LayerNorm(encoder_size)

        # 输出投影层
        self.output_proj = nn.Linear(encoder_size, 1)

        # 归一化层
        self.layer_norm = nn.LayerNorm(encoder_size)

        # 位置编码
        self.pos_encoder = BERTTimeEmbedding(
            max_position_embeddings=out_length,
            embedding_dim=encoder_size
        )

    def forward(self, hist_enc, init_pos):
        # 1. 生成恒速恒加速模型的初始预测并嵌入
        cv_ca_features = self.cv_ca_proj(init_pos)  # [B, T, 2] -> [B, T, encoder_size]
        cv_ca_features = self.layer_norm(cv_ca_features)

        # 2. 添加位置编码

        pos_queries = self.pos_encoder(cv_ca_features)
        decoder_inputs = cv_ca_features + pos_queries

        initial_hidden = hist_enc[..., -1, :].unsqueeze(0).contiguous()

        # 3. 通过GRU一次性处理整个序列
        # gru_outputs: [B, out_length, hidden_size]
        gru_outputs, _ = self.gru(decoder_inputs, initial_hidden)

        # 4. 使用注意力机制一次性计算所有时间步
        # Query: gru_outputs, Key/Value: hist_enc
        # attn_outputs: [B, out_length, hidden_size]
        attn_outputs, _ = self.attention(gru_outputs, hist_enc, hist_enc)

        # 5. 融合与后续处理 (FFN等)
        # output: [B, out_length, hidden_size * 2]
        output = t.cat((gru_outputs, attn_outputs), dim=2)

        # output: [B, out_length, some_dim]
        output = self.attn_combine(output)
        output = F.silu(output)

        # FFN增强
        ffn_output = output + self.ffn(output)
        decoder_outputs = self.ffn_layer_norm(ffn_output)

        # 10. 投影到位置预测
        position_pred = self.output_proj(decoder_outputs)
        position_pred = F.relu(position_pred)  # 确保位置是正的

        return position_pred, decoder_outputs


# 参数预测头 - 简化为每个轨迹只预测一组参数
class ParameterPredictionHead(nn.Module):
    """改进的参数预测头：使用多头注意力机制，参考原版本设计并融入现代注意力改进"""

    def __init__(self, encoder_size, cf_type, dropout, n_head):
        super().__init__()
        self.encoder_size = encoder_size
        # self.cf_type = cf_type # Unused
        self.dropout = dropout
        self.n_head = n_head

        # 可学习的参数查询向量
        self.param_query = nn.Parameter(t.Tensor(1, 1, encoder_size))
        nn.init.xavier_uniform_(self.param_query, gain=1.414)

        # 历史编码的自注意力机制 - 使用更多头数提升表达能力
        self.hist_attention = nn.MultiheadAttention(
            embed_dim=encoder_size,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )

        # 历史注意力后的FFN - 使用SwiGLU with dropout
        self.hist_ffn = SwiGLU(encoder_size, encoder_size, dropout=dropout)

        # 轨迹编码的交叉注意力机制
        self.traj_attention = nn.MultiheadAttention(
            embed_dim=encoder_size,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )

        # 轨迹注意力后的FFN - 使用SwiGLU with dropout
        self.traj_ffn = SwiGLU(encoder_size, encoder_size, dropout=dropout)

        # 消融实验：只保留1个IDM专家
        self.idm_expert = MLP(f_in=encoder_size, f_out=6, activation='tanh',
                              hidden_dim=encoder_size * 2, hidden_layers=2, dropout=dropout)

        # 归一化层
        self.layer_norm1 = nn.LayerNorm(encoder_size)  # 输入归一化
        self.layer_norm2 = nn.LayerNorm(encoder_size)  # 历史注意力后归一化
        self.layer_norm3 = nn.LayerNorm(encoder_size)  # 历史FFN后归一化
        self.layer_norm4 = nn.LayerNorm(encoder_size)  # 轨迹注意力后归一化
        self.layer_norm5 = nn.LayerNorm(encoder_size)  # 轨迹FFN后归一化

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, hist_enc, traj_encoding=None):
        batch_size = hist_enc.shape[0]

        # 1. 准备参数查询向量
        param_query = self.param_query.repeat(batch_size, 1, 1)  # [B, 1, encoder_size]

        # 2. 历史编码的自注意力处理
        param_query_norm = self.layer_norm1(param_query)
        hist_enc_norm = self.layer_norm1(hist_enc)

        # 历史注意力
        hist_attn_output, _ = self.hist_attention(
            param_query_norm,
            hist_enc_norm,
            hist_enc_norm
        )
        param_query = param_query + self.dropout_layer(hist_attn_output)
        param_query = self.layer_norm2(param_query)

        # 历史注意力后的FFN
        hist_ffn_output = self.hist_ffn(param_query)
        param_query = param_query + hist_ffn_output
        param_query = self.layer_norm3(param_query)

        # 3. 如果有轨迹编码，使用交叉注意力机制增强参数查询
        if traj_encoding is not None:
            traj_encoding_norm = self.layer_norm1(traj_encoding)
            traj_attn_output, _ = self.traj_attention(
                param_query,
                traj_encoding_norm,
                traj_encoding_norm
            )
            param_query = param_query + self.dropout_layer(traj_attn_output)
            param_query = self.layer_norm4(param_query)

            # 轨迹注意力后的FFN
            traj_ffn_output = self.traj_ffn(param_query)
            param_query = param_query + traj_ffn_output
            param_query = self.layer_norm5(param_query)

        # 4. 生成参数 (只有一个专家)
        param = self.idm_expert(param_query)  # [B, 1, 6]
        idm_params = F.softplus(param)

        # 返回参数 (列表形式以保持接口兼容，但只有1个)
        return {
            'idm_params': [idm_params],
            'cf_probs': None
        }


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()  # 初始化参数
        self.device = args['device']
        self.encoder_size = args['encoder_size']
        self.n_head = args['n_head']
        self.cf_type = args['cf_type']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.f_length = args['f_length']

        self.train_flag = args['train_flag']
        self.batch_size = args['batch_size']

        self.dropout = args['dropout']
        self.transformer_layer = args['transformer_layer']
        self.num_mc = args["num_mc"]
        self.veh_num = args['veh_num']
        self.para_length = 1  # 简化为每个轨迹只预测一组参数

        self.encoder = nn.Linear(self.f_length, self.encoder_size)

        self.gelu = nn.GELU()
        self.mamba_Config = MambaConfig(d_model=self.encoder_size, n_layers=1, d_state=8)
        self.mamba = Mamba(self.mamba_Config)
        self.in_emb = BERTTimeEmbedding(max_position_embeddings=self.in_length, embedding_dim=self.encoder_size)

        # 消融：移除类型查询向量
        # self.type_query = nn.Parameter(t.Tensor(1, 1, self.encoder_size))
        # nn.init.xavier_uniform_(self.type_query, gain=1.414)

        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.encoder_size,
            nhead=self.n_head,
            dim_feedforward=self.encoder_size * 4,
            dropout=self.dropout,
            batch_first=True,
            activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(
            self.TransformerEncoderLayer, num_layers=self.transformer_layer)

        # 消融：移除类型预测头
        # self.type_pred = nn.Sequential(...)

        # 轨迹预测头和参数预测头
        self.trajectory_head = TrajectoryPredictionHead(
            args,
            encoder_size=self.encoder_size,
            in_length=self.in_length,
            out_length=self.out_length,
            cf_type=self.cf_type
        )

        self.parameter_head = ParameterPredictionHead(
            encoder_size=self.encoder_size,
            cf_type=self.cf_type, dropout=self.dropout, n_head=self.n_head
        )

    def generate_cv_ca_predictions(self, hist, dt=0.1):
        """生成恒速(CV)和恒加速度(CA)模型的初始预测"""
        batch_size = hist.shape[0]

        # 获取最后一个时间步的位置和速度
        last_pos = hist[:, -1, -3:-2]  # 位置
        last_vel = hist[:, -1, 2:3]  # 速度

        # 计算加速度 (使用最后两个时间步)
        if hist.shape[1] >= 2:
            prev_vel = hist[:, -2, 2:3]
            accel = (last_vel - prev_vel) / dt
        else:
            accel = t.zeros_like(last_vel)

        # 生成未来的时间步 t_future [out_length]
        t_future = (t.arange(1, self.out_length + 1, device=last_pos.device).float()) * dt  # [out_length]

        # 扩展 last_pos, last_vel 和 accel 到 [B, out_length]
        last_pos_expanded = last_pos.expand(-1, self.out_length)  # [B, out_length]
        last_vel_expanded = last_vel.expand(-1, self.out_length)  # [B, out_length]
        accel_expanded = accel.expand(-1, self.out_length)  # [B, out_length]

        # 恒速模型预测: x(t) = x0 + v0 * t
        cv_preds = last_pos_expanded + last_vel_expanded * t_future  # [B, out_length]

        # 恒加速度模型预测: x(t) = x0 + v0 * t + 0.5 * a * t^2
        ca_preds = last_pos_expanded + last_vel_expanded * t_future + 0.5 * accel_expanded * (
                t_future ** 2)  # [B, out_length]

        # 堆叠预测结果
        cv_trajectory = cv_preds.unsqueeze(2)  # [B, out_length, 1]
        ca_trajectory = ca_preds.unsqueeze(2)  # [B, out_length, 1]

        # 合并CV和CA预测
        init_predictions = t.cat([cv_trajectory, ca_trajectory], dim=-1)  # [B, out_length, 2]

        return init_predictions

    def forward(self, Hist):
        # 编码历史轨迹
        hist_enc = self.gelu(self.encoder(Hist))
        hist_mamba = self.mamba(hist_enc)

        # 消融：不添加类型查询，直接过Transformer
        # hist_att_in = t.cat([hist_mamba, self.type_query.repeat([hist_enc.shape[0], 1, 1])], dim=1)
        hist_att = self.transformer_encoder(hist_mamba)

        # 消融：移除类型预测
        # cf_type_pred = self.type_pred(hist_att[:, -1, :])
        # cf_type_probs = F.softmax(cf_type_pred, dim=-1)

        # 6. 应用加权映射到历史编码，作为记忆 (消融：直接使用历史编码)
        memory = hist_att  # [:, :-1, :]  # 注意：之前因为加了query所以length+1，现在没加，直接用

        # 生成CV和CA模型的初始预测
        init_predictions = self.generate_cv_ca_predictions(Hist)

        # 轨迹预测头 (移除cf_type_probs)
        position_pred, traj_encoding = self.trajectory_head(
            memory,
            init_predictions
        )

        # 参数预测头 (移除cf_type_pred)
        param_outputs = self.parameter_head(
            memory,
            # traj_encoding
        )

        return {
            'position_pred': position_pred,
            'cf_type_pred': None,  # 返回None
            'params': param_outputs,
            'traj_encoding': traj_encoding
        }


class predictor:
    def __init__(self, args):
        self.args = args
        self.epsilon = 1e-10
        self.device = args['device']
        self.train_flag = args['train_flag']
        self.out_dim = args['out_dim']
        self.dropout = args['dropout']
        self.batch_size = args['batch_size']
        self.out_length = args['out_length']
        self.in_length = args['in_length']
        self.dt = args['time_step']
        self.para_length = 1  # 简化为每个轨迹只预测一组参数
        self.cf_type = args['cf_type']

    def idm_model(self, params, veh_state, delta_V_t):
        """改进的IDM模型，使用可学习的加速度指数"""
        # 提取参数
        s0 = params[:, 0:1]  # 最小安全间距
        T = params[:, 1:2]  # 安全时距
        b = params[:, 2:3]  # 舒适减速度
        a_max = params[:, 3:4]  # 最大加速度
        v0 = params[:, 4:5]  # 期望速度
        delta = params[:, 5:6]  # 加速度指数 (可学习)

        # 当前车速和间隙
        V_t = veh_state[:, 1:2]
        H_t = veh_state[:, 0:1]

        # 计算期望间距
        s_star = V_t * T + (V_t * delta_V_t) / (2 * t.sqrt(a_max * b))
        s_star = s0 + t.maximum(s_star, t.tensor(0.0, device=s_star.device))
        # 计算加速度 (使用可学习的指数)
        a_t = a_max * (1 - (V_t / (v0 + self.epsilon)) ** delta - (s_star / (H_t + self.epsilon)) ** 2)
        a_t = t.clamp(a_t, min=-5, max=5)

        return a_t

    def veh_dynamic(self, veh_state, idm_params_list, next_v):
        """只使用1个IDM专家计算车辆动态"""
        # 当前车速和间隙
        V_t = veh_state[:, 1:2]
        H_t = veh_state[:, 0:1]

        # 计算速度差
        delta_V_t = next_v - V_t

        # 提取IDM参数 (只有一个)
        idm_params = idm_params_list[0].squeeze(1)  # [B, 6]

        # 计算IDM加速度
        a_t = self.idm_model(idm_params, veh_state, delta_V_t)  # [B, 1]

        # 更新速度和位置
        V_t_next = V_t + a_t * self.dt
        V_t_next = t.clamp(V_t_next, min=0)  # 确保速度非负

        H_t_next = H_t + delta_V_t * self.dt
        H_t_next = t.clamp(H_t_next, min=0)  # 确保间隙非负

        veh_state_next = t.cat([H_t_next, V_t_next], dim=-1)
        return veh_state_next

    def forward(self, model_outputs, nextv, veh_state):
        """
        使用模型输出进行预测，只使用IDM专家
        """
        # 提取模型输出
        position_pred = model_outputs['position_pred']  # [B, out_length, 1]
        params = model_outputs['params']
        # cf_probs = params['cf_probs']  # Unused
        # acc_params_list = params['acc_params']  # Unused
        idm_params_list = params['idm_params']  # 列表，包含IDM模型的参数 (1个)

        # 初始化车辆状态序列
        veh_states = [veh_state]  # 存储所有时间步的车辆状态

        # 逐步计算车辆状态
        for t_step in range(self.out_length):
            # 获取当前时间步的前车速度
            next_v_t = nextv[:, t_step:t_step + 1]

            # 计算下一个时间步的车辆状态
            next_state = self.veh_dynamic(
                veh_states[-1],
                idm_params_list,
                next_v_t
            )

            # 存储车辆状态
            veh_states.append(next_state)

        # 堆叠所有时间步的车辆状态 [B, out_length, 2]
        veh_states_tensor = t.stack(veh_states[1:], dim=1)

        # 构建完整的预测结果
        dynamic_positions = veh_states_tensor[:, :, 0]  # 间隙
        dynamic_velocities = veh_states_tensor[:, :, 1]  # 速度

        # 构建动态预测结果
        dynamic_pred = t.cat([
            dynamic_positions.unsqueeze(-1),
            dynamic_velocities.unsqueeze(-1)
        ], dim=-1)

        # 构建直接预测结果（只有位置）
        direct_pred = position_pred

        return {
            'dynamic_pred': dynamic_pred,  # 动态模型预测的间隙和速度
            'direct_pred': direct_pred,  # 直接预测的位置
            'veh_states': veh_states_tensor  # 所有时间步的车辆状态
        }


class MoEGRU(nn.Module):
    def __init__(self, args):
        super(MoEGRU, self).__init__()
        self.encoder = Encoder(args)

    def forward(self, hist):
        # 编码历史轨迹并预测未来轨迹和参数
        outputs = self.encoder(hist)

        return outputs