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
    """SwiGLU激活函数"""

    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_out, bias=bias)
        self.w2 = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x):
        return F.silu(self.w1(x)) * self.w2(x)

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
        self.cf_type = cf_type
        self.dropout = args['dropout']
        self.num_layers = args['transformer_layer']
        self.n_head = args['n_head']
        

        
        # 恒速恒加速模型的投影层
        self.cv_ca_proj = nn.Linear(2, encoder_size)
        
        # 特征融合层 - 直接融合跟驰类型概率向量
        self.fusion_layer = nn.Linear(encoder_size + cf_type, encoder_size)
        
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
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # 位置编码
        self.pos_encoder = BERTTimeEmbedding(
            max_position_embeddings=out_length, 
            embedding_dim=encoder_size
        )

    def forward_step(self, input_tensor, hidden, encoder_outputs):
        # 通过GRU单步解码
        output, hidden = self.gru(input_tensor, hidden)

        # 使用注意力机制关注编码器输出
        attn_output, _ = self.attention(output, encoder_outputs, encoder_outputs)
        
        # 融合注意力输出和GRU输出
        output = t.cat((output, attn_output), dim=2)
        output = self.attn_combine(output)
        output = F.silu(output)
        
        # 添加FFN层增强特征表达能力
        ffn_output = self.ffn(output)
        output = output + self.dropout_layer(ffn_output)
        output = self.ffn_layer_norm(output)
        
        return output, hidden

    def forward(self, hist_enc, cf_type_probs, init_pos):
        batch_size = hist_enc.shape[0]
        
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
    """优化后的参数预测头：使用加权聚合机制，保持训练脚本兼容性"""

    def __init__(self, encoder_size, cf_type, dropout, n_head):
        super().__init__()
        self.encoder_size = encoder_size
        self.cf_type = cf_type
        self.dropout = dropout
        self.n_head = n_head

        # 自适应查询生成器（保持一定复杂度）
        self.adaptive_query_generator = nn.Sequential(
            nn.Linear(encoder_size, encoder_size),
            nn.GELU(),
            nn.Linear(encoder_size, encoder_size)
        )

        # 参数查询
        self.param_query = nn.Parameter(t.randn(1, 1, encoder_size))
        nn.init.xavier_uniform_(self.param_query)

        # 使用加权聚合替代标准注意力（参数更少但保持性能）
        self.hist_attention = WeightedAggregation(encoder_size, dropout)
        self.traj_attention = WeightedAggregation(encoder_size, dropout)

        # 特征融合层（动态计算输入维度）
        # combined_features实际是128维(64+64)
        # 删除feature_fusion和traj_adapter，按用户要求不进行融合

        # FFN层（适度减少expansion factor）
        self.ffn = SwiGLU(encoder_size, encoder_size)  # 输出维度与输入维度相同

        # 保持all_experts属性以兼容训练脚本
        self.all_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_size, encoder_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(encoder_size, 6)  # 6个参数
            ) for _ in range(cf_type)
        ])

        # 归一化层
        self.norm1 = nn.LayerNorm(encoder_size)
        self.norm2 = nn.LayerNorm(encoder_size)
        self.norm3 = nn.LayerNorm(encoder_size)

    def forward(self, hist_enc, cf_type_pred, traj_encoding=None):
        batch_size = hist_enc.shape[0]

        # 自适应查询生成
        adaptive_query = self.adaptive_query_generator(hist_enc.mean(dim=1, keepdim=True))
        param_query = self.param_query.expand(batch_size, -1, -1) + adaptive_query
        param_query = self.norm1(param_query)

        # 按顺序先进行历史编码注意力
        hist_attn_output = self.hist_attention(param_query, hist_enc)
        param_query = param_query + hist_attn_output

        # 如果有轨迹编码，再进行预测编码注意力
        if traj_encoding is not None:
            traj_attn_output = self.traj_attention(param_query, traj_encoding)
            param_query = param_query + traj_attn_output

        param_query = self.norm2(param_query)

        # FFN处理
        ffn_output = self.ffn(param_query)
        param_query = param_query + ffn_output
        param_query = self.norm3(param_query)

        # 使用专家网络预测参数
        param_input = param_query.squeeze(1)  # [B, encoder_size]
        all_params = []

        for expert in self.all_experts:
            expert_output = expert(param_input)  # [B, 6]
            expert_output = F.softplus(expert_output)  # 确保参数为正
            all_params.append(expert_output.unsqueeze(1))  # [B, 1, 6]

        # 分离ACC和IDM参数
        acc_params = all_params[:2]  # 前两个专家用于ACC
        idm_params = all_params[2:]  # 后两个专家用于IDM

        # 计算类型概率
        cf_probs = F.softmax(cf_type_pred, dim=-1)
        
        # 返回所有专家的参数和概率
        return {
            'acc_params': acc_params,
            'idm_params': idm_params,
            'cf_probs': cf_probs
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
        self.type_query = nn.Parameter(t.Tensor(1, 1, self.encoder_size))
        nn.init.xavier_uniform_(self.type_query, gain=1.414)
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.encoder_size,
            nhead=self.n_head,
            dim_feedforward=self.encoder_size * 4, 
            dropout=self.dropout,
            batch_first=True,
        activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(
            self.TransformerEncoderLayer, num_layers=self.transformer_layer)
        self.type_pred = nn.Sequential(
            SwiGLU(self.encoder_size, self.encoder_size * 2),
            nn.Linear(self.encoder_size * 2, self.encoder_size),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder_size, self.cf_type)
        )
        # 映射历史轨迹编码到未来时间步，为每种跟驰类型创建不同的映射
        self.mapping = nn.Parameter(t.Tensor(self.cf_type, self.out_length, self.in_length))
        nn.init.xavier_uniform_(self.mapping, gain=1.414)
        # 轨迹预测头和参数预测头
        self.trajectory_head = TrajectoryPredictionHead(
            args,
            encoder_size=self.encoder_size,
            in_length=self.in_length,
            out_length=self.out_length,
            cf_type=self.cf_type
        )
        
        # self.parameter_head = ParameterPredictionHead(
        #     encoder_size=self.encoder_size,
        #     cf_type=self.cf_type, dropout=self.dropout, n_head=self.n_head
        # )


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
        
        # 添加类型查询并通过Transformer
        hist_att_in = t.cat([hist_mamba, self.type_query.repeat([hist_enc.shape[0], 1, 1])], dim=1)
        hist_att = self.transformer_encoder(hist_att_in)
        
        # 预测跟驰类型
        cf_type_pred = self.type_pred(hist_att[:, -1, :])  # 未经softmax的logits
        cf_type_probs = F.softmax(cf_type_pred, dim=-1)  # 转换为概率
        # 5. 根据跟驰类型概率加权映射矩阵
        weighted_mapping = t.einsum('bc,coi->boi', cf_type_probs, self.mapping)  # [B, out_length, in_length]
        weighted_mapping = F.softmax(weighted_mapping, dim=2)  # 在in_length维度上进行softmax

        # 6. 应用加权映射到历史编码，作为记忆
        memory = t.bmm(weighted_mapping, hist_att[:, :-1, :])  # [B, out_length, encoder_size]
        # 生成CV和CA模型的初始预测
        init_predictions = self.generate_cv_ca_predictions(Hist)
        
        # 轨迹预测头
        position_pred, traj_encoding = self.trajectory_head(
            memory,
            cf_type_probs,
            init_predictions
        )
        
        # # 参数预测头 (使用轨迹编码增强)
        # param_outputs = self.parameter_head(
        #     memory,
        #     cf_type_pred,
        #     traj_encoding
        # )
        
        return {
            'position_pred': position_pred,
            'cf_type_pred': cf_type_pred,
        }


class predictor(nn.Module):
    """预测器，用于生成最终的预测结果"""

    def __init__(self, args):
        super(predictor, self).__init__()
        self.args = args
        self.out_length = args['out_length']

    def forward(self, model_outputs, nextv, veh_state):
        """
        Args:
            model_outputs: 模型输出字典
            nextv: 前车未来速度 [B, out_length]
            veh_state: 车辆当前状态 [B, 2] (间隙, 速度)
        """
        position_pred = model_outputs['position_pred']  # [B, out_length, 1]

        # 基于位置预测计算其他动态量
        batch_size = position_pred.shape[0]

        # 初始化动态预测
        dynamic_pred = torch.zeros(batch_size, self.out_length, 2).to(position_pred.device)

        # 获取初始状态
        current_gap = veh_state[:, 0:1]  # [B, 1]
        current_velocity = veh_state[:, 1:2]  # [B, 1]

        for t in range(self.out_length):
            if t == 0:
                # 第一个时间步：使用当前状态
                predicted_gap = current_gap + position_pred[:, t, 0:1]
                predicted_velocity = current_velocity
            else:
                # 后续时间步：基于位置变化计算
                position_change = position_pred[:, t, 0:1] - position_pred[:, t - 1, 0:1]
                predicted_gap = dynamic_pred[:, t - 1, 0:1] + position_change
                # 简单的速度估计（可以改进）
                predicted_velocity = dynamic_pred[:, t - 1, 1:2] + position_change * 0.1

            dynamic_pred[:, t, 0] = predicted_gap.squeeze(-1)
            dynamic_pred[:, t, 1] = predicted_velocity.squeeze(-1)

        return {
            'dynamic_pred': dynamic_pred,  # [B, out_length, 2] (间隙, 速度)
            'direct_pred': position_pred  # [B, out_length, 1] (位置)
        }


class MoEGRU(nn.Module):
    def __init__(self, args):
        super(MoEGRU, self).__init__()
        self.encoder = Encoder(args)
        
    def forward(self, hist):
        # 编码历史轨迹并预测未来轨迹和参数
        outputs = self.encoder(hist)
        
        return outputs

class Seq2SeqBaseline(Encoder):
    """兼容性别名"""
    pass