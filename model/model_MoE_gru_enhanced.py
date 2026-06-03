import torch as t
import torch.nn as nn
import torch.nn.functional as F
from module.module import BERTTimeEmbedding, MLP
from module.mamba import Mamba, MambaConfig
import math


class SwiGLU(nn.Module):
    """SwiGLU激活函数"""
    def __init__(self, dim_in, dim_out, bias=True):
        super().__init__()
        self.w1 = nn.Linear(dim_in, dim_out, bias=bias)
        self.w2 = nn.Linear(dim_in, dim_out, bias=bias)
        self.w3 = nn.Linear(dim_out, dim_out, bias=bias)
        
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class MLPMixer(nn.Module):
    """MLPMixer用于时空特征混合"""
    def __init__(self, seq_len, hidden_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Token mixing (时间维度混合)
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, seq_len),
            nn.Dropout(dropout)
        )
        
        # Channel mixing (特征维度混合)
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x: [B, seq_len, hidden_dim]
        # Token mixing
        residual = x
        x = self.token_mixing(x.transpose(1, 2)).transpose(1, 2)
        x = x + residual
        
        # Channel mixing
        residual = x
        x = self.channel_mixing(x)
        x = x + residual
        
        return x


class FlashAttention(nn.Module):
    """简化版Flash Attention实现"""
    def __init__(self, dim, num_heads=8, dropout=0.1, causal=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 使用PyTorch的scaled_dot_product_attention（如果可用）
        if hasattr(F, 'scaled_dot_product_attention'):
            x = F.scaled_dot_product_attention(
                q, k, v, 
                dropout_p=self.dropout.p if self.training else 0,
                is_causal=self.causal
            )
        else:
            # 回退到标准实现
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if self.causal:
                mask = t.triu(t.ones(N, N, device=x.device), diagonal=1).bool()
                attn.masked_fill_(mask, float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class StandardCrossAttention(nn.Module):
    """标准交叉注意力：Query来自decoder，Key和Value来自encoder"""
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, query, key_value, mask=None):
        """
        Args:
            query: [B, query_len, dim] - decoder输入
            key_value: [B, kv_len, dim] - encoder输入
            mask: 可选的注意力掩码
        """
        B, query_len, d_model = query.shape
        _, kv_len, _ = key_value.shape
        
        # 计算QKV
        q = self.q_proj(query)  # [B, query_len, dim]
        kv = self.kv_proj(key_value)  # [B, kv_len, dim*2]
        k, v = kv.chunk(2, dim=-1)  # [B, kv_len, dim], [B, kv_len, dim]
        
        # 重塑为多头格式
        q = q.reshape(B, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.dropout_layer(attn)
            attn_output = attn @ v
        
        # 重塑回原始形状
        attn_output = attn_output.transpose(1, 2).reshape(B, query_len, d_model)
        return self.out_proj(attn_output)


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


class SharedExpertNetwork(nn.Module):
    """共享编码器+分离头的专家网络设计
    
    大幅减少参数：4个专家共享前面的编码层，只在最后分化
    """
    def __init__(self, input_dim, hidden_dim, num_experts=4, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        
        # 共享编码器（所有专家共用）
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 专家特定的头部（每个专家一个）
        self.expert_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, 6)  # 6个参数
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        # 共享编码
        shared_features = self.shared_encoder(x)  # [B, hidden_dim // 2]
        
        # 各专家分别处理
        expert_outputs = []
        for head in self.expert_heads:
            output = head(shared_features)  # [B, 6]
            expert_outputs.append(output.unsqueeze(1))  # [B, 1, 6]
        
        return expert_outputs


class RMSNorm(nn.Module):
    """RMSNorm归一化"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(t.ones(dim))
        self.eps = eps
        
    def forward(self, x):
        norm = t.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.scale


class TrajectoryPredictionHead(nn.Module):
    """改进的轨迹预测头，结合GRU和注意力机制"""
    def __init__(self, args, encoder_size, in_length, out_length, cf_type):
        super().__init__()
        self.encoder_size = encoder_size
        self.in_length = in_length
        self.out_length = out_length
        self.cf_type = cf_type
        self.dropout = args['dropout']
        self.num_layers = args['transformer_layer']
        self.n_head = args['n_head']
        
        # 移除MLPMixer，简化特征处理
        
        # CV/CA特征处理 - 简化处理，直接使用原始预测值
        self.cv_ca_proj = nn.Linear(2, encoder_size // 4)
        
        # 跟驰类型概率的映射权重（保留原始设计）
        self.mapping = nn.Parameter(t.Tensor(cf_type, out_length, in_length))
        nn.init.xavier_uniform_(self.mapping, gain=1.414)
        
        # 特征融合层
        self.fusion_layer = nn.Linear(encoder_size + encoder_size // 4, encoder_size)
        
        # 初始查询向量
        self.query_embed = nn.Parameter(t.randn(1, 1, encoder_size))
        nn.init.xavier_uniform_(self.query_embed)
        
        # GRU解码器（保留原始设计）
        self.gru = nn.GRU(
            input_size=encoder_size,
            hidden_size=encoder_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # 自注意力（使用因果注意力）
        self.self_attention = FlashAttention(
            dim=encoder_size,
            num_heads=self.n_head,
            dropout=self.dropout,
            causal=True  # 自注意力使用因果性
        )
        
        # 交叉注意力（不使用因果注意力）
        self.cross_attention = StandardCrossAttention(
            dim=encoder_size,
            num_heads=self.n_head,
            dropout=self.dropout
        )
        
        # 轻量级FFN使用SwiGLU（减少expansion factor）
        self.ffn = nn.Sequential(
            SwiGLU(encoder_size, encoder_size * 2),  # 从4倍减少到2倍
            nn.Linear(encoder_size * 2, encoder_size),
            nn.Dropout(self.dropout)
        )
        self.self_attn_ffn =nn.Sequential(
            SwiGLU(encoder_size, encoder_size * 2),  # 从4倍减少到2倍
            nn.Linear(encoder_size * 2, encoder_size),
            nn.Dropout(self.dropout)
        )
        # 输出投影层
        self.output_proj = nn.Linear(encoder_size, 1)
        
        # 归一化层使用RMSNorm
        self.layer_norm = RMSNorm(encoder_size // 4)  # 用于cv_ca_features
        self.self_attn_norm = RMSNorm(encoder_size)  # 自注意力层归一化
        self.cross_attn_norm = RMSNorm(encoder_size)  # 交叉注意力层归一化
        self.ffn_layer_norm = RMSNorm(encoder_size)  # FFN层归一化
        self.self_attn_ffn_norm = RMSNorm(encoder_size)  # 自注意力FFN层归一化

        # 位置编码
        self.pos_encoder = BERTTimeEmbedding(
            max_position_embeddings=out_length, 
            embedding_dim=encoder_size // 4
        )

    # 移除decode_step方法，改为直接在forward中实现transformer decoder架构

    def forward(self, hist_enc, cf_type_probs, init_pos):
        batch_size = hist_enc.shape[0]
        
        # 1. 处理CV/CA特征
        cv_ca_features = self.cv_ca_proj(init_pos)  # [B, out_length, encoder_size//4]
        cv_ca_features = self.layer_norm(cv_ca_features)
        
        # 2. 根据跟驰类型概率加权映射矩阵（保留用于可视化）
        weighted_mapping = t.einsum('bc,coi->boi', cf_type_probs, self.mapping)
        weighted_mapping = F.softmax(weighted_mapping, dim=2)
        
        # 3. 应用加权映射到历史编码
        memory = t.bmm(weighted_mapping, hist_enc)  # [B, out_length, encoder_size]
        
        # 4. 合并memory和cv_ca_features
        combined_features = t.cat([memory, cv_ca_features], dim=-1)  # [B, out_length, encoder_size + encoder_size//4]
        decoder_input = self.fusion_layer(combined_features)  # [B, out_length, encoder_size]
        
        # 5. 添加位置编码（在融合后进行，更符合transformer架构）
        pos_queries = self.pos_encoder(decoder_input[:, :, :self.encoder_size//4])  # 使用部分特征计算位置编码
        decoder_input = decoder_input + t.cat([pos_queries, t.zeros_like(decoder_input[:, :, self.encoder_size//4:])], dim=-1)
        
        # 6. 通过GRU处理整个序列（发挥GRU处理序列的优势）
        gru_output, _ = self.gru(decoder_input)  # [B, out_length, encoder_size]
        
        # 7. Transformer Decoder架构
        # 7.1 自注意力子层（带因果掩码）
        residual = gru_output
        gru_output = self.self_attn_norm(gru_output)  # Pre-norm
        self_attn_output = self.self_attention(gru_output)  # [B, out_length, encoder_size]
        gru_output = residual + self_attn_output  # 残差连接
        
        # 自注意力后的FFN（可选，用于增强局部特征）
        residual = gru_output
        gru_output = self.self_attn_ffn_norm(gru_output)
        self_attn_ffn_output = self.self_attn_ffn(gru_output)
        gru_output = residual + self_attn_ffn_output

        # 7.2 交叉注意力子层（标准实现，避免额外计算成本）
        residual = gru_output
        gru_output = self.cross_attn_norm(gru_output)  # Pre-norm
        
        # 标准交叉注意力：Query来自decoder，Key和Value来自encoder
        cross_attn_output = self.cross_attention(gru_output, hist_enc)
        
        gru_output = residual + cross_attn_output  # 残差连接
        
        # 7.3 FFN子层
        residual = gru_output
        gru_output = self.ffn_layer_norm(gru_output)  # Pre-norm
        ffn_output = self.ffn(gru_output)
        final_output = residual + ffn_output  # 残差连接
        
        # 8. 输出预测
        position_pred = self.output_proj(final_output)  # [B, out_length, 1]
        position_pred = F.relu(position_pred)
        
        return position_pred, final_output


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
        self.norm1 = RMSNorm(encoder_size)
        self.norm2 = RMSNorm(encoder_size)
        self.norm3 = RMSNorm(encoder_size)
        
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
        
        return {
            'acc_params': acc_params,
            'idm_params': idm_params,
            'cf_probs': cf_probs
        }


class Encoder(nn.Module):
    """改进的编码器"""
    def __init__(self, args):
        super().__init__()
        self.device = args['device']
        self.encoder_size = args['encoder_size']
        self.n_head = args['n_head']
        self.cf_type = args['cf_type']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.f_length = args['f_length']
        self.dropout = args['dropout']
        self.transformer_layer = args['transformer_layer']
        
        # 输入编码器
        self.encoder = nn.Linear(self.f_length, self.encoder_size)
        self.relu = nn.GELU()
        
        # Mamba配置（保持原始设置但稍作改进）
        self.mamba_config = MambaConfig(d_model=self.encoder_size, n_layers=1, d_state=16)
        self.mamba = Mamba(self.mamba_config)
        
        # 类型查询
        self.type_query = nn.Parameter(t.randn(1, 1, self.encoder_size))
        nn.init.xavier_uniform_(self.type_query)
        
        # 使用Flash Attention的Transformer层
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': FlashAttention(self.encoder_size, self.n_head, self.dropout),
                'ffn': nn.Sequential(
                    RMSNorm(self.encoder_size),
                    SwiGLU(self.encoder_size, self.encoder_size * 4),
                    nn.Linear(self.encoder_size * 4, self.encoder_size),
                    nn.Dropout(self.dropout)
                ),
                'norm1': RMSNorm(self.encoder_size),
                'norm2': RMSNorm(self.encoder_size)
            }) for _ in range(self.transformer_layer)
        ])
        
        # 类型预测器
        self.type_predictor = nn.Sequential(
            SwiGLU(self.encoder_size, self.encoder_size * 2),
            nn.Linear(self.encoder_size * 2, self.encoder_size),
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder_size, self.cf_type)
        )
        
        # 预测头
        self.trajectory_head = TrajectoryPredictionHead(
            args, self.encoder_size, self.in_length, self.out_length, self.cf_type
        )
        
        self.parameter_head = ParameterPredictionHead(
            self.encoder_size, self.cf_type, self.dropout, self.n_head
        )
        
        # 归一化层
        self.norm = RMSNorm(self.encoder_size)
        
    def generate_cv_ca_predictions(self, hist, dt=0.1):
        """生成恒速(CV)和恒加速度(CA)模型的初始预测"""
        batch_size = hist.shape[0]
        
        # 获取最后一个时间步的位置和速度
        last_pos = hist[:, -1, -3:-2]  # 位置
        last_vel = hist[:, -1, 2:3]  # 速度
        
        # 计算加速度
        if hist.shape[1] >= 2:
            prev_vel = hist[:, -2, 2:3]
            accel = (last_vel - prev_vel) / dt
        else:
            accel = t.zeros_like(last_vel)
        
        # 数值稳定性检查
        accel = t.clamp(accel, min=-10, max=10)
        
        # 生成未来时间步
        t_future = (t.arange(1, self.out_length + 1, device=last_pos.device).float()) * dt
        
        # 扩展维度
        last_pos_expanded = last_pos.expand(-1, self.out_length)
        last_vel_expanded = last_vel.expand(-1, self.out_length)
        accel_expanded = accel.expand(-1, self.out_length)
        
        # CV预测
        cv_preds = last_pos_expanded + last_vel_expanded * t_future
        
        # CA预测
        ca_preds = last_pos_expanded + last_vel_expanded * t_future + 0.5 * accel_expanded * (t_future ** 2)
        
        # 堆叠预测结果 - 直接返回[B, out_length, 2]格式
        init_predictions = t.stack([cv_preds, ca_preds], dim=-1)  # [B, out_length, 2]
        
        return init_predictions
    
    def forward(self, hist):
        # 编码历史轨迹
        hist_enc = self.relu(self.encoder(hist))
        
        # Mamba处理
        mamba_output = self.mamba(hist_enc)
        
        # 添加类型查询
        type_query = self.type_query.expand(hist_enc.shape[0], -1, -1)
        transformer_input = t.cat([mamba_output, type_query], dim=1)
        
        # 使用Flash Attention的Transformer编码
        historical_encoding = transformer_input
        for layer in self.transformer_layers:
            # 自注意力
            residual = historical_encoding
            historical_encoding = layer['norm1'](historical_encoding)
            historical_encoding = layer['attention'](historical_encoding)
            historical_encoding = residual + historical_encoding
            
            # FFN
            residual = historical_encoding
            historical_encoding = layer['ffn'](historical_encoding)
            historical_encoding = residual + historical_encoding
            historical_encoding = layer['norm2'](historical_encoding)
        
        # 提取类型特征并预测
        type_features = historical_encoding[:, -1, :]
        cf_type_logits = self.type_predictor(type_features)
        cf_type_probs = F.softmax(cf_type_logits, dim=-1)
        
        # 生成初始预测
        init_predictions = self.generate_cv_ca_predictions(hist)
        
        # 轨迹预测 - 使用historical_encoding而不是mamba_output
        trajectory_pred, traj_encoding = self.trajectory_head(
            historical_encoding[:, :-1, :],  # 去除类型查询部分
            cf_type_probs, 
            init_predictions
        )
        
        # 参数预测 - 同样使用historical_encoding
        param_outputs = self.parameter_head(
            historical_encoding[:, :-1, :],  # 去除类型查询部分
            cf_type_logits, 
            traj_encoding
        )
        
        return {
            'position_pred': trajectory_pred,
            'cf_type_pred': cf_type_logits,
            'params': param_outputs,
            'traj_encoding': traj_encoding
        }


class predictor:
    """预测器类，保持原始设计"""
    def __init__(self, args):
        self.args = args
        self.epsilon = 1e-8
        self.device = args['device']
        self.train_flag = args['train_flag']
        self.out_dim = args['out_dim']
        self.dropout = args['dropout']
        self.batch_size = args['batch_size']
        self.out_length = args['out_length']
        self.in_length = args['in_length']
        self.dt = args['time_step']
        self.cf_type = args['cf_type']
        
    def acc_model(self, params, veh_state, lead_vel, delta_V_t):
        """ACC模型"""
        kp = F.softplus(params[:, 0:1]) + self.epsilon
        kd = F.softplus(params[:, 1:2]) + self.epsilon
        v_des = F.softplus(params[:, 2:3]) + self.epsilon
        h0 = F.softplus(params[:, 3:4]) + self.epsilon
        h1 = F.softplus(params[:, 4:5]) + self.epsilon
        kv = F.softplus(params[:, 5:6]) + self.epsilon
        
        V_t = veh_state[:, 1:2]
        H_t = veh_state[:, 0:1]
        
        h_des = h0 + h1 * V_t
        spacing_error = H_t - h_des
        velocity_error = v_des - V_t
        
        a_t = kp * spacing_error + kd * delta_V_t + kv * velocity_error
        a_t = t.clamp(a_t, min=-6, max=4)
        
        return a_t
    
    def idm_model(self, params, veh_state, delta_V_t):
        """IDM模型"""
        s0 = F.softplus(params[:, 0:1]) + self.epsilon
        T = F.softplus(params[:, 1:2]) + self.epsilon
        b = F.softplus(params[:, 2:3]) + self.epsilon
        a_max = F.softplus(params[:, 3:4]) + self.epsilon
        v0 = F.softplus(params[:, 4:5]) + self.epsilon
        delta = F.softplus(params[:, 5:6]) + 1.0
        
        V_t = veh_state[:, 1:2]
        H_t = veh_state[:, 0:1]
        
        interaction_term = V_t * delta_V_t / (2 * t.sqrt(a_max * b + self.epsilon) + self.epsilon)
        s_star = s0 + V_t * T + t.clamp(interaction_term, min=-100, max=100)
        
        velocity_ratio = t.clamp(V_t / (v0 + self.epsilon), min=0, max=2)
        spacing_ratio = t.clamp(s_star / (H_t + self.epsilon), min=0, max=10)
        
        a_t = a_max * (1 - velocity_ratio**delta - spacing_ratio**2)
        a_t = t.clamp(a_t, min=-6, max=4)
        
        return a_t
    
    def veh_dynamic(self, veh_state, acc_params_list, idm_params_list, cf_probs, next_v):
        """车辆动力学计算"""
        V_t = veh_state[:, 1:2]
        H_t = veh_state[:, 0:1]
        
        delta_V_t = next_v - V_t
        
        # 并行计算所有专家的加速度
        all_accelerations = []
        
        # ACC模型
        for acc_params in acc_params_list:
            acc_a = self.acc_model(acc_params.squeeze(1), veh_state, next_v, delta_V_t)
            all_accelerations.append(acc_a)
        
        # IDM模型
        for idm_params in idm_params_list:
            idm_a = self.idm_model(idm_params.squeeze(1), veh_state, delta_V_t)
            all_accelerations.append(idm_a)
        
        # 堆叠加速度
        all_accelerations = t.stack(all_accelerations, dim=1)
        
        # 加权平均
        a_t = t.sum(cf_probs.unsqueeze(-1) * all_accelerations, dim=1)
        
        # 更新状态
        V_t_next = V_t + a_t * self.dt
        V_t_next = t.clamp(V_t_next, min=0, max=50)
        
        H_t_next = H_t + delta_V_t * self.dt
        H_t_next = t.clamp(H_t_next, min=0.1, max=500)
        
        veh_state_next = t.cat([H_t_next, V_t_next], dim=-1)
        return veh_state_next
    
    def forward(self, model_outputs, nextv, veh_state):
        """前向传播"""
        position_pred = model_outputs['position_pred']
        params = model_outputs['params']
        cf_probs = params['cf_probs']
        acc_params_list = params['acc_params']
        idm_params_list = params['idm_params']
        
        batch_size = veh_state.shape[0]
        veh_states = [veh_state]
        
        for t_step in range(self.out_length):
            next_v_t = nextv[:, t_step:t_step+1]
            
            next_state = self.veh_dynamic(
                veh_states[-1],
                acc_params_list,
                idm_params_list,
                cf_probs,
                next_v_t
            )
            
            veh_states.append(next_state)
        
        veh_states_tensor = t.stack(veh_states[1:], dim=1)
        
        dynamic_positions = veh_states_tensor[:, :, 0]
        dynamic_velocities = veh_states_tensor[:, :, 1]
        
        dynamic_pred = t.cat([
            dynamic_positions.unsqueeze(-1),
            dynamic_velocities.unsqueeze(-1)
        ], dim=-1)
        
        direct_pred = position_pred
        
        return {
            'dynamic_pred': dynamic_pred,
            'direct_pred': direct_pred,
            'veh_states': veh_states_tensor
        }


class MoEGRUEnhanced(nn.Module):
    """增强版MoE-GRU模型"""
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(args)
        
    def forward(self, hist):
        return self.encoder(hist)