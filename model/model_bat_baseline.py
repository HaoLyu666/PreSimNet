import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def outputActivation(x):
    """
    简化的输出激活函数，适用于跟驰轨迹预测
    """
    return x

class BehaviorAwareAttention(nn.Module):
    """行为感知注意力机制，基于BAT模型的核心思想"""
    
    def __init__(self, query_dim, key_value_dim, hidden_dim):
        super(BehaviorAwareAttention, self).__init__()
        self.query_dim = query_dim
        self.key_value_dim = key_value_dim
        self.hidden_dim = hidden_dim
        
        # 增强的注意力权重计算 - 增加更多层
        self.attention_fc1 = nn.Linear(query_dim + key_value_dim, hidden_dim * 2)
        self.attention_fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention_fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.context_vector = nn.Parameter(torch.randn(hidden_dim // 2))
        
        # 多头注意力机制 - 确保embed_dim能被num_heads整除
        self.num_heads = 4
        self.head_dim = hidden_dim // self.num_heads
        # 调整embed_dim为最接近的能被4整除的数
        adjusted_embed_dim = ((key_value_dim + 3) // 4) * 4
        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=adjusted_embed_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 如果需要调整维度，添加投影层
        if key_value_dim != adjusted_embed_dim:
            self.dim_adjust = nn.Linear(key_value_dim, adjusted_embed_dim)
            self.dim_restore = nn.Linear(adjusted_embed_dim, key_value_dim)
        else:
            self.dim_adjust = None
            self.dim_restore = None
        
        # 查询投影层（用于维度匹配）
        self.query_proj = nn.Linear(query_dim, key_value_dim)
        
        # 注意力融合层
        self.attention_fusion = nn.Sequential(
            nn.Linear(key_value_dim * 2, hidden_dim),  # 修正维度：两个key_value_dim的输出拼接
            nn.ReLU(),
            nn.Linear(hidden_dim, key_value_dim)
        )
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, query, key_value):
        """
        Args:
            query: [B, query_dim] - 查询向量（通常是跟驰车特征）
            key_value: [B, seq_len, input_dim] - 键值对（通常是前车历史特征）
        Returns:
            attended_output: [B, input_dim] - 注意力加权后的输出
            attention_weights: [B, seq_len] - 注意力权重
        """
        batch_size, seq_len, _ = key_value.shape
        
        # 方法1：增强的单头注意力
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)  # [B, seq_len, query_dim]
        combined = torch.cat([query_expanded, key_value], dim=-1)  # [B, seq_len, query_dim + input_dim]
        
        # 多层注意力权重计算
        attention_scores = self.relu(self.attention_fc1(combined))  # [B, seq_len, hidden_dim * 2]
        attention_scores = self.dropout(attention_scores)
        attention_scores = self.relu(self.attention_fc2(attention_scores))  # [B, seq_len, hidden_dim]
        attention_scores = self.dropout(attention_scores)
        attention_scores = self.tanh(self.attention_fc3(attention_scores))  # [B, seq_len, hidden_dim // 2]
        attention_scores = torch.matmul(attention_scores, self.context_vector)  # [B, seq_len]
        attention_weights = self.softmax(attention_scores)  # [B, seq_len]
        
        # 单头注意力输出
        attended_output_single = torch.sum(key_value * attention_weights.unsqueeze(-1), dim=1)  # [B, input_dim]
        
        # 方法2：多头注意力
        # 为多头注意力准备query
        query_for_multihead = self.query_proj(query).unsqueeze(1)  # [B, 1, key_value_dim]
        
        # 调整维度（如果需要）
        if self.dim_adjust is not None:
            key_adj = self.dim_adjust(key_value)
            value_adj = self.dim_adjust(key_value)
            query_adj = self.dim_adjust(query_for_multihead)
        else:
            key_adj = key_value
            value_adj = key_value
            query_adj = query_for_multihead
            
        attended_output_multi, _ = self.multi_head_attention(
            query=query_adj,
            key=key_adj,
            value=value_adj
        )
        
        # 恢复维度（如果需要）
        if self.dim_restore is not None:
            attended_output_multi = self.dim_restore(attended_output_multi)
            
        attended_output_multi = attended_output_multi.squeeze(1)  # [B, input_dim]
        
        # 融合两种注意力机制的输出
        fusion_input = torch.cat([attended_output_single, attended_output_multi], dim=-1)
        attended_output = self.attention_fusion(fusion_input)
        
        return attended_output, attention_weights

class BATCarFollowingNet(nn.Module):
    """基于BAT模型思想的跟驰轨迹预测网络"""
    
    def __init__(self, args):
        super(BATCarFollowingNet, self).__init__()
        
        # 网络参数 - 增加维度以增加参数量
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['encoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.cf_type = args['cf_type']
        
        # 嵌入层大小 - 保持原始维度
        self.input_embedding_size = args['encoder_size']              # 保持原始维度
        self.dyn_embedding_size = args['encoder_size']                # 保持原始维度
        self.social_embedding_size = args['encoder_size']             # 保持原始维度
        self.feature_embedding_size = args['encoder_size']            # 特征嵌入维度
        
        # 输入嵌入层 - 保持简单
        self.leading_emb = nn.Linear(3, self.input_embedding_size)
        self.following_emb = nn.Linear(3, self.input_embedding_size)
        
        # 特征增强层
        self.feature_enhancement = nn.Sequential(
            nn.Linear(self.input_embedding_size, self.feature_embedding_size),
            nn.ReLU()
        )
        
        # LSTM编码器 - 保持简单
        self.leading_lstm = nn.LSTM(
            self.input_embedding_size, 
            self.encoder_size, 
            num_layers=1,  # 保持1层
            batch_first=True,
            bidirectional=True  # 使用双向LSTM
        )
        self.following_lstm = nn.LSTM(
            self.input_embedding_size, 
            self.encoder_size, 
            num_layers=1,  # 保持1层
            batch_first=True,
            bidirectional=True  # 使用双向LSTM
        )
        
        # LSTM输出投影层（因为双向LSTM输出维度翻倍）
        self.leading_lstm_proj = nn.Linear(self.encoder_size * 2, self.encoder_size)
        self.following_lstm_proj = nn.Linear(self.encoder_size * 2, self.encoder_size)
        
        # 动态嵌入层 - 保持简单
        self.leading_dyn_emb = nn.Linear(self.encoder_size, self.dyn_embedding_size)
        self.following_dyn_emb = nn.Linear(self.encoder_size, self.dyn_embedding_size)
        
        # BAT核心创新1：行为感知注意力机制
        self.behavior_attention = BehaviorAwareAttention(
            query_dim=self.dyn_embedding_size,
            key_value_dim=self.input_embedding_size,
            hidden_dim=self.encoder_size
        )
        
        # 相对位置编码 - 保持简单
        self.rel_pos_embedding = nn.Sequential(
            nn.Linear(2, self.encoder_size // 2),
            nn.ReLU(),
            nn.Linear(self.encoder_size // 2, self.encoder_size)
        )
        
        # 社会交互层 - 保持简单
        # 输入：前车动态特征 + 跟驰车动态特征 + 注意力输出 + 相对位置编码
        social_input_dim = self.dyn_embedding_size * 2 + self.input_embedding_size + self.encoder_size
        self.social_interaction = nn.Sequential(
            nn.Linear(social_input_dim, self.social_embedding_size),
            nn.ReLU()
        )
        
        # 社会交互增强层
        self.social_enhancement = nn.Sequential(
            nn.Linear(self.social_embedding_size, self.social_embedding_size),
            nn.ReLU()
        )
        
        # 增强的跟驰类型预测器 - 增加深度和复杂度
        hidden_size = self.encoder_size
        self.mu_fc1 = nn.Sequential(
            nn.Linear(self.social_embedding_size + self.dyn_embedding_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        self.mu_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, self.social_embedding_size + self.dyn_embedding_size)
        )
        
        self.cf_type_classifier = nn.Sequential(
            nn.Linear(self.social_embedding_size + self.dyn_embedding_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, self.cf_type)
        )
        
        # 解码器LSTM - 保持简单
        decoder_input_size = self.social_embedding_size + self.dyn_embedding_size + self.cf_type
        self.dec_lstm = nn.LSTM(
            decoder_input_size, 
            self.decoder_size, 
            num_layers=1,  # 保持1层
            batch_first=True,
            bidirectional=True  # 使用双向LSTM
        )
        
        # 解码器输出投影层
        self.decoder_proj = nn.Linear(self.decoder_size * 2, self.decoder_size)
        
        # 增强的输出层 - 适度增加深度
        self.position_output = nn.Sequential(
            nn.Linear(self.decoder_size, self.decoder_size // 2),
            nn.ReLU(),
            nn.Linear(self.decoder_size // 2, 1)
        )
        
        # 激活函数
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.LeakyReLU(0.1)
        self.normalize = nn.LayerNorm(self.social_embedding_size + self.dyn_embedding_size)
        
    def forward(self, hist):
        """
        前向传播，基于BAT模型的处理流程
        Args:
            hist: [B, seq_len, features] 历史轨迹数据
        """
        batch_size = hist.shape[0]
        
        # 1. 特征提取 - 与CSLSTM相同
        # 前车特征：速度、加速度、位置
        leading_features = torch.stack([
            hist[:, :, 0],  # 前车速度
            hist[:, :, 1],  # 前车加速度
            hist[:, :, 8]   # 前车位置
        ], dim=2)
        
        # 跟驰车特征：速度、加速度、位置
        following_features = torch.stack([
            hist[:, :, 2],  # 后车速度
            hist[:, :, 3],  # 后车加速度
            hist[:, :, 7]   # 后车位置
        ], dim=2)
        
        # 2. 输入嵌入
        leading_emb = self.leading_emb(leading_features)
        following_emb = self.following_emb(following_features)
        
        # 特征增强
        leading_enhanced = self.feature_enhancement(leading_emb)
        following_enhanced = self.feature_enhancement(following_emb)
        
        # 3. 增强的LSTM编码（双向LSTM）
        _, (leading_enc, _) = self.leading_lstm(leading_enhanced)
        _, (following_enc, _) = self.following_lstm(following_enhanced)
        
        # 处理双向LSTM的输出（取最后一层的前向和后向隐藏状态）
        leading_enc = torch.cat([leading_enc[-2], leading_enc[-1]], dim=1)  # [B, encoder_size * 2]
        following_enc = torch.cat([following_enc[-2], following_enc[-1]], dim=1)  # [B, encoder_size * 2]
        
        # 投影到原始维度
        leading_enc = self.leading_lstm_proj(leading_enc)  # [B, encoder_size]
        following_enc = self.following_lstm_proj(following_enc)  # [B, encoder_size]
        
        # 4. 增强的动态嵌入
        leading_dyn = self.leading_dyn_emb(leading_enc)
        following_dyn = self.following_dyn_emb(following_enc)
        
        # 5. BAT核心创新1：行为感知注意力机制
        # 使用跟驰车作为query，前车历史轨迹作为key-value
        attended_leading, attention_weights = self.behavior_attention(
            query=following_dyn,  # [B, dyn_embedding_size]
            key_value=leading_emb  # [B, seq_len, input_embedding_size]
        )
        
        # 6. BAT核心创新2：增强的相对位置编码
        # 使用最后时刻的相对特征
        relative_distance = hist[:, -1, 4:5]  # 最后时刻的车间间隙
        relative_velocity = hist[:, -1, 6:7]  # 最后时刻的速度差
        relative_features = torch.cat([relative_distance, relative_velocity], dim=1)  # [B, 2]
        rel_pos_emb = self.rel_pos_embedding(relative_features)  # [B, encoder_size]
        
        # 7. 增强的社会交互建模
        # 融合：前车动态特征 + 跟驰车动态特征 + 注意力输出 + 相对位置编码
        interaction_input = torch.cat([leading_dyn, following_dyn, attended_leading, rel_pos_emb], dim=1)
        social_enc = self.social_interaction(interaction_input)
        
        # 应用社会交互增强层
        social_enc = self.social_enhancement(social_enc)
        
        # 8. 增强的跟驰类型预测
        combined_features = torch.cat([social_enc, following_dyn], dim=1)
        maneuver_state = self.mu_fc1(combined_features)
        maneuver_state = self.normalize(self.mu_fc(maneuver_state))
        cf_type_pred = self.cf_type_classifier(maneuver_state)
        cf_type_prob = self.softmax(cf_type_pred)
        
        # 9. 增强的解码器（双向LSTM）
        decoder_input = torch.cat([social_enc, following_dyn, cf_type_prob], dim=1)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, self.out_length, 1)
        decoder_output, _ = self.dec_lstm(decoder_input)
        
        # 投影双向LSTM输出
        decoder_output = self.decoder_proj(decoder_output)  # [B, out_length, decoder_size]
        
        # 10. 增强的位置预测
        position_pred = self.position_output(decoder_output)  # [B, out_length, 1]
        
        return {
            'position_pred': position_pred,
            'cf_type_pred': cf_type_pred,
            'attention_weights': attention_weights,  # BAT特有：注意力权重
            'social_encoding': social_enc  # 返回社会编码用于分析
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
                position_change = position_pred[:, t, 0:1] - position_pred[:, t-1, 0:1]
                predicted_gap = dynamic_pred[:, t-1, 0:1] + position_change
                # 简单的速度估计（可以改进）
                predicted_velocity = dynamic_pred[:, t-1, 1:2] + position_change * 0.1
            
            dynamic_pred[:, t, 0] = predicted_gap.squeeze(-1)
            dynamic_pred[:, t, 1] = predicted_velocity.squeeze(-1)
        
        return {
            'dynamic_pred': dynamic_pred,  # [B, out_length, 2] (间隙, 速度)
            'direct_pred': position_pred   # [B, out_length, 1] (位置)
        }

# 为了兼容性，保留原始类名
class Seq2SeqBaseline(BATCarFollowingNet):
    """兼容性别名"""
    pass