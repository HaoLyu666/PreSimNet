# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange, repeat

def outputActivation(x):
    """
    简化的输出激活函数，适用于跟驰轨迹预测
    """
    return x

class RelativePositionEncoding(nn.Module):
    """相对位置和速度编码模块，基于HLTP的核心改进"""
    
    def __init__(self, input_dim, hidden_dim):
        super(RelativePositionEncoding, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 相对位置编码网络
        self.position_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 相对速度编码网络
        self.velocity_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, relative_features):
        """
        Args:
            relative_features: [B, seq_len, input_dim] 相对特征（间隙、速度差等）
        Returns:
            encoded_features: [B, seq_len, hidden_dim] 编码后的相对特征
        """
        # 分别编码位置和速度信息
        pos_encoded = self.position_encoder(relative_features)
        vel_encoded = self.velocity_encoder(relative_features)
        
        # 融合编码
        combined = torch.cat([pos_encoded, vel_encoded], dim=-1)
        fused_features = self.fusion_network(combined)
        
        return fused_features

class SpatialGraphConvolution(nn.Module):
    """简化的空间图卷积模块，模拟HLTP中的图卷积思想"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpatialGraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 图卷积层
        self.graph_conv1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.graph_conv2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 注意力权重计算
        self.attention_weights = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, leading_features, following_features):
        """
        Args:
            leading_features: [B, seq_len, input_dim] 前车特征
            following_features: [B, seq_len, input_dim] 跟驰车特征
        Returns:
            graph_output: [B, seq_len, output_dim] 图卷积输出
        """
        batch_size, seq_len, _ = leading_features.shape
        
        # 计算车辆间的注意力权重
        combined_features = torch.cat([leading_features, following_features], dim=-1)
        attention_weights = self.attention_weights(combined_features)  # [B, seq_len, 1]
        
        # 加权融合特征
        weighted_leading = leading_features * attention_weights
        weighted_following = following_features * (1 - attention_weights)
        fused_features = weighted_leading + weighted_following
        
        # 图卷积处理
        graph_conv1_out = self.graph_conv1(fused_features)
        graph_output = self.graph_conv2(graph_conv1_out)
        
        return graph_output

class InformerLikeTransformer(nn.Module):
    """类似Informer的时空融合模块"""
    
    def __init__(self, d_model, nhead, num_layers, dropout=0.1):
        super(InformerLikeTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
    def forward(self, spatial_features, temporal_features):
        """
        Args:
            spatial_features: [B, seq_len, d_model] 空间特征
            temporal_features: [B, seq_len, d_model] 时间特征
        Returns:
            fused_output: [B, seq_len, d_model] 融合后的特征
        """
        batch_size, seq_len, _ = spatial_features.shape
        
        # 特征融合
        combined_features = spatial_features + temporal_features
        
        # 添加位置编码
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        combined_features = combined_features + pos_enc
        
        # Transformer编码
        fused_output = self.transformer_encoder(combined_features)
        
        return fused_output

class GLU(nn.Module):
    """门控线性单元"""
    
    def __init__(self, input_size, hidden_layer_size, dropout_rate=None):
        super(GLU, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.activation_layer = nn.Linear(input_size, hidden_layer_size)
        self.gated_layer = nn.Linear(input_size, hidden_layer_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.dropout_rate is not None:
            x = self.dropout(x)
        activation = self.activation_layer(x)
        gated = self.sigmoid(self.gated_layer(x))
        return torch.mul(activation, gated), gated

class AddAndNorm(nn.Module):
    """残差连接和层归一化"""
    
    def __init__(self, hidden_layer_size):
        super(AddAndNorm, self).__init__()
        self.normalize = nn.LayerNorm(hidden_layer_size)

    def forward(self, x1, x2, x3=None):
        if x3 is not None:
            x = torch.add(torch.add(x1, x2), x3)
        else:
            x = torch.add(x1, x2)
        return self.normalize(x)

class HLTPCarFollowingNet(nn.Module):
    """基于HLTP思想的跟驰轨迹预测网络"""
    
    def __init__(self, args):
        super(HLTPCarFollowingNet, self).__init__()
        
        # 网络参数
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['encoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.cf_type = args['cf_type']
        self.n_head = args.get('n_head', 4)
        self.dropout = args.get('dropout', 0.1)
        self.train_flag = args.get('train_flag', True)
        
        # 嵌入层大小
        self.input_embedding_size = args['encoder_size']
        self.dyn_embedding_size = args['encoder_size']
        self.social_embedding_size = args['encoder_size']
        
        # 1. 输入嵌入层
        self.leading_emb = nn.Linear(3, self.input_embedding_size)  # 前车：速度、加速度、位置
        self.following_emb = nn.Linear(3, self.input_embedding_size)  # 跟驰车：速度、加速度、位置
        
        # 2. HLTP核心改进1：相对位置和速度编码
        self.relative_encoding = RelativePositionEncoding(
            input_dim=4,  # 间隙、车头间距、速度差、加速度差
            hidden_dim=self.encoder_size
        )
        
        # 3. LSTM编码器
        self.leading_lstm = nn.LSTM(
            self.input_embedding_size, 
            self.encoder_size, 
            num_layers=2,
            batch_first=True,
            dropout=self.dropout
        )
        self.following_lstm = nn.LSTM(
            self.input_embedding_size, 
            self.encoder_size, 
            num_layers=2,
            batch_first=True,
            dropout=self.dropout
        )
        
        # 4. HLTP核心改进2：空间图卷积
        self.spatial_graph_conv = SpatialGraphConvolution(
            input_dim=self.encoder_size,
            hidden_dim=self.encoder_size,
            output_dim=self.encoder_size
        )
        
        # 5. 动态嵌入层
        self.leading_dyn_emb = nn.Linear(self.encoder_size, self.dyn_embedding_size)
        self.following_dyn_emb = nn.Linear(self.encoder_size, self.dyn_embedding_size)
        
        # 6. HLTP核心改进3：增强的注意力机制
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=self.encoder_size,
            num_heads=self.n_head,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 7. GLU门控单元
        self.first_glu = GLU(
            input_size=self.encoder_size,
            hidden_layer_size=self.encoder_size,
            dropout_rate=self.dropout
        )
        
        self.second_glu = GLU(
            input_size=self.encoder_size,
            hidden_layer_size=self.encoder_size,
            dropout_rate=self.dropout
        )
        
        # 8. HLTP核心改进4：Informer类似的时空融合
        self.informer_transformer = InformerLikeTransformer(
            d_model=self.encoder_size,
            nhead=self.n_head,
            num_layers=2,
            dropout=self.dropout
        )
        
        # 9. 残差连接和归一化
        self.addAndNorm = AddAndNorm(self.encoder_size)
        
        # 10. 社会交互层
        social_input_dim = self.dyn_embedding_size * 2 + self.encoder_size * 2  # 前车+跟驰车+相对编码+图卷积
        self.social_interaction = nn.Sequential(
            nn.Linear(social_input_dim, self.social_embedding_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.social_embedding_size, self.social_embedding_size)
        )
        
        # 11. 跟驰类型预测网络
        hidden_size = self.encoder_size
        self.mu_fc1 = nn.Sequential(
            nn.Linear(self.social_embedding_size + self.dyn_embedding_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        self.mu_fc = nn.Sequential(
            nn.Linear(hidden_size, self.social_embedding_size + self.dyn_embedding_size),
            nn.ReLU()
        )
        
        self.cf_type_classifier = nn.Sequential(
            nn.Linear(self.social_embedding_size + self.dyn_embedding_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_size, self.cf_type)
        )
        
        # 12. 映射参数（类似STDAN）
        self.mapping = nn.Parameter(
            torch.Tensor(self.in_length, self.out_length, self.cf_type)
        )
        nn.init.xavier_uniform_(self.mapping, gain=1.414)
        
        # 13. 维度投影层（确保映射特征与解码器输入维度匹配）
        decoder_input_size = self.social_embedding_size + self.dyn_embedding_size + self.cf_type
        self.dec_projection = nn.Linear(self.encoder_size, decoder_input_size)
        
        # 14. 解码器LSTM
        self.dec_lstm = nn.LSTM(
            decoder_input_size, 
            self.decoder_size, 
            num_layers=2,
            batch_first=True,
            dropout=self.dropout
        )
        
        # 14. 输出层
        self.position_output = nn.Sequential(
            nn.Linear(self.decoder_size, self.decoder_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.decoder_size // 2, 1)
        )
        
        # 激活函数
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.activation = nn.LeakyReLU(0.1)
        self.normalize = nn.LayerNorm(self.social_embedding_size + self.dyn_embedding_size)
        
    def forward(self, hist, cf_type_enc=None):
        """
        前向传播，基于HLTP模型的处理流程
        Args:
            hist: [B, seq_len, features] 历史轨迹数据
            cf_type_enc: [B, cf_type] 真实跟驰类型（可选，训练时使用）
        """
        batch_size = hist.shape[0]
        
        # 1. 特征提取
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
        
        # 相对特征：间隙、车头间距、速度差、加速度差
        relative_features = torch.stack([
            hist[:, :, 4],  # 车间间隙
            hist[:, :, 5],  # 车头间距
            hist[:, :, 6],  # 速度差
            hist[:, :, 9]   # 加速度差
        ], dim=2)
        
        # 2. 输入嵌入
        leading_emb = self.leaky_relu(self.leading_emb(leading_features))
        following_emb = self.leaky_relu(self.following_emb(following_features))
        
        # 3. HLTP改进1：相对位置和速度编码
        relative_encoded = self.relative_encoding(relative_features)  # [B, seq_len, encoder_size]
        
        # 4. LSTM编码
        leading_enc, _ = self.leading_lstm(leading_emb)  # [B, seq_len, encoder_size]
        following_enc, _ = self.following_lstm(following_emb)  # [B, seq_len, encoder_size]
        
        # 5. HLTP改进2：空间图卷积
        graph_conv_output = self.spatial_graph_conv(leading_enc, following_enc)  # [B, seq_len, encoder_size]
        
        # 6. 动态嵌入
        leading_dyn = self.leaky_relu(self.leading_dyn_emb(leading_enc[:, -1, :]))  # [B, encoder_size]
        following_dyn = self.leaky_relu(self.following_dyn_emb(following_enc[:, -1, :]))  # [B, encoder_size]
        
        # 7. HLTP改进3：增强的空间注意力
        spatial_attended, _ = self.spatial_attention(
            query=following_enc,
            key=leading_enc,
            value=leading_enc
        )
        
        # 8. GLU门控和残差连接
        spatial_glu_out, _ = self.first_glu(spatial_attended)
        spatial_residual = self.addAndNorm(following_enc, spatial_glu_out)
        
        # 9. HLTP改进4：Informer类似的时空融合
        informer_output = self.informer_transformer(graph_conv_output, spatial_residual)
        
        # 10. 第二个GLU和残差连接
        temporal_glu_out, _ = self.second_glu(informer_output)
        final_encoded = self.addAndNorm(spatial_residual, temporal_glu_out)
        
        # 11. 社会交互建模
        # 融合：前车动态特征 + 跟驰车动态特征 + 相对编码 + 图卷积输出
        interaction_input = torch.cat([
            leading_dyn, 
            following_dyn, 
            relative_encoded[:, -1, :],  # 最后时刻的相对编码
            graph_conv_output[:, -1, :]  # 最后时刻的图卷积输出
        ], dim=1)
        
        social_enc = self.social_interaction(interaction_input)
        
        # 12. 跟驰类型预测
        combined_features = torch.cat([social_enc, following_dyn], dim=1)
        maneuver_state = self.mu_fc1(combined_features)
        maneuver_state = self.normalize(self.mu_fc(maneuver_state))
        cf_type_pred = self.cf_type_classifier(maneuver_state)
        cf_type_prob = self.softmax(cf_type_pred)
        
        # 13. 类型引导的特征映射（类似STDAN）
        if self.train_flag and cf_type_enc is not None:
            # 训练时使用真实类型
            cf_man = torch.argmax(cf_type_enc, dim=-1).detach().unsqueeze(1)
            cf_enc_tmp = torch.zeros_like(cf_type_prob)
            cf_type_used = cf_enc_tmp.scatter_(1, cf_man, 1)
        else:
            # 测试时使用预测类型
            cf_type_used = cf_type_prob
        
        # 使用映射参数将历史特征映射到未来
        index = cf_type_used.permute(-1, 0)  # [cf_type, B]
        mapping = torch.matmul(self.mapping, index)  # [in_length, out_length, B]
        mapping = F.softmax(mapping.permute(2, 1, 0), dim=-1)  # [B, out_length, in_length]
        
        # 映射最终编码的特征
        dec = torch.matmul(mapping, final_encoded)  # [B, out_length, encoder_size]
        
        # 14. 解码器
        decoder_input = torch.cat([social_enc, following_dyn, cf_type_prob], dim=1)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, self.out_length, 1)
        
        # 将映射特征投影到与解码器输入相同的维度
        dec_projected = self.dec_projection(dec)
        
        # 将映射特征与解码器输入结合
        decoder_input = decoder_input + dec_projected
        
        decoder_output, _ = self.dec_lstm(decoder_input)
        
        # 15. 位置预测
        position_pred = self.position_output(decoder_output)  # [B, out_length, 1]
        
        return {
            'position_pred': position_pred,
            'cf_type_pred': cf_type_pred,
            'spatial_attention': spatial_attended,  # HLTP特有：空间注意力
            'relative_encoding': relative_encoded,  # HLTP特有：相对编码
            'graph_conv_output': graph_conv_output,  # HLTP特有：图卷积输出
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
class Seq2SeqBaseline(HLTPCarFollowingNet):
    """兼容性别名"""
    pass