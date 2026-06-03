from __future__ import division
import math
import torch
import torch as t
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

def outputActivation(x):
    """
    简化的输出激活函数，适用于跟驰轨迹预测
    """
    # 对于跟驰任务，我们只需要位置预测，不需要复杂的概率分布
    return x

class AddAndNorm(nn.Module):
    def __init__(self, hidden_layer_size):
        super(AddAndNorm, self).__init__()
        self.normalize = nn.LayerNorm(hidden_layer_size)

    def forward(self, x1, x2, x3=None):
        if x3 is not None:
            x = t.add(t.add(x1, x2), x3)
        else:
            x = t.add(x1, x2)
        return self.normalize(x)

class GLU(nn.Module):
    # Gated Linear Unit
    def __init__(self, input_size, hidden_layer_size, dropout_rate=None):
        super(GLU, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate
        if dropout_rate is not None:
            self.dropout = nn.Dropout(self.dropout_rate)
        self.activation_layer = t.nn.Linear(input_size, hidden_layer_size)
        self.gated_layer = t.nn.Linear(input_size, hidden_layer_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.dropout_rate is not None:
            x = self.dropout(x)
        activation = self.activation_layer(x)
        gated = self.sigmoid(self.gated_layer(x))
        return t.mul(activation, gated), gated

class CarFollowingEncoder(nn.Module):
    """基于STDAN的跟驰编码器"""
    
    def __init__(self, args):
        super(CarFollowingEncoder, self).__init__()
        self.encoder_size = args['encoder_size']
        self.n_head = args['n_head']
        self.in_length = args['in_length']
        self.dropout = args['dropout']
        
        # 输入嵌入层 - 分别处理前车和跟驰车
        self.leading_emb = nn.Linear(3, self.encoder_size)  # 前车：速度、加速度、位置
        self.following_emb = nn.Linear(3, self.encoder_size)  # 跟驰车：速度、加速度、位置
        
        # LSTM编码器
        self.leading_lstm = nn.LSTM(self.encoder_size, self.encoder_size, batch_first=True)
        self.following_lstm = nn.LSTM(self.encoder_size, self.encoder_size, batch_first=True)
        
        # 激活函数
        self.activation = nn.LeakyReLU(0.1)
        
        # 空间注意力层 - 使用PyTorch自带的MultiheadAttention
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=self.encoder_size,
            num_heads=self.n_head,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 空间注意力后的前馈网络
        self.spatial_ffn = GLU(
            input_size=self.encoder_size,
            hidden_layer_size=self.encoder_size,
            dropout_rate=self.dropout
        )
        
        # 时间注意力层 - 使用TransformerEncoder进行自注意力
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_size,
            nhead=self.n_head,
            dim_feedforward=self.encoder_size * 4,
            dropout=self.dropout,
            activation='relu',
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=4
        )
        
        # 归一化层
        self.layer_norm1 = nn.LayerNorm(self.encoder_size)
        self.layer_norm2 = nn.LayerNorm(self.encoder_size)
        
    def forward(self, leading_features, following_features):
        """
        Args:
            leading_features: [B, seq_len, 3] 前车特征
            following_features: [B, seq_len, 3] 跟驰车特征
        """
        batch_size = leading_features.shape[0]
        
        # 嵌入层
        leading_emb = self.activation(self.leading_emb(leading_features))
        following_emb = self.activation(self.following_emb(following_features))
        
        # LSTM编码
        leading_enc, _ = self.leading_lstm(leading_emb)  # [B, seq_len, encoder_size]
        following_enc, _ = self.following_lstm(following_emb)  # [B, seq_len, encoder_size]
        
        # 空间注意力：跟驰车关注前车 - 使用PyTorch自带的MultiheadAttention
        # 现在使用batch_first=True，所以输入维度是 [B, seq_len, encoder_size]
        spatial_attended, _ = self.spatial_attention(
            query=following_enc,  # [B, seq_len, encoder_size]
            key=leading_enc,      # [B, seq_len, encoder_size]
            value=leading_enc     # [B, seq_len, encoder_size]
        )
        
        # 残差连接和层归一化
        spatial_residual = self.layer_norm1(following_enc + spatial_attended)
        
        # 空间注意力后的前馈网络
        spatial_ffn_out, _ = self.spatial_ffn(spatial_residual)
        
        # 第二次残差连接
        values = self.layer_norm1(spatial_residual + spatial_ffn_out)
        
        # 时间注意力 - 使用TransformerEncoder进行自注意力
        temporal_attended = self.temporal_transformer(values)
        
        # 最终残差连接和层归一化
        final_values = self.layer_norm2(values + temporal_attended)
        
        return final_values  # [B, seq_len, encoder_size]

class CarFollowingDecoder(nn.Module):
    """基于STDAN的跟驰解码器"""
    
    def __init__(self, args):
        super(CarFollowingDecoder, self).__init__()
        self.encoder_size = args['encoder_size']
        self.out_length = args['out_length']
        self.cf_type = args['cf_type']
        self.use_cf_type = args.get('use_cf_type', True)
        
        # LSTM解码器
        self.lstm = nn.LSTM(self.encoder_size, self.encoder_size)
        
        # 输出层 - 位置预测
        self.linear1 = nn.Linear(self.encoder_size, 1)
        
        # 如果使用跟驰类型信息
        if self.use_cf_type:
            self.dec_linear = nn.Linear(self.encoder_size + self.cf_type, self.encoder_size)
        
    def forward(self, dec, cf_type_enc=None):
        """
        Args:
            dec: [B, out_length, encoder_size] 映射后的解码器输入
            cf_type_enc: [B, cf_type] 跟驰类型编码
        """
        if self.use_cf_type and cf_type_enc is not None:
            # 扩展跟驰类型到序列长度
            cf_type_enc = cf_type_enc.unsqueeze(1).repeat(1, self.out_length, 1)
            # 拼接特征和类型信息
            dec = torch.cat((dec, cf_type_enc), -1)
            dec = self.dec_linear(dec)
        
        # 转换为LSTM期望的格式 [seq_len, batch, features]
        dec = dec.permute(1, 0, 2)  # [out_length, B, encoder_size]
        
        # LSTM解码
        h_dec, _ = self.lstm(dec)
        
        # 位置预测
        fut_pred = self.linear1(h_dec)
        
        return fut_pred

class CarFollowingGenerator(nn.Module):
    """基于STDAN的跟驰生成器，实现类型引导的特征映射"""
    
    def __init__(self, args):
        super(CarFollowingGenerator, self).__init__()
        self.encoder_size = args['encoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.cf_type = args['cf_type']
        self.n_head = args.get('n_head', 4)
        self.att_out = args.get('att_out', 16)
        self.train_flag = args.get('train_flag', True)
        self.use_true_cf = args.get('use_true_cf', False)
        
        # 解码器
        self.decoder = CarFollowingDecoder(args)
        
        # 跟驰类型预测网络
        hidden_size = self.encoder_size // 2
        self.mu_fc1 = nn.Linear(self.encoder_size, hidden_size)
        self.mu_fc = nn.Linear(hidden_size, self.encoder_size)
        self.cf_type_classifier = nn.Linear(self.encoder_size, self.cf_type)
        
        # 激活函数和归一化
        self.activation = nn.LeakyReLU(0.1)
        self.normalize = nn.LayerNorm(self.encoder_size)
        
        # 关键：映射参数，将历史特征映射到未来
        self.mapping = nn.Parameter(torch.Tensor(self.in_length, self.out_length, self.cf_type))
        nn.init.xavier_uniform_(self.mapping, gain=1.414)
        
    def forward(self, values, cf_type_enc=None):
        """
        Args:
            values: [B, seq_len, encoder_size] 编码器输出的历史特征
            cf_type_enc: [B, cf_type] 真实跟驰类型（训练时使用）
        """
        # 使用最后一个时间步预测跟驰类型
        maneuver_state = values[:, -1, :]  # [B, encoder_size]
        maneuver_state = self.activation(self.mu_fc1(maneuver_state))
        maneuver_state = self.activation(self.normalize(self.mu_fc(maneuver_state)))
        cf_type_pred = F.softmax(self.cf_type_classifier(maneuver_state), dim=-1)

        # 训练时的处理
        if self.use_true_cf and cf_type_enc is not None:
            # 使用真实跟驰类型
            cf_man = torch.argmax(cf_type_enc, dim=-1).detach()
        else:
            # 使用预测的跟驰类型
            cf_man = torch.argmax(cf_type_pred, dim=-1).detach().unsqueeze(1)
            cf_enc_tmp = torch.zeros_like(cf_type_pred)
            cf_type_enc = cf_enc_tmp.scatter_(1, cf_man, 1)

        # 使用跟驰类型索引进行映射
        index = cf_type_enc.permute(-1, 0)  # [cf_type, B]
        mapping = torch.matmul(self.mapping, index)  # [in_length, out_length, B]
        mapping = F.softmax(mapping.permute(2, 1, 0), dim=-1)  # [B, out_length, in_length]

        # 将历史特征映射到未来时间步
        # values: [B, in_length, encoder_size] (已经是batch_first格式)
        # mapping: [B, out_length, in_length], values: [B, in_length, encoder_size]
        # 批量矩阵乘法: [B, out_length, in_length] x [B, in_length, encoder_size] = [B, out_length, encoder_size]
        dec = torch.matmul(mapping, values)  # [B, out_length, encoder_size]

        # 解码生成未来轨迹
        fut_pred = self.decoder(dec, cf_type_enc if cf_type_enc is not None else cf_type_pred)

        return fut_pred, cf_type_pred


class CarFollowingNet(nn.Module):
    """基于STDAN的跟驰轨迹预测网络"""
    
    def __init__(self, args):
        super(CarFollowingNet, self).__init__()
        
        # 网络参数
        self.encoder_size = args['encoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.cf_type = args['cf_type']  # 跟驰类型数量
        self.train_flag = args['train_flag']
        
        # 编码器
        self.encoder = CarFollowingEncoder(args)
        
        # 生成器（包含映射和解码）
        self.generator = CarFollowingGenerator(args)
        
        # 激活函数
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, hist, cf_type_enc=None):
        """
        前向传播
        Args:
            hist: [B, seq_len, features] 历史轨迹数据
            cf_type_enc: [B, cf_type] 真实跟驰类型（可选，训练时使用）
        """
        batch_size = hist.shape[0]
        
        # 提取前车和跟驰车的特征
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
        
        # 编码器处理
        encoded_features = self.encoder(leading_features, following_features)
        # encoded_features: [seq_len, B, encoder_size]
        

        position_pred, cf_type_pred = self.generator(encoded_features, cf_type_enc)
        # position_pred: [out_length, B, 1] -> [B, out_length, 1]
        position_pred = position_pred.permute(1, 0, 2)

        return {
            'position_pred': position_pred,
            'cf_type_pred': cf_type_pred
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
class Seq2SeqBaseline(CarFollowingNet):
    """兼容性别名"""
    pass