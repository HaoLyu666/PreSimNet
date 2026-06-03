from __future__ import division
import torch
from torch import nn

def outputActivation(x):
    """
    简化的输出激活函数，适用于跟驰轨迹预测
    """
    # 对于跟驰任务，我们只需要位置预测，不需要复杂的概率分布
    return x

class CarFollowingNet(nn.Module):
    """基于PiP架构的跟驰轨迹预测网络（移除planning功能）"""
    
    def __init__(self, args):
        super(CarFollowingNet, self).__init__()
        self.args = args
        
        # 网络层大小
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['encoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.cf_type = args['cf_type']  # 跟驰类型数量
        
        # PiP相关参数设置
        self.temporal_embedding_size = args['encoder_size'] // 2
        self.soc_conv_depth = args['encoder_size'] // 2
        self.soc_conv2_depth = args['encoder_size'] // 4
        self.dynamics_encoding_size = args['encoder_size']
        
        # 计算社会上下文大小（基于简化的全连接层）
        self.social_context_size = self.soc_conv2_depth
        self.targ_enc_size = self.social_context_size + self.dynamics_encoding_size
        
        # 激活函数
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        ## 定义网络层
        
        # 输入嵌入层 - 分别处理前车和跟驰车
        self.leading_emb = nn.Linear(3, self.temporal_embedding_size)  # 前车：速度、加速度、位置
        self.following_emb = nn.Linear(3, self.temporal_embedding_size)  # 跟驰车：速度、加速度、位置
        
        # 时间卷积层：分别处理前车和跟驰车轨迹
        self.leading_temporalConv = nn.Conv1d(in_channels=3, out_channels=self.temporal_embedding_size, kernel_size=3, padding=1)
        self.following_temporalConv = nn.Conv1d(in_channels=3, out_channels=self.temporal_embedding_size, kernel_size=3, padding=1)
        
        # 编码器LSTM：分别编码前车和跟驰车的历史轨迹
        self.leading_lstm = nn.LSTM(input_size=self.temporal_embedding_size, hidden_size=self.encoder_size, num_layers=1)
        self.following_lstm = nn.LSTM(input_size=self.temporal_embedding_size, hidden_size=self.encoder_size, num_layers=1)
        
        # 动态嵌入层 - 分别处理前车和跟驰车
        self.leading_dyn_emb = nn.Linear(self.encoder_size, self.dynamics_encoding_size)
        self.following_dyn_emb = nn.Linear(self.encoder_size, self.dynamics_encoding_size)
        
        # 简化的社会交互层（替代复杂的卷积池化）
        self.social_interaction = nn.Sequential(
            nn.Linear(self.encoder_size, self.soc_conv_depth),
            self.leaky_relu,
            nn.Linear(self.soc_conv_depth, self.soc_conv2_depth),
            self.leaky_relu
        )
        
        # 跟驰类型分类器
        self.cf_type_classifier = nn.Linear(self.targ_enc_size, self.cf_type)
        
        # 解码器LSTM
        self.dec_lstm = nn.LSTM(
            input_size=self.targ_enc_size + self.cf_type,
            hidden_size=self.decoder_size,
            num_layers=1
        )
        
        # 输出层：直接输出位置预测
        self.position_output = nn.Linear(self.decoder_size, 1)
        
    def forward(self, hist):
        """
        前向传播
        Args:
            hist: [B, seq_len, features] 历史轨迹数据
                  features包括：前车速度、前车加速度、后车速度、后车加速度、车间间隙、车头间距、速度差、后车位置、前车位置、加速度差
        """
        batch_size = hist.shape[0]
        
        # 提取前车和跟驰车的特征
        # 前车特征：速度、加速度、位置
        leading_features = torch.stack([
            hist[:, :, 0],  # 前车速度
            hist[:, :, 1],  # 前车加速度
            hist[:, :, 8]   # 前车位置
        ], dim=2)  # [B, seq_len, 3]
        
        # 跟驰车特征：速度、加速度、位置  
        following_features = torch.stack([
            hist[:, :, 2],  # 后车速度
            hist[:, :, 3],  # 后车加速度
            hist[:, :, 7]   # 后车位置
        ], dim=2)  # [B, seq_len, 3]
        
        # 分别进行时间卷积编码
        leading_conv = self.leaky_relu(self.leading_temporalConv(leading_features.permute(0, 2, 1)))  # [B, temporal_embedding_size, seq_len]
        following_conv = self.leaky_relu(self.following_temporalConv(following_features.permute(0, 2, 1)))  # [B, temporal_embedding_size, seq_len]
        
        # 分别进行LSTM编码
        _, (leading_enc, _) = self.leading_lstm(leading_conv.permute(2, 0, 1))  # [1, B, encoder_size]
        _, (following_enc, _) = self.following_lstm(following_conv.permute(2, 0, 1))  # [1, B, encoder_size]
        
        # 提取最后的隐藏状态并进行动态嵌入
        leading_enc = leading_enc.view(leading_enc.shape[1], leading_enc.shape[2])  # [B, encoder_size]
        following_enc = following_enc.view(following_enc.shape[1], following_enc.shape[2])  # [B, encoder_size]
        
        leading_dyn = self.leaky_relu(self.leading_dyn_emb(leading_enc))  # [B, dynamics_encoding_size]
        following_dyn = self.leaky_relu(self.following_dyn_emb(following_enc))  # [B, dynamics_encoding_size]
        
        # 社会交互建模 - 使用简化的全连接层
        # 直接使用前车编码特征进行社会交互建模
        social_context = self.social_interaction(leading_enc)  # [B, soc_conv2_depth]
        
        # 连接社会上下文和跟驰车动态编码（目标车辆）
        target_enc = torch.cat((social_context, following_dyn), 1)  # [B, targ_enc_size]
        
        # 跟驰类型预测 - 基于社会交互和跟驰车特征
        cf_type_pred = self.cf_type_classifier(target_enc)
        cf_type_prob = self.softmax(cf_type_pred)
        
        # 解码器输入：社会交互特征 + 跟驰车动态特征 + 跟驰类型
        decoder_input = torch.cat([social_context, following_dyn, cf_type_prob], dim=1)
        
        # 扩展到序列长度
        decoder_input = decoder_input.unsqueeze(1).repeat(1, self.out_length, 1)
        
        # LSTM解码
        decoder_output, _ = self.dec_lstm(decoder_input)
        
        # 位置预测
        position_pred = self.position_output(decoder_output)  # [B, out_length, 1]
        
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