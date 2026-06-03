from __future__ import division
import torch
from torch.autograd import Variable
import torch.nn as nn

def outputActivation(x):
    """
    简化的输出激活函数，适用于跟驰轨迹预测
    """
    # 对于跟驰任务，我们只需要位置预测，不需要复杂的概率分布
    return x

class CarFollowingNet(nn.Module):
    """跟驰轨迹预测网络"""
    
    def __init__(self, args):
        super(CarFollowingNet, self).__init__()

        
        ## 网络层大小
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['encoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.cf_type = args['cf_type']  # 跟驰类型数量
        
        # 嵌入层大小
        self.input_embedding_size = args['encoder_size']
        self.dyn_embedding_size = args['encoder_size']
        self.social_embedding_size = args['encoder_size']
        
        ## 定义网络层
        
        # 输入嵌入层 - 分别处理前车和跟驰车
        self.leading_emb = torch.nn.Linear(3, self.input_embedding_size)  # 前车：速度、加速度、位置
        self.following_emb = torch.nn.Linear(3, self.input_embedding_size)  # 跟驰车：速度、加速度、位置
        
        # 编码器LSTM - 分别编码前车和跟驰车的历史轨迹
        self.leading_lstm = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 2, batch_first=True)
        self.following_lstm = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 2, batch_first=True)
        
        # 车辆动态嵌入
        self.leading_dyn_emb = torch.nn.Linear(self.encoder_size, self.dyn_embedding_size)
        self.following_dyn_emb = torch.nn.Linear(self.encoder_size, self.dyn_embedding_size)
        
        # 社会交互层 - 简化的交互建模
        self.social_interaction = torch.nn.Sequential(
            torch.nn.Linear(self.dyn_embedding_size * 2, self.social_embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.social_embedding_size, self.social_embedding_size)
        )
        
        # 跟驰类型分类器
        # 跟驰类型预测网络
        hidden_size = self.encoder_size // 2
        self.mu_fc1 = nn.Linear(self.social_embedding_size + self.dyn_embedding_size, hidden_size)
        self.mu_fc = nn.Linear(hidden_size, self.social_embedding_size + self.dyn_embedding_size)
        self.cf_type_classifier = torch.nn.Linear(
            self.social_embedding_size + self.dyn_embedding_size, 
            self.cf_type
        )
        
        # 解码器LSTM
        self.dec_lstm = torch.nn.LSTM(
            self.social_embedding_size + self.dyn_embedding_size + self.cf_type, 
            self.decoder_size, 
            2,
            batch_first=True)
        
        # 输出层
        self.position_output = torch.nn.Linear(self.decoder_size, 1)  # 位置预测
        
        # 激活函数
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        # 激活函数和归一化
        self.activation = nn.LeakyReLU(0.1)
        self.normalize = nn.LayerNorm(self.social_embedding_size + self.dyn_embedding_size)
        
    def forward(self, hist):
        """
        前向传播
        Args:
            hist: [B, seq_len, features] 历史轨迹数据
                  features包括：前车速度、前车加速度、后车速度、后车加速度、车间间隙、车头间距、速度差、后车位置、前车位置、加速度差
        """
        batch_size = hist.shape[0]
        
        # 提取前车和跟驰车的特征
        # 根据loader2.py中的数据格式：
        # hist[:, :, 0]: 前车速度
        # hist[:, :, 1]: 前车加速度  
        # hist[:, :, 2]: 后车速度
        # hist[:, :, 3]: 后车加速度
        # hist[:, :, 4]: 车间间隙
        # hist[:, :, 5]: 车头间距
        # hist[:, :, 6]: 速度差
        # hist[:, :, 7]: 后车位置
        # hist[:, :, 8]: 前车位置
        # hist[:, :, 9]: 加速度差
        
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
        
        # 嵌入层
        leading_emb = self.leaky_relu(self.leading_emb(leading_features))
        following_emb = self.leaky_relu(self.following_emb(following_features))
        
        # LSTM编码
        _, (leading_enc, _) = self.leading_lstm(leading_emb)
        _, (following_enc, _) = self.following_lstm(following_emb)
        
        # 提取最后的隐藏状态
        leading_enc = leading_enc[0]  # [B, encoder_size]
        following_enc = following_enc[0]   # [B, encoder_size]
        
        # 动态嵌入
        leading_dyn = self.leaky_relu(self.leading_dyn_emb(leading_enc))
        following_dyn = self.leaky_relu(self.following_dyn_emb(following_enc))
        
        # 社会交互建模
        interaction_input = torch.cat([leading_dyn, following_dyn], dim=1)
        social_enc = self.leaky_relu(self.social_interaction(interaction_input))
        
        # 组合特征用于分类和预测
        combined_features = torch.cat([social_enc, following_dyn], dim=1)
        
        # 跟驰类型预测
        maneuver_state = self.activation(self.mu_fc1(combined_features))
        maneuver_state = self.activation(self.normalize(self.mu_fc(maneuver_state)))
        cf_type_pred = self.cf_type_classifier(maneuver_state)
        

        cf_type_prob = self.softmax(cf_type_pred)
        
        # 解码器输入：社会交互特征 + 跟驰车动态特征 + 跟驰类型
        decoder_input = torch.cat([social_enc, following_dyn, cf_type_prob], dim=1)
        
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