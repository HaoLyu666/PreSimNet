import torch
import torch as t
from torch import nn
import torch.nn.functional as F
from module.module import BERTTimeEmbedding, MLP
from torch import Tensor
from typing import Optional


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.device = args['device']
        self.encoder_size = args['encoder_size']
        self.n_head = args['n_head']
        self.in_length = args['in_length']
        self.f_length = args['f_length']
        self.dropout = args['dropout']
        self.cf_type = args['cf_type']
        self.num_layers = args['transformer_layer']
        
        # 输入特征编码
        self.encoder = nn.Linear(self.f_length, self.encoder_size)
        
        # 位置编码
        self.pos_encoder = BERTTimeEmbedding(
            max_position_embeddings=self.in_length, 
            embedding_dim=self.encoder_size
        )
        
        # 单向GRU编码器
        self.gru = nn.GRU(
            input_size=self.f_length,
            hidden_size=self.encoder_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=False  # 改为单向GRU
        )
        
        # 跟驰类型预测头
        self.type_pred = MLP(
            f_in=self.encoder_size,
            f_out=self.cf_type,
            activation='tanh',
            hidden_dim=self.encoder_size // 2,
            hidden_layers=3,
            dropout=self.dropout
        )
        
        # 归一化层
        self.layer_norm = nn.LayerNorm(self.encoder_size)
        
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, hist):
        batch_size = hist.shape[0]
        
        # 编码历史轨迹
        # hist_enc = self.relu(self.encoder(hist))
        
        # 添加位置编码
        # hist_enc = hist_enc + self.pos_encoder(hist_enc)
        
        # 通过GRU编码器
        outputs, hidden = self.gru(hist)
        
        # 获取最后一个隐藏状态用于跟驰类型预测
        # 对于单向GRU，直接使用最后一层的隐藏状态
        last_hidden = hidden[-1]  # [batch_size, encoder_size]
        
        # 预测跟驰类型，只使用GRU最后一个特征
        cf_type_pred = self.type_pred(last_hidden)  # 未经softmax的logits
        cf_type_probs = F.softmax(cf_type_pred, dim=-1)  # 转换为概率
        
        # 返回编码结果和跟驰类型预测
        return outputs, hidden, cf_type_pred, cf_type_probs


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.encoder_size = args['encoder_size']
        self.n_head = args['n_head']
        self.out_length = args['out_length']
        self.dropout = args['dropout']
        self.num_layers = args['transformer_layer']
        
        # 相对位置编码
        self.rel_pos_encoder = BERTTimeEmbedding(
            max_position_embeddings=self.out_length, 
            embedding_dim=self.encoder_size
        )
        
        # 初始查询向量
        self.query_embed = nn.Parameter(t.randn(1, 1, self.encoder_size))
        nn.init.xavier_uniform_(self.query_embed)
        
        # GRU解码器
        self.gru = nn.GRU(
            input_size=self.encoder_size,
            hidden_size=self.encoder_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=self.encoder_size,
            num_heads=self.n_head,
            dropout=self.dropout,
            batch_first=True
        )
        
        # 注意力输出融合
        self.attn_combine = nn.Linear(self.encoder_size * 2, self.encoder_size)
        
        # 输出投影层
        self.output_proj = nn.Linear(self.encoder_size, 1)
        
        # 归一化层
        self.layer_norm = nn.LayerNorm(self.encoder_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward_step(self, input_tensor, hidden, encoder_outputs):
        # 通过GRU单步解码
        output, hidden = self.gru(input_tensor, hidden)
        
        # 使用注意力机制关注编码器输出
        attn_output, _ = self.attention(output, encoder_outputs, encoder_outputs)
        
        # 融合注意力输出和GRU输出
        output = t.cat((output, attn_output), dim=2)
        output = self.attn_combine(output)
        output = F.relu(output)
        
        return output, hidden

    def forward(self, encoder_outputs, encoder_hidden, cf_type_probs=None):
        batch_size = encoder_outputs.shape[0]
        
        # 初始化解码器输入和隐藏状态
        decoder_input = self.query_embed.repeat(batch_size, 1, 1)
        
        # 添加相对位置编码
        pos_queries = t.zeros(batch_size, self.out_length, self.encoder_size, device=decoder_input.device)
        pos_queries = self.rel_pos_encoder(pos_queries)
        
        # 使用编码器的隐藏状态初始化解码器隐藏状态
        decoder_hidden = encoder_hidden
        
        # 存储所有输出
        all_outputs = []
        
        # # 自回归解码
        # for i in range(self.out_length):
        #     # 为当前步骤添加位置编码
        #     if i > 0:
        #         decoder_input = decoder_input + (pos_queries[:, i:i+1, :] - pos_queries[:, i-1:i, :])
        #     else:
        #         decoder_input = decoder_input + pos_queries[:, 0:1, :]
            
        #     # 单步解码
        #     decoder_output, decoder_hidden = self.forward_step(
        #         decoder_input, decoder_hidden, encoder_outputs
        #     )
            
        #     # 存储输出
        #     all_outputs.append(decoder_output)
            
        #     # 更新下一步的输入
        #     decoder_input = decoder_output
        
        # 堆叠所有输出
        decoder_outputs, _ = self.gru(pos_queries, decoder_hidden)  # [B, out_length, encoder_size]
        
        # 投影到位置预测
        position_pred = self.output_proj(decoder_outputs)
        position_pred = F.relu(position_pred)  # 确保位置是正的
        
        return position_pred


class Seq2SeqBaseline(nn.Module):
    def __init__(self, args):
        super(Seq2SeqBaseline, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        
    def forward(self, hist):
        # 编码历史轨迹
        encoder_outputs, encoder_hidden, cf_type_pred, cf_type_probs = self.encoder(hist)
        
        # 解码预测未来轨迹，不使用跟驰类别
        position_pred = self.decoder(encoder_outputs, encoder_hidden)
        
        return {
            'position_pred': position_pred,
            'cf_type_pred': cf_type_pred
        }


class predictor:
    def __init__(self, args):
        self.args = args
        self.device = args['device']
        self.out_length = args['out_length']
        self.dt = args['time_step']
        
    def forward(self, model_outputs, nextv, veh_state):
        """
        简单地返回模型预测的位置
        
        Args:
            model_outputs: 模型输出字典，包含位置预测和跟驰类型预测
            nextv: 前车未来速度 [B, out_length]
            veh_state: 初始车辆状态 [B, 2] (间隙, 速度)
        """
        # 提取位置预测
        position_pred = model_outputs['position_pred']  # [B, out_length, 1]
        position_flat = position_pred.squeeze(-1)  # [B, out_length]
        
        # 构建完整的预测结果 [B, out_length, 2] (间隙, 速度)
        # 注意：我们不预测速度，但为了与原始评估代码兼容，我们需要返回一个占位符
        # 使用零填充速度维度
        velocities = t.zeros_like(position_flat)  # [B, out_length]
        
        # 构建完整的预测结果
        PS = t.cat([position_flat.unsqueeze(-1), velocities.unsqueeze(-1)], dim=-1)
        
        return {
            'dynamic_pred': PS,  # 包含间隙和速度(零)的预测
            'direct_pred': position_pred  # 直接位置预测
        }