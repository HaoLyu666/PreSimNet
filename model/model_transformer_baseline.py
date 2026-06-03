import torch
import torch as t
from torch import nn
import torch.nn.functional as F
from module.module import BERTTimeEmbedding, MLP
from torch import Tensor
from typing import Optional


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.device = args['device']
        self.encoder_size = args['encoder_size']
        self.n_head = args['n_head']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.f_length = args['f_length']
        self.dropout = args['dropout']
        self.transformer_layer = args['transformer_layer']
        self.cf_type = args['cf_type']
        
        # 输入特征编码
        self.encoder = nn.Linear(self.f_length, self.encoder_size)
        
        # 位置编码
        self.pos_encoder = BERTTimeEmbedding(
            max_position_embeddings=self.in_length, 
            embedding_dim=self.encoder_size
        )
        
        # 跟驰类型查询向量
        self.type_query = nn.Parameter(t.Tensor(1, 1, self.encoder_size))
        nn.init.xavier_uniform_(self.type_query, gain=1.414)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.encoder_size,
            nhead=self.n_head,
            dim_feedforward=self.encoder_size * 4, 
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.transformer_layer
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
        # 编码历史轨迹
        hist_enc = self.relu(self.encoder(hist))
        
        # 添加位置编码
        hist_enc = hist_enc + self.pos_encoder(hist_enc)
        
        # 添加类型查询并通过Transformer
        hist_att_in = t.cat([hist_enc, self.type_query.repeat([hist_enc.shape[0], 1, 1])], dim=1)
        hist_att = self.transformer_encoder(hist_att_in)
        
        # 预测跟驰类型
        cf_type_pred = self.type_pred(hist_att[:, -1, :])  # 未经softmax的logits
        cf_type_probs = F.softmax(cf_type_pred, dim=-1)  # 转换为概率
        
        # 返回编码结果和跟驰类型预测
        return hist_enc, cf_type_pred, cf_type_probs


class TransformerDecoder(nn.Module):
    def __init__(self, args):
        super(TransformerDecoder, self).__init__()
        self.encoder_size = args['encoder_size']
        self.n_head = args['n_head']
        self.out_length = args['out_length']
        self.dropout = args['dropout']
        self.transformer_layer = args['transformer_layer']
        self.cf_type = args['cf_type']
        
        # 相对位置编码
        self.rel_pos_encoder = BERTTimeEmbedding(
            max_position_embeddings=self.out_length, 
            embedding_dim=self.encoder_size
        )
        
        # 初始查询向量
        self.query_embed = nn.Parameter(t.randn(1, self.out_length, self.encoder_size))
        nn.init.xavier_uniform_(self.query_embed)
        
        # 特征融合层 - 融合跟驰类型概率向量
        self.fusion_layer = nn.Linear(self.encoder_size + self.cf_type, self.encoder_size)
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.encoder_size,
            nhead=self.n_head,
            dim_feedforward=self.encoder_size * 4,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=self.transformer_layer
        )
        
        # 输出投影层
        self.output_proj = nn.Linear(self.encoder_size, 1)
        
        # 归一化层
        self.layer_norm = nn.LayerNorm(self.encoder_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, memory, cf_type_probs):
        batch_size = memory.shape[0]
        
        # 准备查询向量
        query = self.query_embed.repeat(batch_size, 1, 1)
        
        # 添加相对位置编码
        query = query + self.rel_pos_encoder(query)
        
        # # 融合跟驰类型概率到查询
        # cf_probs_expanded = cf_type_probs.unsqueeze(1).repeat(1, self.out_length, 1)  # [B, out_length, cf_type]
        # query_with_cf = t.cat([query, cf_probs_expanded], dim=-1)
        # query = self.fusion_layer(query_with_cf)
        # query = F.gelu(query)
        # query = self.dropout_layer(query)
        
        # 生成因果掩码
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            self.out_length, device=query.device
        )
        
        # 通过Transformer解码器
        decoder_output = self.transformer_decoder(
            query, 
            memory,
            tgt_mask=tgt_mask
        )
        
        # 投影到位置预测
        position_pred = self.output_proj(decoder_output)
        position_pred = F.relu(position_pred)  # 确保位置是正的
        
        return position_pred


class TransformerBaseline(nn.Module):
    def __init__(self, args):
        super(TransformerBaseline, self).__init__()
        args['transformer_layer'] = 2
        self.encoder = TransformerEncoder(args)
        self.decoder = TransformerDecoder(args)
        
    def forward(self, hist):
        # 编码历史轨迹
        memory, cf_type_pred, cf_type_probs = self.encoder(hist)
        
        # 解码预测未来轨迹
        position_pred = self.decoder(memory, cf_type_probs)
        
        return {
            'position_pred': position_pred,
            'cf_type_pred': cf_type_pred
        }


Seq2SeqBaseline = TransformerBaseline


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
