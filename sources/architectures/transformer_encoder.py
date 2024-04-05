import torch
import torch.nn as nn

from .multihead_attention import MultiHeadAttention
from .transformer_embedding import TransformerEmbedding
from .position_wise_feed_forward import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout = 0.1):
        super(EncoderLayer, self).__init__()
        self.multihead_attention = MultiHeadAttention(d_model, n_heads)
        self.dropout_layer_1 = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        
        self.position_wise_feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.dropout_layer_2 = nn.Dropout(dropout)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        
        
    def forward(self, x, src_mask = None):
        
        mha_out = self.multihead_attention(x, x, x, src_mask)
        residual_1 = self.dropout_layer_1(mha_out)
        x = self.layer_norm_1(residual_1 + x)
        
        ffn_out = self.position_wise_feed_forward(x)
        residual_2 = self.dropout_layer_2(ffn_out)
        x = self.layer_norm_2(residual_2 + x)
        return x


class Encoder(nn.Module):
    
    def __init__(self, num_layers, n_heads, d_model, d_ff, enc_vocab_size, max_len, dropout = 0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        
        self.src_embedding_layer = TransformerEmbedding(d_model, enc_vocab_size, max_len, dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        
    def forward(self, x, src_mask = None):
        x = self.src_embedding_layer(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_mask)
        return x
