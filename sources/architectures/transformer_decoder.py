import torch
import torch.nn as nn 

from .multihead_attention import *
from .transformer_embedding import *
from .position_wise_feed_forward import *

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.dropout_layer_1 = nn.Dropout(dropout)
        self.norm_layer_1 = nn.LayerNorm(d_model)

        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        self.dropout_layer_2 = nn.Dropout(dropout)
        self.norm_layer_2 = nn.LayerNorm(d_model)

        self.position_wise_feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.dropout_layer_3 = nn.Dropout(dropout)
        self.norm_layer_3 = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_output, src_mask, trg_mask):
        # Self Attention
        self_attention_out = self.self_attention(x, x, x, trg_mask)
        residual_1 = self.dropout_layer_1(self_attention_out)
        self_attention_out = self.norm_layer_1(x + residual_1)

        # Cross Attention
        cross_attention_out = self.cross_attention(x, enc_output, enc_output, src_mask)
        residual_2 = self.dropout_layer_2(cross_attention_out)
        cross_attention_out = self.norm_layer_2(self_attention_out + residual_2)

        # Positionwise Feed Forward
        ffn_out = self.position_wise_feed_forward(cross_attention_out)
        residual_3 = self.dropout_layer_3(ffn_out)
        dec_out = self.norm_layer_3(cross_attention_out + residual_3)

        return dec_out
    
class Decoder(nn.Module):
    def __init__(self, num_layers, n_heads, d_model, d_ff, dec_vocab_size, max_len, dropout):
        super(Decoder, self).__init__()
        self.trg_embedding_layer = TransformerEmbedding(d_model, dec_vocab_size, max_len, dropout)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, enc_output, src_mask, trg_mask):
        x = self.trg_embedding_layer(x)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, enc_output, src_mask, trg_mask)
        return x