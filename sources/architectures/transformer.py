import torch
import torch.nn as nn

from .transformer_encoder import *
from .transformer_decoder import *

class Transformer(nn.Module):
    
    def __init__(self, d_model, num_layers, src_vocab_size, trg_vocab_size, n_heads, d_ff,  max_len, dropout = 0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, n_heads,  d_model, d_ff, src_vocab_size, max_len, dropout)
        self.decoder = Decoder(num_layers, n_heads,  d_model, d_ff, trg_vocab_size, max_len, dropout)
        self.projection_layer = nn.Linear(d_model, trg_vocab_size)
        
        # Switch to xavier initialization (shown to be beneficial)
        self.init_with_xavier()
        
        # Share the embedding layer weights between encoder and decoder
        self.encoder.src_embedding_layer.token_embedding.weight = self.decoder.trg_embedding_layer.token_embedding.weight
        self.projection_layer.weight = self.decoder.trg_embedding_layer.token_embedding.weight
        
    def encode(self, src_token_ids, src_mask):
        return self.encoder(src_token_ids, src_mask)
    
    def decode(self, trg_token_ids, enc_output, src_mask, trg_mask):
        return self.decoder(trg_token_ids, enc_output, src_mask, trg_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
    def init_with_xavier(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)