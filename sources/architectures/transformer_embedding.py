import torch
import torch.nn as nn
import math
from .positional_encoding import *

class TransformerEmbedding(nn.Module):
    
    def __init__(self, d_model, vocab_size, max_len, dropout = 0.1):
        super(TransformerEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        self.token_embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        token_embedding = self.token_embedding(x) * math.sqrt(self.d_model)
        positional_encoding = self.positional_encoding(token_embedding)
        output = self.dropout(positional_encoding)
        return output