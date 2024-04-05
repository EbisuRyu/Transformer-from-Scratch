import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len = 500, dropout = 0.1):
        
        super(PositionalEncoding, self).__init__()
        
        # Parameters
        self.max_len = max_len
        self.d_model = d_model
        
        # Initialize positional embeddings
        self.position_embedding = torch.zeros(max_len, d_model)
        self.position_embedding.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.pe = self.generate_positional_encoding(self.d_model, self.max_len)

    
    def generate_positional_encoding(self, d_model, max_len):
        
        pos = torch.arange(0, max_len)
        pos = pos.to(torch.float32).unsqueeze(dim = 1)
        _2i = torch.arange(0, d_model, step = 2).to(torch.float32)
        self.position_embedding[:, 0::2] = torch.sin(pos / torch.pow(10000, (_2i / d_model))).to(torch.float32)
        self.position_embedding[:, 1::2] = torch.cos(pos / torch.pow(10000, (_2i / d_model))).to(torch.float32)
        self.position_embedding = self.position_embedding.unsqueeze(0)
        return self.position_embedding
                                            
        
    def forward(self, embedding_batch):
        # Generate positional encodings
        seq_length = embedding_batch.size(1)
        pe_batch = self.pe[:, :seq_length].to(embedding_batch.device)
        return self.dropout(embedding_batch + pe_batch)