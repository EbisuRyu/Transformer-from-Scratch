import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
  
  def __init__(self, d_head):
    super(ScaledDotProductAttention, self).__init__()
    self.d_head = d_head
    self.attention_dropout = nn.Dropout(0.1)

  def forward(self, query, key, value, mask = None):

    # Compute scaled dot product attention
    attention_weight = torch.einsum('bhqe, bhke -> bhqk', query, key)
    scaled_attention_weight = attention_weight / math.sqrt(self.d_head)
    # Apply masking if provided
    if mask is not None:
      scaled_attention_weight = scaled_attention_weight.masked_fill(mask == 0, float('-inf'))

    # Apply softmax to normalize attention weights
    attention_score = nn.functional.softmax(scaled_attention_weight, dim = -1)
    attention_score = self.attention_dropout(attention_score)
    # Contextual weighted sum using attention scores
    weighted_value = torch.einsum('bhlv, bhve -> bhle', attention_score, value)

    return weighted_value, attention_score


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        assert (self.d_head * self.n_heads == self.d_model), 'Embeding size needs to be div by n_heads'
        
        
        self.query_embed_layer = nn.Linear(self.d_model, self.d_model, bias = True)
        self.key_embed_layer = nn.Linear(self.d_model, self.d_model, bias = True)
        self.value_embed_layer = nn.Linear(self.d_model, self.d_model, bias = True)
        
        
        self.scaled_dot_product_attention = ScaledDotProductAttention(self.d_head)
        self.fc_out = nn.Linear(self.d_model, self.d_model)
        
    def _split_into_heads(self, query, key, value):
        batch_size = query.shape[0]
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]
        query = query.reshape(batch_size, query_len, self.n_heads, self.d_head)
        key = key.reshape(batch_size, key_len, self.n_heads, self.d_head)
        value = value.reshape(batch_size, value_len, self.n_heads, self.d_head)
        
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        return query, key, value
    
    def _concatenate_heads(self, attention_output):
        attention_output = attention_output.transpose(1, 2)
        batch_size, seq_len = attention_output.shape[0 : 2]
        attention_output = attention_output.reshape(batch_size, seq_len, -1)
        return attention_output
    
    def forward(self, query, key, value, mask = None):

        # values shape: (value_len, n_heads, d_head)
        # keys shape: (key_len, n_heads, d_head)
        # queries shape: (N, query_len, n_heads, d_head)
        
        value_embeddings = self.value_embed_layer(value)
        key_embeddings = self.key_embed_layer(key)
        query_embeddings = self.query_embed_layer(query)
        head_query_embeddings, head_key_embeddings, head_value_embeddings = self._split_into_heads(query_embeddings, key_embeddings, value_embeddings)
        # head_value_embeddings shape: (batch, n_heads, value_len, d_head)
        # head_key_embeddings shape: (batch, n_heads, key_len, d_head)
        # head_query_embeddings shape: (batch, n_heads, query_len, d_head)
        
        attention_output, _ = self.scaled_dot_product_attention(head_query_embeddings, head_key_embeddings, head_value_embeddings, mask = mask)
        # print(attention_output.shape)
        
        concatenated_attention_output = self._concatenate_heads(attention_output)
        # print(output.shape)
        attention_output = self.fc_out(concatenated_attention_output)
        # print(output.shape)
        return attention_output