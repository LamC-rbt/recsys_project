import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout_rate: float = 0.5):
        super().__init__()
        self.num_heads = num_heads

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.val_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)  # Change the dropout rate as needed

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, causality: bool = False):
        # Linear projections
        Q = self.query_proj(queries)
        K = self.key_proj(keys)
        V = self.val_proj(keys)

        # Split and concatenate along head dimension
        Q_ = torch.cat(Q.chunk(self.num_heads, dim=2), dim=0)
        K_ = torch.cat(K.chunk(self.num_heads, dim=2), dim=0)
        V_ = torch.cat(V.chunk(self.num_heads, dim=2), dim=0)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q_, K_.transpose(1, 2))
        attn_scores = attn_scores / (K_.size(-1) ** 0.5)

        # Key masking
        key_masks = torch.sign(torch.sum(torch.abs(keys), dim=-1))
        key_masks = key_masks.repeat(self.num_heads, 1)
        key_masks = key_masks.unsqueeze(1).repeat(1, queries.size(1), 1)
        attn_scores = attn_scores.masked_fill(key_masks == 0, float('-inf'))

        # Apply causal masking if needed
        if causality:
            diag_vals = torch.ones_like(attn_scores[0])
            tril = torch.tril(diag_vals)
            masks = tril.unsqueeze(0).repeat(attn_scores.size(0), 1, 1)
            attn_scores = attn_scores.masked_fill(masks == 0, float('-inf'))

        # Softmax normalization
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)

        # Query masking
        query_masks = torch.sign(torch.sum(torch.abs(queries), dim=-1))
        query_masks = query_masks.repeat(self.num_heads, 1)
        query_masks = query_masks.unsqueeze(-1).repeat(1, 1, keys.size(1))
        attn_weights *= query_masks

        # Attention weights (for visualization or diagnostics)
        attn_chunks = attn_weights.chunk(self.num_heads, dim=0)
        attention_weights = torch.stack(attn_chunks, dim=1)

        # Apply dropout
        attn_weights = self.dropout(attn_weights)

        # Weighted sum
        outputs = torch.matmul(attn_weights, V_)

        # Merge heads
        outputs = torch.cat(outputs.chunk(self.num_heads, dim=0), dim=2)
        return outputs, attention_weights


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, hidden_dim: int, dropout_rate: float = 0.5, causality: bool = True):
        super().__init__()
        self.first_norm = nn.LayerNorm(dim)
        self.second_norm = nn.LayerNorm(dim)

        self.multihead_attention = MultiHeadAttention(dim, num_heads, dropout_rate)

        self.dense1 = nn.Linear(dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.causality = causality

    def forward(self, seq: torch.Tensor, mask: torch.Tensor = None):
        # Layer norm before attention
        x = self.first_norm(seq)
        queries = x
        keys = seq
        # Multi-head attention
        x, attentions = self.multihead_attention(queries, keys, self.causality)

        # Add & Norm
        x = x + queries
        x = self.second_norm(x)

        # Feed-forward network
        residual = x
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)

        # Add & Norm
        x = x + residual

        # Apply mask if provided
        if mask is not None:
            x *= mask
        return x, attentions