"""
Attention mechanisms for MiniGPT: Scaled Dot-Product Attention & Multi-Head Attention.

Implements:
1. Scaled Dot-Product Attention:
     Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) + mask ) @ V
2. Multi-Head Attention (MHA):
     Splitting into multiple heads, computing attention per head,
     concatenating results, and projecting through a linear layer.
3. Causal Masking:
     Upper-triangular future-token mask preventing tokens from attending to future tokens.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

    Args:
        q: Query tensor of shape (..., seq_len_q, d_k)
        k: Key tensor of shape (..., seq_len_k, d_k)
        v: Value tensor of shape (..., seq_len_v, d_v)
        mask: Optional mask tensor broadcastable to (..., seq_len_q, seq_len_k).
              Positions with 0 (or False) will receive -infinity before softmax.
        dropout_p: Dropout probability applied to attention weights.
        training: Whether currently in training mode (for dropout).

    Returns:
        output: Weighted sum of values, shape (..., seq_len_q, d_v)
        weights: Attention weights after softmax, shape (..., seq_len_q, seq_len_k)
    """
    d_k = q.size(-1)

    # 1. Compute raw similarity scores: (Q @ K^T)
    # Shape: (..., seq_len_q, seq_len_k)
    scores = torch.matmul(q, k.transpose(-2, -1))

    # 2. Scale by square root of d_k to prevent vanishing gradients for large dimensions
    scores = scores / math.sqrt(d_k)

    # 3. Apply mask if provided (replace forbidden positions with large negative number)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # 4. Softmax along the last dimension (each query attends across keys)
    weights = F.softmax(scores, dim=-1)

    # 5. Apply dropout to attention weights (during training only)
    if dropout_p > 0.0 and training:
        weights = F.dropout(weights, p=dropout_p)

    # 6. Compute weighted sum of values: weights @ V
    # Shape: (..., seq_len_q, d_v)
    output = torch.matmul(weights, v)

    return output, weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module with explicit splitting, attention, and merging.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_context_length: int = 1024,
        dropout: float = 0.1,
        bias: bool = False,
        causal: bool = True,
    ):
        """
        Args:
            d_model: Total embedding dimension.
            n_heads: Number of parallel attention heads.
            max_context_length: Maximum context length for causal mask buffer.
            dropout: Dropout probability.
            bias: Whether linear layers include bias.
            causal: Whether to apply autoregressive causal masking.
        """
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        self.dropout_p = dropout

        # Explicit linear projections for Query, Key, and Value
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        # Final output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: lower-triangular matrix of 1s (shape: 1, 1, max_len, max_len)
        # Position (i, j) is 1 if j <= i (past/present), and 0 if j > i (future)
        if causal:
            mask = torch.tril(torch.ones(max_context_length, max_context_length))
            self.register_buffer("causal_mask", mask.view(1, 1, max_context_length, max_context_length))
        else:
            self.causal_mask = None

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for Multi-Head Attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            return_attention: If True, returns attention weights for inspection.

        Returns:
            out: Projected output of shape (batch_size, seq_len, d_model).
            attn_weights: Attention weights of shape (batch_size, n_heads, seq_len, seq_len) if requested.
        """
        batch_size, seq_len, _ = x.size()

        # Step 1: Linear projections
        # Each has shape: (batch_size, seq_len, d_model)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Step 2: Split heads and transpose
        # Shape: (batch_size, seq_len, n_heads, d_k) -> (batch_size, n_heads, seq_len, d_k)
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Step 3: Slice causal mask for the current sequence length
        mask = None
        if self.causal and self.causal_mask is not None:
            mask = self.causal_mask[:, :, :seq_len, :seq_len]

        # Step 4: Scaled dot-product attention per head
        # out shape: (batch_size, n_heads, seq_len, d_k)
        # attn_weights shape: (batch_size, n_heads, seq_len, seq_len)
        out, attn_weights = scaled_dot_product_attention(
            q=q,
            k=k,
            v=v,
            mask=mask,
            dropout_p=self.dropout_p,
            training=self.training,
        )

        # Step 5: Merge/concatenate heads back
        # Transpose: (batch_size, seq_len, n_heads, d_k)
        # Contiguous view: (batch_size, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Step 6: Final linear projection and residual dropout
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        if return_attention:
            return out, attn_weights
        return out, None
