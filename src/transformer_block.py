"""
Transformer Decoder Block for MiniGPT.

Assembles:
1. Multi-Head Self-Attention with Causal Masking.
2. Feed-Forward Network (FFN): Linear -> GELU -> Dropout -> Linear -> Dropout.
3. Pre-Layer Normalization (GPT-2 style): Stabilizes gradient flow.
4. Residual Connections around both sublayers: x = x + Sublayer(LN(x)).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from src.attention import MultiHeadAttention


class FeedForward(nn.Module):
    """
    Two-layer Position-wise Feed-Forward Network with GELU activation.
    
    Formula:
        FFN(x) = Linear2(Dropout(GELU(Linear1(x))))
    """

    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder block featuring pre-LayerNorm and residual connections.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        max_context_length: int = 1024,
        dropout: float = 0.1,
        bias: bool = False,
        causal: bool = True,
    ):
        """
        Args:
            d_model: Embedding dimension.
            n_heads: Number of attention heads.
            ffn_dim: Hidden dimension of the feed-forward network (typically 4 * d_model).
            max_context_length: Maximum sequence length.
            dropout: Dropout rate.
            bias: Whether linear layers use bias.
            causal: Whether attention should be causal (autoregressive).
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_context_length=max_context_length,
            dropout=dropout,
            bias=bias,
            causal=causal,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model,
            ffn_dim=ffn_dim,
            dropout=dropout,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            return_attention: Whether to return attention weights.

        Returns:
            out: Output tensor of shape (batch_size, seq_len, d_model).
            attn_weights: Attention weights of shape (batch_size, n_heads, seq_len, seq_len) if requested.
        """
        # Sublayer 1: Multi-Head Self-Attention with Pre-LayerNorm & Residual Connection
        normed_x1 = self.ln_1(x)
        attn_out, attn_weights = self.attn(normed_x1, return_attention=return_attention)
        x = x + attn_out

        # Sublayer 2: Feed-Forward Network with Pre-LayerNorm & Residual Connection
        normed_x2 = self.ln_2(x)
        ffn_out = self.ffn(normed_x2)
        x = x + ffn_out

        if return_attention:
            return x, attn_weights
        return x, None
