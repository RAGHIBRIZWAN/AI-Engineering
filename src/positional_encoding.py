"""
Positional Encoding modules for MiniGPT.

Transformers operate over sets of tokens without recurrence or convolution;
positional encodings inject order information into token representations.

Implements:
1. SinusoidalPositionalEncoding:
     PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
     PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
2. LearnedPositionalEncoding:
     Trainable lookup table mapping position indices to d_model vectors.
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed Sinusoidal Positional Encoding as described in 'Attention Is All You Need' (Vaswani et al., 2017).
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        """
        Args:
            d_model: Dimensionality of token representations.
            max_len: Maximum sequence length supported.
            dropout: Dropout probability applied to the sum of embeddings + positions.
        """
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Create constant position matrix pe: shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # shape: (max_len, 1)

        # Division term: 10000^(2i / d_model) = exp(2i * -log(10000.0) / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

        # Even indices: sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # Odd indices: cos
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register pe as a buffer (saved with state_dict, but not treated as a learnable parameter)
        # Shape: (1, max_len, d_model) for easy broadcasting over batch dimension
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to token embeddings.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encodings added, shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        # Slice precomputed positional encodings up to seq_len
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Embeddings (used in GPT-2).
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        """
        Args:
            d_model: Dimensionality of token representations.
            max_len: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned position embeddings to token embeddings.

        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with position embeddings added, shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        pos_embeddings = self.pos_emb(positions).unsqueeze(0) # shape: (1, seq_len, d_model)
        x = x + pos_embeddings
        return self.dropout(x)
