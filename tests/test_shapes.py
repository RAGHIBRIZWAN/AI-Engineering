"""
Shape verification and tensor-trace tests for all MiniGPT components.
"""

import pytest
import torch

from src.positional_encoding import SinusoidalPositionalEncoding, LearnedPositionalEncoding
from src.attention import MultiHeadAttention
from src.transformer_block import TransformerBlock, FeedForward
from src.gpt import MiniGPT


def test_positional_encoding_shapes():
    """Verify sinusoidal and learned positional encodings preserve input shapes."""
    B, T, d_model = 2, 32, 64
    x = torch.randn(B, T, d_model)

    sin_pe = SinusoidalPositionalEncoding(d_model=d_model, max_len=128, dropout=0.0)
    learned_pe = LearnedPositionalEncoding(d_model=d_model, max_len=128, dropout=0.0)

    out_sin = sin_pe(x)
    out_learned = learned_pe(x)

    assert out_sin.shape == (B, T, d_model)
    assert out_learned.shape == (B, T, d_model)


def test_feedforward_shapes():
    """Verify FeedForward network shapes and dimension expansion/projection."""
    B, T, d_model, ffn_dim = 2, 16, 64, 256
    ffn = FeedForward(d_model=d_model, ffn_dim=ffn_dim, dropout=0.0)
    x = torch.randn(B, T, d_model)
    out = ffn(x)

    assert out.shape == (B, T, d_model)


def test_transformer_block_shapes():
    """Verify single TransformerBlock input/output shape preservation."""
    B, T, d_model, n_heads, ffn_dim = 3, 20, 64, 4, 256
    block = TransformerBlock(d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim, max_context_length=64)
    x = torch.randn(B, T, d_model)
    out, attn = block(x, return_attention=True)

    assert out.shape == (B, T, d_model)
    assert attn.shape == (B, n_heads, T, T)


def test_minigpt_end_to_end_shapes():
    """Verify full MiniGPT forward pass produces expected logit and loss shapes."""
    batch_size = 4
    seq_len = 32
    vocab_size = 65
    d_model = 128
    n_heads = 4
    n_layers = 2

    model = MiniGPT(
        vocab_size=vocab_size,
        context_length=seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    )

    tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits, loss, attentions = model(tokens, targets=targets, return_attentions=True)

    # Logits shape: (batch_size, seq_len, vocab_size)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    # Loss is a scalar
    assert loss.dim() == 0 and not torch.isnan(loss)
    # Attentions list length matches n_layers
    assert len(attentions) == n_layers
    # Each attention map has shape (batch_size, n_heads, seq_len, seq_len)
    assert attentions[0].shape == (batch_size, n_heads, seq_len, seq_len)


if __name__ == "__main__":
    pytest.main([__file__])
