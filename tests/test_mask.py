"""
Unit tests for Causal Masking correctness and future-token leakage prevention.
"""

import pytest
import torch

from src.attention import MultiHeadAttention, scaled_dot_product_attention
from src.gpt import MiniGPT


def test_causal_mask_zeros_future_weights():
    """
    Verify that in causal self-attention, attention weights for future tokens (j > i) are exactly 0.0.
    """
    B, T, d_k = 1, 6, 16
    q = torch.randn(B, T, d_k)
    k = torch.randn(B, T, d_k)
    v = torch.randn(B, T, d_k)

    # Lower triangular mask: 1 where allowed, 0 where forbidden
    mask = torch.tril(torch.ones(T, T)).unsqueeze(0)

    _, weights = scaled_dot_product_attention(q, k, v, mask=mask, dropout_p=0.0, training=False)

    # weights shape: (B, T, T)
    weights_matrix = weights[0]

    for i in range(T):
        for j in range(T):
            if j > i:
                assert weights_matrix[i, j].item() == 0.0, (
                    f"Leakage detected! Token at position {i} attended to future token at {j} "
                    f"with weight {weights_matrix[i, j].item()}"
                )
            else:
                # Allowed positions must receive non-zero weight
                assert weights_matrix[i, j].item() > 0.0


def test_causal_invariance_to_future_token_perturbation():
    """
    Verify that altering tokens at future positions (t > k) has zero impact
    on the representations and predictions produced at position k.
    """
    torch.manual_seed(42)
    vocab_size = 65
    context_length = 16
    d_model = 32
    n_heads = 2
    n_layers = 2

    model = MiniGPT(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.0,
    )
    model.eval()

    # Sequence A: [t0, t1, t2, t3, t4, t5]
    seq_a = torch.tensor([[10, 20, 30, 40, 50, 60]])

    # Sequence B: Same prefix [t0, t1, t2], but completely different future [99, 88, 77]
    seq_b = torch.tensor([[10, 20, 30, 15, 25, 35]])

    with torch.no_grad():
        logits_a, _, _ = model(seq_a)
        logits_b, _, _ = model(seq_b)

    # For positions 0, 1, 2, logits must be identical between seq_a and seq_b
    prefix_len = 3
    diff = torch.abs(logits_a[:, :prefix_len, :] - logits_b[:, :prefix_len, :]).max().item()

    assert diff < 1e-5, f"Information leaked from future tokens! Max logit difference in prefix: {diff}"


if __name__ == "__main__":
    pytest.main([__file__])
