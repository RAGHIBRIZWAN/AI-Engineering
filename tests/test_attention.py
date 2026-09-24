"""
Unit tests for Scaled Dot-Product Attention and Multi-Head Attention.
"""

import math
import pytest
import torch
import numpy as np

from src.attention import scaled_dot_product_attention, MultiHeadAttention
from src.utils import compute_hand_calculation_example


def test_hand_calculation_match():
    """
    Verify that PyTorch's scaled_dot_product_attention produces the exact same
    numerical output as the Section 6 hand calculation.
    """
    hand_res = compute_hand_calculation_example()

    Q_torch = torch.tensor(hand_res["Q"], dtype=torch.float32).unsqueeze(0) # (1, 3, 2)
    K_torch = torch.tensor(hand_res["K"], dtype=torch.float32).unsqueeze(0) # (1, 3, 2)
    V_torch = torch.tensor(hand_res["V"], dtype=torch.float32).unsqueeze(0) # (1, 3, 2)
    mask_torch = torch.tensor(hand_res["causal_mask"], dtype=torch.float32).unsqueeze(0) # (1, 3, 3)

    out, weights = scaled_dot_product_attention(
        q=Q_torch,
        k=K_torch,
        v=V_torch,
        mask=mask_torch,
        dropout_p=0.0,
        training=False,
    )

    out_np = out.squeeze(0).numpy()
    weights_np = weights.squeeze(0).numpy()

    # Assert attention weights match hand calculation within numerical tolerance
    np.testing.assert_allclose(weights_np, hand_res["masked_weights"], rtol=1e-5, atol=1e-6)
    # Assert weighted output matches hand calculation
    np.testing.assert_allclose(out_np, hand_res["masked_output"], rtol=1e-5, atol=1e-6)


def test_scaled_dot_product_attention_shapes():
    """
    Verify output shapes and probability normalization of scaled dot-product attention.
    """
    B, T, d_k, d_v = 2, 8, 32, 32
    q = torch.randn(B, T, d_k)
    k = torch.randn(B, T, d_k)
    v = torch.randn(B, T, d_v)

    out, weights = scaled_dot_product_attention(q, k, v, dropout_p=0.0, training=False)

    assert out.shape == (B, T, d_v), f"Expected {(B, T, d_v)}, got {out.shape}"
    assert weights.shape == (B, T, T), f"Expected {(B, T, T)}, got {weights.shape}"

    # Verify softmax property: weights along key dimension sum to 1.0
    weight_sums = weights.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)


def test_multi_head_attention_dimensions():
    """
    Verify MultiHeadAttention produces correct input-to-output dimensions.
    """
    B, T, d_model, n_heads = 4, 16, 64, 4
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0, causal=True)
    mha.eval()

    x = torch.randn(B, T, d_model)
    out, weights = mha(x, return_attention=True)

    # Output dimension must match input (B, T, d_model)
    assert out.shape == (B, T, d_model)
    # Attention weights shape: (B, n_heads, T, T)
    assert weights.shape == (B, n_heads, T, T)


if __name__ == "__main__":
    pytest.main([__file__])
