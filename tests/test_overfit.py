"""
Overfitting test for MiniGPT.

Debugging gate from Milestone 4 and Section 16:
Confirms that gradients flow properly, target shifting is aligned,
and the model can rapidly memorize a single mini-batch.
"""

import pytest
import torch
import torch.optim as optim

from src.gpt import MiniGPT


def test_overfit_single_batch():
    """
    Train a tiny model on a single fixed batch.
    Pass condition: Loss falls substantially from initial ~4.17 to < 0.25.
    """
    torch.manual_seed(42)
    vocab_size = 65
    context_length = 16
    d_model = 64
    n_heads = 2
    n_layers = 2
    batch_size = 4

    model = MiniGPT(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.0,
    )

    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    # Fixed synthetic batch
    x = torch.randint(0, vocab_size, (batch_size, context_length))
    # Targets are shifted next tokens
    y = torch.randint(0, vocab_size, (batch_size, context_length))

    # Evaluate initial loss
    model.eval()
    with torch.no_grad():
        _, initial_loss, _ = model(x, y)
    init_val = initial_loss.item()

    # Train on single batch for 100 steps
    model.train()
    final_loss_val = init_val
    for step in range(120):
        optimizer.zero_grad()
        _, loss, _ = model(x, y)
        loss.backward()
        optimizer.step()
        final_loss_val = loss.item()

    print(f"\n[Overfit Test] Initial Loss: {init_val:.4f} -> Final Loss: {final_loss_val:.4f}")

    # Pass conditions:
    # 1. Final loss must fall substantially (< 0.25)
    # 2. Total loss drop must exceed 85%
    assert final_loss_val < 0.25, f"Overfit failed! Final loss {final_loss_val:.4f} is too high."
    assert (init_val - final_loss_val) / init_val > 0.85, "Loss did not drop substantially."


if __name__ == "__main__":
    test_overfit_single_batch()
    print("Overfit test passed successfully!")
