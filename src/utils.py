"""
Utility functions for MiniGPT.

Contains:
- Configuration management (YAML).
- Seed setting for complete reproducibility.
- Parameter count and architecture summary reporting.
- Checkpoint saving and loading.
- Device selection.
- Worked numeric hand-calculation of self-attention (Handout Section 6).
"""

import os
import random
import yaml
import torch
import numpy as np
from typing import Dict, Any, Tuple


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration dictionary to a YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def set_seed(seed: int = 42):
    """Set random seed across all libraries to ensure full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_pref: str = "auto") -> torch.device:
    """Return torch.device object based on preference and availability."""
    if device_pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_pref)


def get_parameter_count(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.

    Returns:
        (total_parameters, trainable_parameters)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model: torch.nn.Module, input_size: Tuple[int, int] = (1, 128)):
    """
    Print an organized summary of model layers, tensor shapes, and parameter counts.
    """
    total_params, trainable_params = get_parameter_count(model)
    print("=" * 70)
    print(f"{'MiniGPT Model Architecture Summary':^70}")
    print("=" * 70)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("-" * 70)
    for name, module in model.named_children():
        mod_params = sum(p.numel() for p in module.parameters())
        print(f"  [{name:<18}] {module.__class__.__name__:<25} ({mod_params:,} params)")
    print("=" * 70)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    train_loss: float,
    val_loss: float,
    config: Dict[str, Any],
    save_path: str,
):
    """Save model weights and training state to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "config": config,
    }
    torch.save(checkpoint, save_path)
    print(f"[Checkpoint] Successfully saved to: {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Load model weights and training state from disk."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"[Checkpoint] Successfully loaded from: {checkpoint_path} (step {checkpoint.get('step', '?')})")
    return checkpoint


def compute_hand_calculation_example():
    """
    Executes and formats the exact 3-token self-attention hand calculation
    required by Handout Section 6.
    
    Q, K, V are explicitly specified for a sequence of 3 tokens with d_k = 2, d_v = 2.
    Computes:
      1. Explicit Q, K, V matrices
      2. Q @ K^T
      3. Scale by 1/sqrt(d_k)
      4. Softmax (unmasked)
      5. Weights @ V (unmasked)
      6. Causal mask application & masked softmax
      7. Weights @ V (masked)
    """
    import math

    # Explicit 3-token sequence with d_k = 2, d_v = 2
    # Token 1: "The", Token 2: "cat", Token 3: "sat"
    Q = np.array([
        [1.0, 0.0],  # Token 1 query
        [0.0, 2.0],  # Token 2 query
        [1.0, 1.0],  # Token 3 query
    ], dtype=float)

    K = np.array([
        [1.0, 1.0],  # Token 1 key
        [0.0, 1.0],  # Token 2 key
        [2.0, 0.0],  # Token 3 key
    ], dtype=float)

    V = np.array([
        [1.0, 2.0],  # Token 1 value
        [3.0, 0.0],  # Token 2 value
        [0.0, 4.0],  # Token 3 value
    ], dtype=float)

    d_k = 2
    sqrt_dk = math.sqrt(d_k)

    # Step 2: Q @ K^T
    raw_scores = Q @ K.T

    # Step 3: Divide by sqrt(d_k)
    scaled_scores = raw_scores / sqrt_dk

    # Step 4: Row-wise Softmax (Unmasked)
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    unmasked_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Step 5: Multiply unmasked weights by V
    unmasked_output = unmasked_weights @ V

    # Step 6: Causal Mask (Upper-triangular blocked)
    # Mask is lower-triangular 1s:
    # [[1, 0, 0],
    #  [1, 1, 0],
    #  [1, 1, 1]]
    causal_mask = np.tril(np.ones((3, 3)))
    masked_scores = np.where(causal_mask == 1, scaled_scores, -1e9)

    # Masked row-wise Softmax
    exp_masked = np.exp(masked_scores - np.max(masked_scores, axis=-1, keepdims=True))
    masked_weights = exp_masked / np.sum(exp_masked, axis=-1, keepdims=True)

    # Step 7: Multiply masked weights by V
    masked_output = masked_weights @ V

    return {
        "Q": Q,
        "K": K,
        "V": V,
        "d_k": d_k,
        "sqrt_dk": sqrt_dk,
        "raw_scores": raw_scores,
        "scaled_scores": scaled_scores,
        "unmasked_weights": unmasked_weights,
        "unmasked_output": unmasked_output,
        "causal_mask": causal_mask,
        "masked_scores": masked_scores,
        "masked_weights": masked_weights,
        "masked_output": masked_output,
    }
