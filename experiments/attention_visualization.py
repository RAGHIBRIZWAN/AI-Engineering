"""
Attention and Representation Visualization Suite.

Generates key figures required by Handout Section 12:
- Figure 1: Hand-calculated attention example flow.
- Figure 2: Causal mask matrix vs BERT Bidirectional mask matrix heatmap.
- Figure 3: Multi-head attention tensor shape trace visual summary.
- Figure 4: Positional encoding heatmap and frequency curves.
- Figure 7: Multi-head attention heatmaps extracted from a trained MiniGPT model.
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.utils import compute_hand_calculation_example, load_config, get_device, load_checkpoint
from src.positional_encoding import SinusoidalPositionalEncoding
from src.tokenizer import CharacterTokenizer
from src.gpt import MiniGPT


def plot_figure1_hand_calculation():
    """Generate Figure 1: Step-by-step numeric hand calculation visualization."""
    res = compute_hand_calculation_example()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    tokens = ["Token 1", "Token 2", "Token 3"]

    # 1. Scaled Scores: (Q @ K^T) / sqrt(d_k)
    im0 = axes[0].imshow(res["scaled_scores"], cmap="Blues")
    axes[0].set_title(r"1. Scaled Scores $\frac{Q K^T}{\sqrt{d_k}}$", fontsize=12, fontweight="bold")
    axes[0].set_xticks(range(3))
    axes[0].set_yticks(range(3))
    axes[0].set_xticklabels(tokens)
    axes[0].set_yticklabels(tokens)
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, f"{res['scaled_scores'][i, j]:.2f}", ha="center", va="center", color="black", fontsize=11)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Causal Masked Softmax Weights
    im1 = axes[1].imshow(res["masked_weights"], cmap="YlGnBu", vmin=0, vmax=1)
    axes[1].set_title(r"2. Masked Weights $\mathrm{softmax}(\cdot)$", fontsize=12, fontweight="bold")
    axes[1].set_xticks(range(3))
    axes[1].set_yticks(range(3))
    axes[1].set_xticklabels(tokens)
    axes[1].set_yticklabels(tokens)
    for i in range(3):
        for j in range(3):
            val = res["masked_weights"][i, j]
            color = "white" if val > 0.5 else "black"
            axes[1].text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Final Attention Output (Weights @ V)
    dims = [r"$v_1$", r"$v_2$"]
    im2 = axes[2].imshow(res["masked_output"], cmap="Greens")
    axes[2].set_title(r"3. Final Output $\mathrm{Weights} \times V$", fontsize=12, fontweight="bold")
    axes[2].set_xticks(range(2))
    axes[2].set_yticks(range(3))
    axes[2].set_xticklabels(dims)
    axes[2].set_yticklabels(tokens)
    for i in range(3):
        for j in range(2):
            axes[2].text(j, i, f"{res['masked_output'][i, j]:.2f}", ha="center", va="center", color="black", fontsize=11, fontweight="bold")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle("Figure 1: Numerical Hand Calculation of Causal Self-Attention ($d_k=2, d_v=2$)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = "outputs/figures/figure1_hand_calculation.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Generated] {out_path}")


def plot_figure2_causal_vs_bert_mask():
    """Generate Figure 2: Causal Mask (GPT) vs Bidirectional Mask (BERT)."""
    seq_len = 8
    # GPT Causal Mask: Lower triangular 1s, upper triangular 0s
    gpt_mask = np.tril(np.ones((seq_len, seq_len)))

    # BERT Bidirectional Mask: Full visibility (all 1s) with simulated random masked token
    bert_mask = np.ones((seq_len, seq_len))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Causal Mask Heatmap
    im0 = axes[0].imshow(gpt_mask, cmap="Blues", interpolation="nearest")
    axes[0].set_title("GPT Autoregressive Causal Mask\n(Strict Lower-Triangular: Future Blocked)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Key Position (Attended)", fontsize=11)
    axes[0].set_ylabel("Query Position (Current)", fontsize=11)
    for i in range(seq_len):
        for j in range(seq_len):
            txt = "1" if gpt_mask[i, j] == 1 else "0"
            color = "white" if gpt_mask[i, j] == 1 else "gray"
            axes[0].text(j, i, txt, ha="center", va="center", color=color, fontweight="bold")

    # BERT Bidirectional Heatmap
    im1 = axes[1].imshow(bert_mask, cmap="Purples", interpolation="nearest")
    axes[1].set_title("BERT Bidirectional Mask\n(Full Context Visibility: Past & Future)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Key Position (Attended)", fontsize=11)
    axes[1].set_ylabel("Query Position (Current)", fontsize=11)
    for i in range(seq_len):
        for j in range(seq_len):
            axes[1].text(j, i, "1", ha="center", va="center", color="white", fontweight="bold")

    plt.suptitle("Figure 2: Attention Masking Comparison: GPT Causal LM vs. BERT Bidirectional MLM", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = "outputs/figures/figure2_causal_mask.png"
    plt.savefig(out_path, dpi=300)
    plt.savefig("outputs/figures/causal_mask.png", dpi=300)
    plt.close()
    print(f"[Generated] {out_path}")


def plot_figure3_tensor_shapes():
    """Generate Figure 3: Visual Tensor Shape Trace through Multi-Head Attention."""
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")

    stages = [
        ("Input Tensor (x)", "(B, T, d_model)", "(64, 128, 128)"),
        ("Linear Projections (W_q, W_k, W_v)", "(B, T, d_model) each", "(64, 128, 128) each"),
        ("Reshape to Split Heads", "(B, T, n_heads, d_k)", "(64, 128, 4, 32)"),
        ("Transpose for Scaled Dot-Product", "(B, n_heads, T, d_k)", "(64, 4, 128, 32)"),
        ("Attention Scores (Q @ K^T / sqrt(d_k))", "(B, n_heads, T, T)", "(64, 4, 128, 128)"),
        ("Causal Masking & Softmax", "(B, n_heads, T, T)", "(64, 4, 128, 128)"),
        ("Weighted Value Sum (Weights @ V)", "(B, n_heads, T, d_k)", "(64, 4, 128, 32)"),
        ("Transpose back & Contiguous View", "(B, T, n_heads, d_k)", "(64, 128, 4, 32)"),
        ("Merge/Concatenate Heads", "(B, T, d_model)", "(64, 128, 128)"),
        ("Final Output Linear Projection (W_o)", "(B, T, d_model)", "(64, 128, 128)"),
    ]

    y_pos = np.linspace(0.9, 0.05, len(stages))
    ax.text(0.05, 0.98, "Transformation Stage", fontsize=12, fontweight="bold")
    ax.text(0.50, 0.98, "Symbolic Shape", fontsize=12, fontweight="bold")
    ax.text(0.78, 0.98, "Concrete Shape Example", fontsize=12, fontweight="bold")
    ax.plot([0.02, 0.98], [0.95, 0.95], color="black", lw=1.5)

    for idx, (stage, sym, conc) in enumerate(stages):
        y = y_pos[idx]
        bg_col = "#f0f4f8" if idx % 2 == 0 else "#ffffff"
        ax.axhspan(y - 0.035, y + 0.035, xmin=0.02, xmax=0.98, color=bg_col, zorder=0)
        ax.text(0.05, y, stage, fontsize=10.5, va="center", color="#1a202c", zorder=1)
        ax.text(0.50, y, sym, fontsize=10.5, fontfamily="monospace", va="center", color="#2b6cb0", zorder=1)
        ax.text(0.78, y, conc, fontsize=10.5, fontfamily="monospace", va="center", color="#2f855a", zorder=1)

    plt.title("Figure 3: Multi-Head Attention Tensor Shape Trace (Split, Attention, Merge)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    out_path = "outputs/figures/figure3_multihead_shapes.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Generated] {out_path}")


def plot_figure4_positional_encoding():
    """Generate Figure 4: Sinusoidal Positional Encoding Heatmap and Waveforms."""
    d_model = 128
    max_len = 128
    pe_layer = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
    pe_matrix = pe_layer.pe.squeeze(0).cpu().numpy() # (max_len, d_model)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Heatmap of Positional Encoding Matrix
    im = axes[0].imshow(pe_matrix, aspect="auto", cmap="viridis")
    axes[0].set_title("Positional Encoding Matrix PE(pos, 2i)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Embedding Dimension index (i)", fontsize=11)
    axes[0].set_ylabel("Token Sequence Position (pos)", fontsize=11)
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # Line plots of Sinusoidal Waves at different dimensions
    positions = np.arange(max_len)
    axes[1].plot(positions, pe_matrix[:, 0], label="dim = 0 (highest frequency)", color="#e53e3e", lw=2)
    axes[1].plot(positions, pe_matrix[:, 8], label="dim = 8", color="#dd6b20", lw=2)
    axes[1].plot(positions, pe_matrix[:, 32], label="dim = 32", color="#319795", lw=2)
    axes[1].plot(positions, pe_matrix[:, 64], label="dim = 64 (low frequency)", color="#3182ce", lw=2)
    axes[1].set_title("Sinusoidal Frequencies across Token Positions", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Position (pos)", fontsize=11)
    axes[1].set_ylabel("PE Value", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", fontsize=9.5)

    plt.suptitle("Figure 4: Sinusoidal Positional Encoding Visualization (Vaswani et al., 2017)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = "outputs/figures/figure4_positional_encoding.png"
    plt.savefig(out_path, dpi=300)
    plt.savefig("outputs/figures/positional_encoding.png", dpi=300)
    plt.close()
    print(f"[Generated] {out_path}")


def plot_figure7_trained_attention_map(checkpoint_path: str = "checkpoints/best_model.pt"):
    """Generate Figure 7: Multi-Head Attention Maps from a trained MiniGPT model."""
    device = get_device("auto")
    config = load_config("configs/tiny_gpt.yaml")
    model_cfg = config["model"]

    # Load tokenizer
    tok_path = "checkpoints/tokenizer.json"
    if os.path.exists(tok_path):
        tokenizer = CharacterTokenizer.load(tok_path)
    else:
        with open("data/input.txt", "r", encoding="utf-8") as f:
            tokenizer = CharacterTokenizer.from_text(f.read())

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=model_cfg["context_length"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        ffn_dim=model_cfg.get("ffn_dim", 4 * model_cfg["d_model"]),
        dropout=0.0,
        bias=model_cfg.get("bias", False),
        pos_encoding=model_cfg.get("pos_encoding", "sinusoidal"),
    ).to(device)

    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model, device=device)
    else:
        print(f"[Warning] Trained checkpoint not found at {checkpoint_path}. Using initial weights for Figure 7.")

    model.eval()

    sample_sentence = "ROMEO:\nWhat light"
    tokens = tokenizer.encode(sample_sentence)
    token_chars = [repr(ch)[1:-1] for ch in sample_sentence] # readable char representation
    seq_len = len(tokens)

    x = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        _, _, all_attns = model(x, return_attentions=True)

    # Pick the last Transformer layer's attention map: shape (1, n_heads, seq_len, seq_len)
    last_layer_attn = all_attns[-1][0].cpu().numpy()
    n_heads = last_layer_attn.shape[0]

    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]

    for h in range(n_heads):
        im = axes[h].imshow(last_layer_attn[h], cmap="magma", vmin=0, vmax=1)
        axes[h].set_title(f"Head {h + 1}", fontsize=11, fontweight="bold")
        axes[h].set_xticks(range(seq_len))
        axes[h].set_yticks(range(seq_len))
        axes[h].set_xticklabels(token_chars, fontsize=9)
        axes[h].set_yticklabels(token_chars, fontsize=9)
        axes[h].set_xlabel("Key Token", fontsize=10)
        if h == 0:
            axes[h].set_ylabel("Query Token", fontsize=10)

    fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.06, pad=0.25, label="Attention Weight")
    plt.suptitle("Figure 7: Multi-Head Attention Maps Extracted from Trained MiniGPT (Layer 4)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = "outputs/figures/figure7_attention_map.png"
    plt.savefig(out_path, dpi=300)
    plt.savefig("outputs/figures/attention_map.png", dpi=300)
    plt.close()
    print(f"[Generated] {out_path}")


if __name__ == "__main__":
    os.makedirs("outputs/figures", exist_ok=True)
    print("Generating Figure 1 (Hand calculation)...")
    plot_figure1_hand_calculation()
    print("Generating Figure 2 (Causal vs BERT mask)...")
    plot_figure2_causal_vs_bert_mask()
    print("Generating Figure 3 (Tensor shape trace)...")
    plot_figure3_tensor_shapes()
    print("Generating Figure 4 (Positional encoding)...")
    plot_figure4_positional_encoding()
    print("Generating Figure 7 (Trained attention map)...")
    plot_figure7_trained_attention_map()
    print("All attention visualizations generated successfully!")
