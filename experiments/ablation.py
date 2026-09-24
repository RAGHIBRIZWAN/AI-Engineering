"""
Ablation Experiments Suite for MiniGPT.

Implements the three controlled experiments required by Handout Section 13:
1. Depth Ablation: 2 vs. 4 vs. 6 Transformer layers.
2. Heads Ablation: 2 vs. 4 vs. 8 Attention heads.
3. Positional Encoding Ablation: Sinusoidal vs. Learned positional encodings.

Saves metrics to `outputs/logs/ablation_results.json` and renders Figure 9
(`outputs/figures/figure9_ablation_results.png`).
"""

import os
import time
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.dataset import CorpusManager
from src.gpt import MiniGPT
from src.utils import set_seed, get_device


def train_ablation_model(
    corpus_mgr: CorpusManager,
    n_layers: int = 4,
    n_heads: int = 4,
    context_length: int = 128,
    pos_encoding: str = "sinusoidal",
    d_model: int = 128,
    max_iters: int = 500,
    batch_size: int = 64,
    lr: float = 5e-4,
    device: torch.device = torch.device("cpu"),
):
    """
    Train a model variation for a fixed number of steps to compare convergence.
    """
    tokenizer = corpus_mgr.tokenizer
    set_seed(42)

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_dim=4 * d_model,
        dropout=0.1,
        bias=False,
        pos_encoding=pos_encoding,
    ).to(device)

    num_params = model.get_num_params()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    start_time = time.time()
    model.train()
    for step in range(max_iters):
        x, y = corpus_mgr.get_batch("train", batch_size, context_length, device)
        _, loss, _ = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    train_time = time.time() - start_time

    # Evaluate validation loss
    model.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(30):
            xv, yv = corpus_mgr.get_batch("val", batch_size, context_length, device)
            _, vloss, _ = model(xv, yv)
            val_losses.append(vloss.item())
    final_val_loss = sum(val_losses) / len(val_losses)

    # Sample generation
    prompt = "ROMEO:\n"
    p_tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    gen_tokens = model.generate(p_tokens, max_new_tokens=60, temperature=0.8)[0].tolist()
    sample_text = tokenizer.decode(gen_tokens)

    return {
        "num_params": num_params,
        "val_loss": round(final_val_loss, 4),
        "train_time_sec": round(train_time, 2),
        "sample": sample_text.strip(),
    }


def run_ablations(data_path: str = "data/input.txt", iters_per_run: int = 500):
    device = get_device("auto")
    print(f"\n[Ablation Suite] Initializing on {device}...")
    corpus_mgr = CorpusManager(data_path)

    results = {}

    # --- Experiment 1: Depth (2 vs 4 vs 6 Layers) ---
    print("\n" + "=" * 60)
    print("Experiment 1: Depth (2 vs 4 vs 6 Transformer Layers)")
    print("=" * 60)
    depth_results = {}
    for layers in [2, 4, 6]:
        print(f"Training Depth={layers} layers...")
        res = train_ablation_model(
            corpus_mgr=corpus_mgr,
            n_layers=layers,
            n_heads=4,
            max_iters=iters_per_run,
            device=device,
        )
        depth_results[f"{layers}_layers"] = res
        print(f"  Layers: {layers} | Params: {res['num_params']:,} | Val Loss: {res['val_loss']} | Time: {res['train_time_sec']}s")
    results["depth_ablation"] = depth_results

    # --- Experiment 2: Attention Heads (2 vs 4 vs 8 Heads) ---
    print("\n" + "=" * 60)
    print("Experiment 2: Attention Heads (2 vs 4 vs 8 Heads)")
    print("=" * 60)
    heads_results = {}
    for heads in [2, 4, 8]:
        print(f"Training Heads={heads}...")
        res = train_ablation_model(
            corpus_mgr=corpus_mgr,
            n_layers=4,
            n_heads=heads,
            max_iters=iters_per_run,
            device=device,
        )
        heads_results[f"{heads}_heads"] = res
        print(f"  Heads: {heads} | Params: {res['num_params']:,} | Val Loss: {res['val_loss']} | Time: {res['train_time_sec']}s")
    results["heads_ablation"] = heads_results

    # --- Experiment 3: Positional Encoding (Sinusoidal vs Learned) ---
    print("\n" + "=" * 60)
    print("Experiment 3: Positional Encoding (Sinusoidal vs Learned)")
    print("=" * 60)
    pos_results = {}
    for p_type in ["sinusoidal", "learned"]:
        print(f"Training Positional Encoding={p_type}...")
        res = train_ablation_model(
            corpus_mgr=corpus_mgr,
            n_layers=4,
            n_heads=4,
            pos_encoding=p_type,
            max_iters=iters_per_run,
            device=device,
        )
        pos_results[p_type] = res
        print(f"  Type: {p_type} | Params: {res['num_params']:,} | Val Loss: {res['val_loss']} | Time: {res['train_time_sec']}s")
    results["position_ablation"] = pos_results

    # Save to JSON
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    with open("outputs/logs/ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n[Ablations Complete] Saved metrics to outputs/logs/ablation_results.json")

    # Render Figure 9: Ablation Results Plots
    render_figure9_plots(results)

    return results


def render_figure9_plots(results: dict):
    """Render comprehensive comparison bar charts for Figure 9."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Depth plot
    depth_keys = list(results["depth_ablation"].keys())
    depth_labels = [k.replace("_", " ").title() for k in depth_keys]
    depth_losses = [results["depth_ablation"][k]["val_loss"] for k in depth_keys]
    bars1 = axes[0].bar(depth_labels, depth_losses, color=["#4299e1", "#3182ce", "#2b6cb0"])
    axes[0].set_title("Experiment 1: Depth (Layers)\nvs Validation Loss", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Validation Loss", fontsize=10)
    axes[0].set_ylim(min(depth_losses) - 0.2, max(depth_losses) + 0.2)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.3f}", ha="center", fontsize=10, fontweight="bold")

    # 2. Heads plot
    head_keys = list(results["heads_ablation"].keys())
    head_labels = [k.replace("_", " ").title() for k in head_keys]
    head_losses = [results["heads_ablation"][k]["val_loss"] for k in head_keys]
    bars2 = axes[1].bar(head_labels, head_losses, color=["#48bb78", "#38a169", "#2f855a"])
    axes[1].set_title("Experiment 2: Attention Heads\nvs Validation Loss", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Validation Loss", fontsize=10)
    axes[1].set_ylim(min(head_losses) - 0.2, max(head_losses) + 0.2)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.3f}", ha="center", fontsize=10, fontweight="bold")

    # 3. Position plot
    pos_keys = list(results["position_ablation"].keys())
    pos_labels = [k.capitalize() for k in pos_keys]
    pos_losses = [results["position_ablation"][k]["val_loss"] for k in pos_keys]
    bars3 = axes[2].bar(pos_labels, pos_losses, color=["#ed8936", "#dd6b20"])
    axes[2].set_title("Experiment 3: Positional Encoding\nvs Validation Loss", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Validation Loss", fontsize=10)
    axes[2].set_ylim(min(pos_losses) - 0.2, max(pos_losses) + 0.2)
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("Figure 9: MiniGPT Controlled Ablation Studies", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path = "outputs/figures/figure9_ablation_results.png"
    plt.savefig(out_path, dpi=300)
    plt.savefig("outputs/figures/ablation_results.png", dpi=300)
    plt.close()
    print(f"[Generated] {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run controlled ablation experiments")
    parser.add_argument("--iters", type=int, default=200, help="Training iterations per ablation variant")
    args = parser.parse_args()
    run_ablations(iters_per_run=args.iters)
