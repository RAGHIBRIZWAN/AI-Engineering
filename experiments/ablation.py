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

    import psutil
    process = psutil.Process()
    mem_before = process.memory_info().rss
    peak_mem_bytes = mem_before

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
        if (step + 1) % 25 == 0 or step == max_iters - 1:
            curr_mem = process.memory_info().rss
            if curr_mem > peak_mem_bytes:
                peak_mem_bytes = curr_mem

    train_time = time.time() - start_time

    if device.type == "cuda" and torch.cuda.is_available():
        peak_memory_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
    else:
        peak_memory_mb = round(peak_mem_bytes / (1024 * 1024), 2)

    total_tokens = max_iters * batch_size * context_length
    tokens_per_sec = round(total_tokens / max(train_time, 1e-4), 1)

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
        "peak_memory_mb": peak_memory_mb,
        "tokens_per_sec": tokens_per_sec,
        "sample": sample_text.strip(),
    }


def run_ablations(data_path: str = "data/input.txt", iters_per_run: int = 500, only_context: bool = False, render_plots: bool = False):
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

    # --- Experiment 4: Context Length (64 vs 128 vs 256) ---
    print("\n" + "=" * 60)
    print("Experiment 4: Context Length (64 vs 128 vs 256 Tokens)")
    print("=" * 60)
    context_results = {}
    for ctx_len in [64, 128, 256]:
        print(f"Training Context Length={ctx_len}...")
        res = train_ablation_model(
            corpus_mgr=corpus_mgr,
            n_layers=4,
            n_heads=4,
            context_length=ctx_len,
            pos_encoding="sinusoidal",
            max_iters=iters_per_run,
            device=device,
        )
        context_results[f"{ctx_len}_context"] = res
        print(
            f"  Context: {ctx_len} | Params: {res['num_params']:,} | "
            f"Val Loss: {res['val_loss']} | Mem: {res['peak_memory_mb']}MB | "
            f"Throughput: {res['tokens_per_sec']} tok/s | Time: {res['train_time_sec']}s"
        )
    results["context_length_ablation"] = context_results

    # Save to JSON
    os.makedirs("outputs/logs", exist_ok=True)
    with open("outputs/logs/ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n[Ablations Complete] Saved metrics to outputs/logs/ablation_results.json")

    # Render Figure 9: Ablation Results Plots only if requested (preserve existing figure by default)
    if render_plots:
        render_figure9_plots(results)

    return results


def run_context_length_only(data_path: str = "data/input.txt", iters_per_run: int = 200):
    """Run only the Context Length ablation and append to existing outputs/logs/ablation_results.json."""
    device = get_device("auto")
    print(f"\n[Context Length Ablation] Initializing on {device}...", flush=True)
    corpus_mgr = CorpusManager(data_path)

    log_path = "outputs/logs/ablation_results.json"
    results = {}
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            results = json.load(f)

    print("\n" + "=" * 60, flush=True)
    print("Experiment 4: Context Length (64 vs 128 vs 256 Tokens)", flush=True)
    print("=" * 60, flush=True)
    context_results = results.get("context_length_ablation", {})
    for ctx_len in [64, 128, 256]:
        print(f"Training Context Length={ctx_len}...", flush=True)
        res = train_ablation_model(
            corpus_mgr=corpus_mgr,
            n_layers=4,
            n_heads=4,
            context_length=ctx_len,
            pos_encoding="sinusoidal",
            max_iters=iters_per_run,
            device=device,
        )
        context_results[f"{ctx_len}_context"] = res
        print(
            f"  Context: {ctx_len} | Params: {res['num_params']:,} | "
            f"Val Loss: {res['val_loss']} | Mem: {res['peak_memory_mb']}MB | "
            f"Throughput: {res['tokens_per_sec']} tok/s | Time: {res['train_time_sec']}s",
            flush=True,
        )
        # Incremental save
        results["context_length_ablation"] = context_results
        os.makedirs("outputs/logs", exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(f"\n[Context Length Ablation Complete] Saved metrics to {log_path}", flush=True)
    return results


def render_figure9_plots(results: dict):
    """Render comprehensive comparison bar charts for Figure 9 across all 4 ablations."""
    has_ctx = "context_length_ablation" in results
    n_cols = 4 if has_ctx else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(19, 4.8) if has_ctx else (15, 4.5))

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

    # 4. Context Length plot
    if has_ctx:
        ctx_keys = list(results["context_length_ablation"].keys())
        ctx_labels = [k.replace("_", " ").title() for k in ctx_keys]
        ctx_losses = [results["context_length_ablation"][k]["val_loss"] for k in ctx_keys]
        bars4 = axes[3].bar(ctx_labels, ctx_losses, color=["#9f7aea", "#805ad5", "#6b46c1"])
        axes[3].set_title("Experiment 4: Context Length\nvs Validation Loss", fontsize=11, fontweight="bold")
        axes[3].set_ylabel("Validation Loss", fontsize=10)
        axes[3].set_ylim(min(ctx_losses) - 0.2, max(ctx_losses) + 0.2)
        for bar in bars4:
            yval = bar.get_height()
            axes[3].text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.3f}", ha="center", fontsize=10, fontweight="bold")

    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Figure 9: MiniGPT Controlled Ablation Studies", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs("outputs/figures", exist_ok=True)
    out_path = "outputs/figures/figure9_ablation_results.png"
    plt.savefig(out_path, dpi=300)
    plt.savefig("outputs/figures/ablation_results.png", dpi=300)
    plt.close()
    print(f"[Generated] {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run controlled ablation experiments")
    parser.add_argument("--iters", type=int, default=200, help="Training iterations per ablation variant")
    parser.add_argument("--only-context", action="store_true", help="Run only the Context Length ablation")
    parser.add_argument("--render-plots", action="store_true", help="Render Figure 9 plots")
    parser.add_argument("--plots-only", action="store_true", help="Render plots from existing ablation_results.json without retraining")
    args = parser.parse_args()

    if args.plots_only:
        log_path = "outputs/logs/ablation_results.json"
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                saved_results = json.load(f)
            render_figure9_plots(saved_results)
        else:
            print(f"[Error] {log_path} not found.")
    elif args.only_context:
        res = run_context_length_only(iters_per_run=args.iters)
        if args.render_plots:
            render_figure9_plots(res)
    else:
        run_ablations(iters_per_run=args.iters, render_plots=args.render_plots)
