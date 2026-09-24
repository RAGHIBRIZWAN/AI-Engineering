"""
Training script for MiniGPT.

Handles:
1. Loading configuration and dataset (Tiny Shakespeare).
2. Fitting and saving the deterministic CharacterTokenizer.
3. Instantiating the MiniGPT causal language model.
4. Setting up the AdamW optimizer and cosine learning rate scheduler with warmup.
5. Executing the training loop with periodic validation loss estimation.
6. Periodic text generation sampling during training.
7. Checkpoint saving (best and final models).
8. Logging training metrics and rendering loss curve figures.
"""

import os
import time
import math
import json
import argparse
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.tokenizer import CharacterTokenizer
from src.dataset import CorpusManager
from src.gpt import MiniGPT
from src.utils import load_config, set_seed, get_device, print_model_summary, save_checkpoint


def get_lr(step: int, warmup_iters: int, lr_decay_iters: int, max_lr: float, min_lr: float) -> float:
    """
    Compute learning rate using linear warmup followed by cosine decay.
    """
    # 1) Linear warmup for warmup_iters steps
    if step < warmup_iters:
        return max_lr * (step + 1) / (warmup_iters + 1)
    # 2) If step > lr_decay_iters, return min learning rate
    if step > lr_decay_iters:
        return min_lr
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    corpus_mgr: CorpusManager,
    eval_iters: int,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Estimate average loss over several batches for both train and validation sets.
    """
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = corpus_mgr.get_batch(split, batch_size, context_length, device)
            _, loss, _ = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(config_path: str = "configs/tiny_gpt.yaml", cli_overrides: Dict[str, Any] = None):
    """
    Main training function.
    """
    config = load_config(config_path)

    # Apply CLI overrides if provided
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                if k in config["training"]:
                    config["training"][k] = v
                elif k in config["model"]:
                    config["model"][k] = v

    # Extract configs
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["data"]
    gen_cfg = config.get("generation", {})

    # Set random seed
    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    # Device configuration
    device = get_device(train_cfg.get("device", "auto"))
    print(f"\n[Environment] Using Device: {device} | Random Seed: {seed}")

    # Ensure output directories exist
    os.makedirs(train_cfg.get("checkpoint_dir", "checkpoints"), exist_ok=True)
    os.makedirs(train_cfg.get("log_dir", "outputs/logs"), exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/samples", exist_ok=True)

    # 1. Load data and setup tokenizer
    data_path = data_cfg.get("data_path", "data/input.txt")
    print(f"[Data] Loading dataset from: {data_path}")
    corpus_mgr = CorpusManager(filepath=data_path, train_split=data_cfg.get("train_split", 0.9))
    tokenizer = corpus_mgr.tokenizer

    # Save tokenizer for reproducible inference and deployment
    tok_save_path = os.path.join(train_cfg.get("checkpoint_dir", "checkpoints"), "tokenizer.json")
    tokenizer.save(tok_save_path)
    print(f"[Tokenizer] Saved vocabulary ({tokenizer.vocab_size} tokens) to {tok_save_path}")

    # Set vocab_size in model configuration
    model_cfg["vocab_size"] = tokenizer.vocab_size

    # 2. Instantiate Model
    print("\n[Model] Initializing MiniGPT...")
    model = MiniGPT(
        vocab_size=model_cfg["vocab_size"],
        context_length=model_cfg["context_length"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        ffn_dim=model_cfg.get("ffn_dim", 4 * model_cfg["d_model"]),
        dropout=model_cfg.get("dropout", 0.1),
        bias=model_cfg.get("bias", False),
        pos_encoding=model_cfg.get("pos_encoding", "sinusoidal"),
        tie_weights=model_cfg.get("tie_weights", False),
    ).to(device)

    print_model_summary(model)

    # 3. Setup Optimizer
    # Separate parameters that require weight decay (weights) from those that do not (biases and LayerNorms)
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": train_cfg.get("weight_decay", 0.01)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=train_cfg.get("learning_rate", 5e-4),
        betas=(train_cfg.get("beta1", 0.9), train_cfg.get("beta2", 0.99)),
    )

    # Training parameters
    max_iters = train_cfg.get("max_iters", 2000)
    batch_size = train_cfg.get("batch_size", 64)
    context_length = model_cfg["context_length"]
    eval_interval = train_cfg.get("eval_interval", 200)
    eval_iters = train_cfg.get("eval_iters", 50)
    sample_interval = train_cfg.get("sample_interval", 400)
    warmup_iters = train_cfg.get("warmup_iters", 100)
    lr_decay_iters = train_cfg.get("lr_decay_iters", max_iters)
    max_lr = train_cfg.get("learning_rate", 5e-4)
    min_lr = train_cfg.get("min_lr", 5e-5)
    grad_clip = train_cfg.get("grad_clip", 1.0)

    # Tracking metrics
    history: Dict[str, List[Any]] = {
        "step": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
        "elapsed_time_sec": [],
    }

    best_val_loss = float("inf")
    start_time = time.time()

    print(f"\n[Training] Starting run: {max_iters} iterations | Batch size: {batch_size} | Context: {context_length}")
    print("-" * 75)

    model.train()
    for iter_num in range(max_iters + 1):
        # Update learning rate according to schedule
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Periodic evaluation on train and validation splits
        if iter_num % eval_interval == 0 or iter_num == max_iters:
            losses = estimate_loss(model, corpus_mgr, eval_iters, batch_size, context_length, device)
            elapsed = time.time() - start_time

            history["step"].append(iter_num)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses["val"])
            history["learning_rate"].append(lr)
            history["elapsed_time_sec"].append(elapsed)

            print(
                f"Step {iter_num:>5}/{max_iters} | "
                f"Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f} | "
                f"LR: {lr:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Save checkpoint if best validation loss is achieved
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                best_path = os.path.join(train_cfg.get("checkpoint_dir", "checkpoints"), "best_model.pt")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=0,
                    step=iter_num,
                    train_loss=losses["train"],
                    val_loss=losses["val"],
                    config=config,
                    save_path=best_path,
                )

        # Periodic sample generation during training
        if iter_num > 0 and (iter_num % sample_interval == 0 or iter_num == max_iters):
            sample_prompt = "KING LEAR:\n"
            prompt_tokens = tokenizer.encode(sample_prompt)
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
            gen_tokens = model.generate(
                prompt_tensor,
                max_new_tokens=100,
                temperature=gen_cfg.get("temperature", 0.8),
                top_k=gen_cfg.get("top_k", 40),
            )[0].tolist()
            generated_text = tokenizer.decode(gen_tokens)
            print(f"\n--- [Sample Generation @ Step {iter_num}] ---")
            print(generated_text.strip())
            print("-" * 50 + "\n")

            # Save sample to outputs/samples
            sample_file = f"outputs/samples/sample_step_{iter_num}.txt"
            with open(sample_file, "w", encoding="utf-8") as f:
                f.write(f"Sample at Step {iter_num}\n")
                f.write(generated_text)

        # Stop after evaluation at max_iters
        if iter_num == max_iters:
            break

        # Standard forward & backward step
        x_batch, y_batch = corpus_mgr.get_batch("train", batch_size, context_length, device)
        _, loss, _ = model(x_batch, y_batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping prevents gradient explosion in deep Transformers
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

    total_time = time.time() - start_time
    print("=" * 75)
    print(f"[Training Complete] Total Wall-Clock Time: {total_time:.2f} seconds ({total_time / 60:.2f} mins)")
    print(f"Final Best Validation Loss: {best_val_loss:.4f}")

    # Save final model checkpoint
    final_path = os.path.join(train_cfg.get("checkpoint_dir", "checkpoints"), "final_model.pt")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=0,
        step=max_iters,
        train_loss=history["train_loss"][-1],
        val_loss=history["val_loss"][-1],
        config=config,
        save_path=final_path,
    )

    # Save training metrics log to outputs/logs/training_log.json
    log_path = os.path.join(train_cfg.get("log_dir", "outputs/logs"), "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[Log] Training history saved to: {log_path}")

    # Plot and save Training vs Validation Loss Curve (Figure 6)
    plt.figure(figsize=(9, 5))
    plt.plot(history["step"], history["train_loss"], label="Training Loss", color="#1f77b4", linewidth=2)
    plt.plot(history["step"], history["val_loss"], label="Validation Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.title("MiniGPT Training vs Validation Loss Curve (Figure 6)", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()

    fig6_path = "outputs/figures/figure6_loss_curve.png"
    plt.savefig(fig6_path, dpi=300)
    plt.savefig("outputs/figures/loss_curve.png", dpi=300)
    plt.close()
    print(f"[Figure 6] Loss curve saved to: {fig6_path}")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MiniGPT from scratch on Tiny Shakespeare")
    parser.add_argument("--config", type=str, default="configs/tiny_gpt.yaml", help="Path to config YAML")
    parser.add_argument("--max-iters", type=int, default=None, help="Override maximum iterations")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Target device (cpu/cuda)")
    args = parser.parse_args()

    overrides = {
        "max_iters": args.max_iters,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "device": args.device,
    }
    train(config_path=args.config, cli_overrides=overrides)
