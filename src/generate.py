"""
CLI Text Generation for MiniGPT.

Enables interactive or scripted text generation from trained checkpoints.
Matches the command line interface and output specification in Handout Section 12.1:
    python -m src.generate --prompt "The future of artificial intelligence" --max-new-tokens 80
"""

import os
import argparse
import torch

from src.tokenizer import CharacterTokenizer
from src.gpt import MiniGPT
from src.utils import load_config, get_device, load_checkpoint


def generate_text(
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = None,
    checkpoint_path: str = "checkpoints/best_model.pt",
    config_path: str = "configs/tiny_gpt.yaml",
    tokenizer_path: str = "checkpoints/tokenizer.json",
    device_pref: str = "auto",
):
    """
    Load checkpoint and generate text from a given prompt.
    """
    # 1. Determine device
    device = get_device(device_pref)

    # 2. Load configuration
    config = load_config(config_path)
    model_cfg = config["model"]

    # 3. Load Tokenizer
    if not os.path.exists(tokenizer_path):
        # Fallback to deriving tokenizer directly from data
        data_path = config["data"].get("data_path", "data/input.txt")
        with open(data_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        tokenizer = CharacterTokenizer.from_text(raw_text)
    else:
        tokenizer = CharacterTokenizer.load(tokenizer_path)

    # 4. Instantiate Model
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=model_cfg["context_length"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        ffn_dim=model_cfg.get("ffn_dim", 4 * model_cfg["d_model"]),
        dropout=0.0, # zero dropout for evaluation / generation
        bias=model_cfg.get("bias", False),
        pos_encoding=model_cfg.get("pos_encoding", "sinusoidal"),
        tie_weights=model_cfg.get("tie_weights", False),
    ).to(device)

    # 5. Load checkpoint weights
    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model, device=device)
    else:
        print(f"[Warning] Checkpoint not found at {checkpoint_path}. Running with random weights.")

    model.eval()

    # 6. Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    if len(prompt_tokens) == 0:
        # If prompt is empty, start with a newline token
        prompt_tokens = [0]
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    # 7. Generate tokens autoregressively
    generated_indices = model.generate(
        prompt_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )[0].tolist()

    generated_text = tokenizer.decode(generated_indices)

    # 8. Compute parameter count in Millions (for Section 12.1 output)
    num_params = model.get_num_params()
    param_str = f"{num_params / 1e6:.1f}M" if num_params >= 1e6 else f"{num_params / 1e3:.1f}K"

    # Print matching expected Handout Section 12.1 format
    print("\n" + "=" * 60)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    print("-" * 60)
    print(f"model parameters: {param_str} ({num_params:,} total parameters)")
    print(f"context length: {model_cfg['context_length']}")
    print(f"temperature: {temperature}")
    print("=" * 60 + "\n")

    return generated_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using trained MiniGPT")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence", help="Text prompt")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K sampling cutoff")
    parser.add_argument("--top-p", type=float, default=None, help="Top-P nucleus sampling cutoff")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="configs/tiny_gpt.yaml", help="Path to config YAML")
    parser.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json", help="Path to tokenizer JSON")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda)")
    args = parser.parse_args()

    generate_text(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        tokenizer_path=args.tokenizer,
        device_pref=args.device,
    )
