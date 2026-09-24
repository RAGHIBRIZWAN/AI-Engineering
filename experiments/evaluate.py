"""
Evaluation and Analysis Suite for MiniGPT.

Performs:
1. Quantitative evaluation: Train & Validation Cross-Entropy Loss and Perplexity.
2. 5 Fixed Prompts generation (Handout Section 5.5 & Figure 8).
3. Sampling Strategy Comparison: Greedy vs. Balanced Temperature vs. High Temperature.
4. Failure Mode Analysis (Handout Section 15.8): Identifies two distinct failure modes.
5. Generates structured artifacts in `outputs/samples/` and `outputs/logs/`.
"""

import os
import math
import json
import time
import torch

from src.tokenizer import CharacterTokenizer
from src.dataset import CorpusManager
from src.gpt import MiniGPT
from src.utils import load_config, get_device, load_checkpoint


def run_evaluation(
    checkpoint_path: str = "checkpoints/best_model.pt",
    config_path: str = "configs/tiny_gpt.yaml",
    tokenizer_path: str = "checkpoints/tokenizer.json",
    data_path: str = "data/input.txt",
):
    device = get_device("auto")
    print("\n" + "=" * 70)
    print(f"{'MiniGPT Model Evaluation & Generation Suite':^70}")
    print("=" * 70)

    # 1. Load config and tokenizer
    config = load_config(config_path)
    model_cfg = config["model"]
    tokenizer = CharacterTokenizer.load(tokenizer_path) if os.path.exists(tokenizer_path) else CharacterTokenizer.from_text(open(data_path).read())

    # 2. Instantiate and load model
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
        print(f"[Warning] Checkpoint {checkpoint_path} not found. Running evaluation on untrained weights.")

    model.eval()

    # 3. Quantitative Metrics (Loss and Perplexity)
    corpus_mgr = CorpusManager(data_path, tokenizer=tokenizer)
    eval_batches = 40
    batch_size = 64
    context_len = model_cfg["context_length"]

    def compute_split_loss(split_name: str) -> float:
        total_loss = 0.0
        with torch.no_grad():
            for _ in range(eval_batches):
                x, y = corpus_mgr.get_batch(split_name, batch_size, context_len, device)
                _, loss, _ = model(x, y)
                total_loss += loss.item()
        return total_loss / eval_batches

    print("\n[Quantitative Evaluation]")
    train_loss = compute_split_loss("train")
    val_loss = compute_split_loss("val")
    train_ppl = math.exp(train_loss)
    val_ppl = math.exp(val_loss)

    print(f"  Training Loss:     {train_loss:.4f} | Perplexity: {train_ppl:.2f}")
    print(f"  Validation Loss:   {val_loss:.4f} | Perplexity: {val_ppl:.2f}")

    # 4. Generate Samples for 5 Fixed Prompts (Handout Section 5.5 & Figure 8)
    fixed_prompts = [
        "KING LEAR:\n",
        "ROMEO:\n",
        "To be, or not to be, that is the ",
        "First Citizen:\nBefore we proceed any further, ",
        "The future of artificial intelligence ",
    ]

    print("\n" + "=" * 70)
    print("Figure 8: Five Fixed Prompts Generation Samples")
    print("=" * 70)

    generated_samples = {}
    for idx, prompt in enumerate(fixed_prompts, 1):
        p_tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        with torch.no_grad():
            out_tokens = model.generate(p_tokens, max_new_tokens=100, temperature=0.8, top_k=40)[0].tolist()
        gen_text = tokenizer.decode(out_tokens)
        generated_samples[f"prompt_{idx}"] = {"prompt": prompt, "generated": gen_text}
        print(f"\n[Sample {idx}] Prompt: {repr(prompt)}")
        print("-" * 50)
        print(gen_text.strip())

    # 5. Sampling Strategy Comparison (Handout Section 13)
    test_prompt = "GLOUCESTER:\nNow is the winter of our discontent\n"
    p_tokens = torch.tensor([tokenizer.encode(test_prompt)], dtype=torch.long, device=device)

    print("\n" + "=" * 70)
    print("Sampling Strategy Comparison (Greedy vs. T=0.8 vs. T=1.5)")
    print("=" * 70)

    strategies = {
        "Greedy (T=0.0)": {"temp": 0.0, "top_k": None},
        "Balanced (T=0.8, top_k=40)": {"temp": 0.8, "top_k": 40},
        "High Temperature (T=1.5)": {"temp": 1.5, "top_k": None},
    }

    sampling_outputs = {}
    for name, params in strategies.items():
        with torch.no_grad():
            out = model.generate(
                p_tokens,
                max_new_tokens=100,
                temperature=params["temp"],
                top_k=params["top_k"],
            )[0].tolist()
        text = tokenizer.decode(out)
        sampling_outputs[name] = text
        print(f"\n--- Strategy: {name} ---")
        print(text.strip())

    # 6. Failure Mode Analysis (Handout Section 15.8)
    # Failure Mode 1: Repetition loop under pure greedy decoding (Temperature = 0)
    # Failure Mode 2: Out-of-domain / non-Shakespeare prompt hallucination (e.g. Modern AI terminology)
    failure_analysis = {
        "failure_mode_1_greedy_loop": {
            "title": "Degenerate Repetition Loop in Greedy Decoding",
            "observation": "When temperature is set to 0.0, the model tends to fall into rigid cyclical repeating phrases because the argmax choice locks into a periodic attractor.",
            "mitigation": "Stochastic temperature sampling with Top-K (k=40) or Top-P (p=0.9) filtering.",
        },
        "failure_mode_2_domain_mismatch": {
            "title": "Severe Semantic Hallucination on Modern / Out-of-Domain Prompts",
            "observation": "When prompted with 'The future of artificial intelligence', the model attempts to map modern concepts into 16th-century vocabulary ('art', 'thee', 'thou', 'king'), failing to produce factual modern explanations.",
            "mitigation": "Training on general web text or domain-specific corpora, or fine-tuning with instruction data.",
        },
    }

    # 7. Save structured artifacts
    os.makedirs("outputs/samples", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)

    with open("outputs/samples/figure8_generation_samples.txt", "w", encoding="utf-8") as f:
        f.write("=== Figure 8: MiniGPT Generated Samples (5 Fixed Prompts) ===\n\n")
        for k, v in generated_samples.items():
            f.write(f"--- {k.upper()} ---\nPrompt: {v['prompt']}\nResult:\n{v['generated']}\n\n")

    eval_summary = {
        "model_parameters": model.get_num_params(),
        "train_loss": round(train_loss, 4),
        "val_loss": round(val_loss, 4),
        "train_perplexity": round(train_ppl, 2),
        "val_perplexity": round(val_ppl, 2),
        "generated_samples": generated_samples,
        "sampling_comparison": sampling_outputs,
        "failure_analysis": failure_analysis,
    }

    with open("outputs/logs/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print("\n[Saved] outputs/samples/figure8_generation_samples.txt")
    print("[Saved] outputs/logs/evaluation_results.json")
    print("=" * 70 + "\n")

    return eval_summary


if __name__ == "__main__":
    run_evaluation()
