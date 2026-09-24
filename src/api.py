"""
FastAPI REST API for MiniGPT.

Exposes endpoints for serving text generation inferences from the trained model:
- GET  /         : Service status and model metadata.
- GET  /health   : Health check endpoint.
- POST /generate : Generate text from a prompt with configurable hyperparameters.
- GET  /info     : Detailed architecture and parameter count information.
"""

import os
import time
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch

from src.tokenizer import CharacterTokenizer
from src.gpt import MiniGPT
from src.utils import load_config, get_device, load_checkpoint

app = FastAPI(
    title="MiniGPT REST API",
    description="A lightweight REST API serving the scratch-built MiniGPT causal language model.",
    version="1.0.0",
)

# Global model state
state = {
    "model": None,
    "tokenizer": None,
    "config": None,
    "device": None,
    "num_params": 0,
}


class GenerateRequest(BaseModel):
    prompt: str = Field(..., example="To be, or not to be")
    max_new_tokens: int = Field(80, ge=1, le=500, description="Number of tokens to generate")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Softmax sampling temperature")
    top_k: Optional[int] = Field(40, ge=0, description="Top-k filtering threshold")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling threshold")


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    full_text: str
    tokens_generated: int
    latency_ms: float
    model_params: str


def load_model_and_tokenizer():
    """Load configuration, tokenizer, and model weights into memory."""
    config_path = "configs/tiny_gpt.yaml"
    checkpoint_path = "checkpoints/best_model.pt"
    tokenizer_path = "checkpoints/tokenizer.json"

    device = get_device("auto")
    config = load_config(config_path) if os.path.exists(config_path) else {}
    model_cfg = config.get("model", {
        "context_length": 128,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 4,
        "ffn_dim": 512,
        "dropout": 0.0,
        "bias": False,
        "pos_encoding": "sinusoidal",
        "tie_weights": False,
    })

    if os.path.exists(tokenizer_path):
        tokenizer = CharacterTokenizer.load(tokenizer_path)
    else:
        # Fallback to dataset text
        data_path = "data/input.txt"
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                tokenizer = CharacterTokenizer.from_text(f.read())
        else:
            # Minimal ASCII fallback
            tokenizer = CharacterTokenizer(chars=[chr(i) for i in range(32, 127)])

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        context_length=model_cfg.get("context_length", 128),
        d_model=model_cfg.get("d_model", 128),
        n_heads=model_cfg.get("n_heads", 4),
        n_layers=model_cfg.get("n_layers", 4),
        ffn_dim=model_cfg.get("ffn_dim", 512),
        dropout=0.0,
        bias=model_cfg.get("bias", False),
        pos_encoding=model_cfg.get("pos_encoding", "sinusoidal"),
        tie_weights=model_cfg.get("tie_weights", False),
    ).to(device)

    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model, device=device)
    elif os.path.exists("checkpoints/final_model.pt"):
        load_checkpoint("checkpoints/final_model.pt", model, device=device)

    model.eval()

    state["model"] = model
    state["tokenizer"] = tokenizer
    state["config"] = config
    state["device"] = device
    state["num_params"] = model.get_num_params()
    print(f"[API] MiniGPT loaded successfully with {state['num_params']:,} parameters on {device}.")


@app.on_event("startup")
def startup_event():
    load_model_and_tokenizer()


@app.get("/")
def root():
    return {
        "service": "MiniGPT Causal Language Model API",
        "status": "online",
        "parameters": f"{state['num_params']:,}",
        "endpoints": {
            "health": "/health",
            "info": "/info",
            "generate": "/generate (POST)",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if state["model"] is not None else "initializing",
        "model_loaded": state["model"] is not None,
        "device": str(state["device"]),
    }


@app.get("/info")
def info():
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {
        "architecture": "Decoder-only Causal Transformer (MiniGPT)",
        "total_parameters": state["num_params"],
        "context_length": state["model"].context_length,
        "d_model": state["model"].d_model,
        "n_heads": state["model"].n_heads,
        "n_layers": state["model"].n_layers,
        "vocab_size": state["tokenizer"].vocab_size,
        "pos_encoding": state["model"].pos_encoding_type,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if state["model"] is None or state["tokenizer"] is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    start_time = time.time()
    tokenizer: CharacterTokenizer = state["tokenizer"]
    model: MiniGPT = state["model"]
    device = state["device"]

    # Encode prompt
    prompt_tokens = tokenizer.encode(req.prompt)
    if len(prompt_tokens) == 0:
        prompt_tokens = [0]

    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        out_tokens = model.generate(
            prompt_tensor,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )[0].tolist()

    full_text = tokenizer.decode(out_tokens)
    completion = tokenizer.decode(out_tokens[len(prompt_tokens):])
    latency_ms = (time.time() - start_time) * 1000.0

    return GenerateResponse(
        prompt=req.prompt,
        completion=completion,
        full_text=full_text,
        tokens_generated=len(out_tokens) - len(prompt_tokens),
        latency_ms=round(latency_ms, 2),
        model_params=f"{state['num_params'] / 1e6:.2f}M",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="127.0.0.1", port=8000, reload=False)
