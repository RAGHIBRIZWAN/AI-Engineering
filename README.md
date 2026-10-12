# MiniGPT: Building a Small LLM from Scratch

[![Live API](https://img.shields.io/badge/Render-Live%20API-success?style=for-the-badge&logo=render)](https://ai-engineering-4r6q.onrender.com)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model%20Checkpoint-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/RaghibRizwan/minigpt-checkpoint)
[![Swagger Docs](https://img.shields.io/badge/Swagger-API%20Docs-blue?style=for-the-badge&logo=fastapi)](https://ai-engineering-4r6q.onrender.com/docs)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

**Authors / Team:**
- **Muhammad Umar** (Student ID: 23K-0023)
- **Raghib Rizwan Rabani** (Student ID: 23K-0012)
- **Syed Fahad Fahaeem Shah** (Student ID: 23K-0062)

A clean, modular, and fully tested implementation of a **decoder-only causal Transformer language model** built from first principles in PyTorch without using pretrained weights or high-level abstractions like Hugging Face `AutoModelForCausalLM`.

Trained on the **Tiny Shakespeare** dataset (~1.11 MB, 40,000 lines), this model demonstrates end-to-end understanding of scaled dot-product attention, multi-head projections, causal masking, positional encodings, autoregressive generation, and cloud-native API deployment.

---

## 🌐 Live Deployed Backend

The trained MiniGPT inference API is deployed and live on Render:

- **Live Service URL:** [`https://ai-engineering-4r6q.onrender.com`](https://ai-engineering-4r6q.onrender.com)
- **Interactive Swagger Documentation:** [`https://ai-engineering-4r6q.onrender.com/docs`](https://ai-engineering-4r6q.onrender.com/docs)
- **ReDoc Documentation:** [`https://ai-engineering-4r6q.onrender.com/redoc`](https://ai-engineering-4r6q.onrender.com/redoc)
- **Health Check Endpoint:** [`https://ai-engineering-4r6q.onrender.com/health`](https://ai-engineering-4r6q.onrender.com/health)
- **Model Info Endpoint:** [`https://ai-engineering-4r6q.onrender.com/info`](https://ai-engineering-4r6q.onrender.com/info)

### Quick Test via cURL

```bash
curl -X POST "https://ai-engineering-4r6q.onrender.com/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "To be, or not to be", "max_new_tokens": 50, "temperature": 0.8, "top_k": 40}'
```

**Live Response:**
```json
{
  "prompt": "To be, or not to be",
  "completion": " heart?\n\nKING EDWARD IV:\nWhat I chad I a",
  "full_text": "To be, or not to be heart?\n\nKING EDWARD IV:\nWhat I chad I a",
  "tokens_generated": 50,
  "latency_ms": 34435.81,
  "model_params": "0.81M"
}
```

### Quick Test via Python

```python
import requests

url = "https://ai-engineering-4r6q.onrender.com/generate"
payload = {
    "prompt": "ROMEO:\n",
    "max_new_tokens": 60,
    "temperature": 0.8,
    "top_k": 40
}
response = requests.post(url, json=payload)
print(response.json()["full_text"])
```

---

## 📦 Hugging Face Checkpoints & Model Downloader

Trained model weights and vocabulary are hosted on the Hugging Face Hub:
- **Hugging Face Repository:** [`RaghibRizwan/minigpt-checkpoint`](https://huggingface.co/RaghibRizwan/minigpt-checkpoint)

Instead of training from scratch, you can immediately download the best trained checkpoint (`best_model.pt`) and character tokenizer (`tokenizer.json`) using the automated script:

```bash
python download_model.py
```

This places the model checkpoint and tokenizer into `checkpoints/` ready for immediate local CLI text generation or FastAPI serving.

---

## 📁 Repository Structure

```
AI-Engineering/
├── README.md                      # Comprehensive project guide & live deployment documentation
├── pyproject.toml                 # Packaging, project metadata, and pytest configuration
├── requirements.txt               # Pinned Python package dependencies (including huggingface_hub)
├── download_model.py              # Automated Hugging Face model & tokenizer checkpoint downloader
├── colab_demo.ipynb               # End-to-end interactive Google Colab notebook
├── assignment.pdf                 # Original assignment handout specifications
├── configs/
│   └── tiny_gpt.yaml              # Hyperparameters & model architectural configuration
├── data/
│   ├── README.md                  # Dataset card & preprocessing documentation
│   ├── .gitkeep
│   └── input.txt                  # Tiny Shakespeare dataset
├── src/
│   ├── __init__.py
│   ├── tokenizer.py               # Deterministic character-level tokenizer
│   ├── dataset.py                 # Contiguous sequence batching & target shifting
│   ├── attention.py               # Scaled Dot-Product & Multi-Head Attention
│   ├── positional_encoding.py     # Sinusoidal & Learned Positional Encodings
│   ├── transformer_block.py       # Pre-LayerNorm Transformer Block & GELU FFN
│   ├── gpt.py                     # Full MiniGPT causal language model
│   ├── train.py                   # Training loop with warmup, cosine LR & logging
│   ├── generate.py                # Section 12.1 CLI text generation interface
│   ├── api.py                     # Production FastAPI REST inference service
│   └── utils.py                   # Checkpoint, logging, and hand calc utilities
├── tests/
│   ├── __init__.py
│   ├── test_attention.py          # Attention math & shape verification
│   ├── test_mask.py               # Causal masking & future token leakage tests
│   ├── test_shapes.py             # Layer-by-layer tensor shape trace tests
│   └── test_overfit.py            # Milestone 4 debugging gate: single batch overfit
├── experiments/
│   ├── __init__.py
│   ├── attention_visualization.py # Visual generation (Figures 1, 2, 3, 4, 7)
│   ├── ablation.py                # 4 controlled ablations (depth, heads, pos, context_length)
│   └── evaluate.py                # Quantitative evaluation metrics & 5 fixed prompt samples
├── outputs/
│   ├── figures/                   # All generated figures (Figure 1 through 10 + Architecture)
│   │   ├── architecture_diagram.png       # Complete Pre-LayerNorm Transformer schematic
│   │   ├── figure1_hand_calculation.png   # Worked numerical self-attention calculations
│   │   ├── figure2_causal_mask.png        # Causal vs. bidirectional attention masks
│   │   ├── figure3_multihead_shapes.png   # MHA tensor projection shapes
│   │   ├── figure4_positional_encoding.png# Sinusoidal heatmaps and frequencies
│   │   ├── figure5_model_summary.png      # Parameter breakdown & layer dimensions
│   │   ├── figure6_loss_curve.png         # Train/Val loss trajectory across 1,500 steps
│   │   ├── figure7_attention_map.png      # Attention maps across heads
│   │   ├── figure8_generation_samples.png # Qualitative text generation comparison
│   │   ├── figure9_ablation_results.png   # Ablation curves (depth, heads, PE, context)
│   │   └── figure10_cli_inference.png     # Screenshot trace of CLI text generation
│   ├── logs/
│   │   ├── training_log.json      # Step-by-step training & validation loss history
│   │   ├── evaluation_results.json# Loss, perplexity, and prompt sample benchmarks
│   │   └── ablation_results.json  # Quantitative metrics for all 4 ablations
│   └── samples/                   # Generated text samples across training milestones
├── checkpoints/
│   ├── best_model.pt              # Best model checkpoint (lowest validation loss)
│   ├── final_model.pt             # Final training step checkpoint
│   └── tokenizer.json             # Serialized character tokenizer vocabulary
└── report/
    ├── report.md                  # Comprehensive 10-section Technical Report
    └── report.pdf                 # Publication-quality compiled PDF report
```

---

## 🚀 Quickstart & Setup

### 1. Clone & Environment Setup
```bash
# Clone the repository
git clone https://github.com/RAGHIBRIZWAN/AI-Engineering.git
cd AI-Engineering

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pre-trained Weights
Download the trained weights directly from Hugging Face:
```bash
python download_model.py
```
*(Alternatively, train the model locally as shown below).*

---

## 🧪 Running the Unit Tests

The test suite validates architectural correctness, numerical consistency, and causal masking:
```bash
python -m pytest tests/ -v
```

**Test Suite Coverage:**
1. **`test_attention.py`**: Exact numeric equivalence between PyTorch attention and manual hand calculations; confirms softmax rows sum to $1.0$.
2. **`test_mask.py`**: Strict lower-triangular causal masking ($j > i \implies 0.0$ attention weight) and future-token perturbation invariance.
3. **`test_shapes.py`**: Dimensionality preservation across embedding, MHA, FFN, and full MiniGPT forward passes.
4. **`test_overfit.py`**: Milestone 4 debugging gate (loss drops from ~4.2 to < 0.25 on a single batch within 100 steps).

---

## 🏋️ Training the Model

Train the baseline 4-layer, 4-head MiniGPT model on Tiny Shakespeare:
```bash
python -m src.train --config configs/tiny_gpt.yaml
```

**Model & Training Specifications:**
| Parameter | Value |
| :--- | :--- |
| **Parameters** | 805,632 (~0.81M) |
| **Vocabulary Size** | 65 characters |
| **Context Length ($T$)** | 128 |
| **Embedding Dimension ($d_{\text{model}}$)** | 128 |
| **Heads ($n_{\text{heads}}$)** | 4 ($d_{\text{head}} = 32$) |
| **Layers ($n_{\text{layers}}$)** | 4 |
| **FFN Dimension ($d_{\text{ffn}}$)** | 512 ($4 \times d_{\text{model}}$) |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.99$, weight decay $= 0.01$) |
| **LR Schedule** | Linear Warmup (100 steps) + Cosine Decay (1500 steps) to $10^{-4}$ |
| **Initial Loss** | ~4.20 (theoretical $-\ln(1/65) = 4.17$) |
| **Final Val Loss** | 1.8212 |
| **Validation Perplexity** | 6.18 |
| **Training Time** | ~5-8 minutes on CPU |

Checkpoints are saved to `checkpoints/best_model.pt` and loss curves are plotted to `outputs/figures/figure6_loss_curve.png`.

---

## 💬 Text Generation (CLI)

Generate text locally from any prompt matching Handout Section 12.1:
```bash
python -m src.generate --prompt "The future of artificial intelligence" --max-new-tokens 80 --temperature 0.8
```

**Sample Output:**
```
Prompt: The future of artificial intelligence 
Generated: The future of artificial intelligence forther,
And all thee lass of thee been to Rereit your blorford.

HENRY BOKE:
You make would Some I

model parameters: 0.8M (805,632 total parameters)
context length: 128
temperature: 0.8
```

---

## ⚡ REST API Deployment (Local & Cloud)

### 1. Cloud-Deployed API (Render)
No local installation required! Interact with the live service directly:
- **Base URL:** `https://ai-engineering-4r6q.onrender.com`
- **Swagger UI:** `https://ai-engineering-4r6q.onrender.com/docs`
- **ReDoc:** `https://ai-engineering-4r6q.onrender.com/redoc`

### 2. Local API Serving
Run the FastAPI application locally with Uvicorn:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```
Interactive docs will be available at `http://localhost:8000/docs`.

### API Specification:
- `GET /`: Service metadata and model parameter count.
- `GET /health`: Health status and compute device.
- `GET /info`: Architectural dimensions, layer counts, and context length.
- `POST /generate`: Autoregressive generation endpoint.
  ```json
  {
    "prompt": "To be, or not to be",
    "max_new_tokens": 80,
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9
  }
  ```

---

## 🔬 Experiments & Ablation Studies

### 1. Visualizations
Generate all architectural, mask, and attention heatmaps:
```bash
python -m experiments.attention_visualization
```
Generates Figures 1, 2, 3, 4, and 7 in `outputs/figures/`.

### 2. Four Controlled Ablations
```bash
python -m experiments.ablation
```
Executes the four required controlled experiments:
1. **Depth:** 2 vs. 4 vs. 6 Transformer layers.
2. **Heads:** 2 vs. 4 vs. 8 Attention heads.
3. **Positional Encoding:** Sinusoidal vs. Learned positional encodings.
4. **Context Length:** 64 vs. 128 vs. 256 tokens.

Summary metrics are saved to `outputs/logs/ablation_results.json` and plotted in `outputs/figures/figure9_ablation_results.png`.

### 3. Quantitative Evaluation Benchmark
```bash
python -m experiments.evaluate
```
Evaluates cross-entropy loss, perplexity across train and validation splits, generates on 5 fixed benchmark prompts, and evaluates decoding strategies (Greedy vs. Temperature vs. Top-K).

---

## 📐 Numerical Hand Calculation Verification

For a 3-token sequence with $d_k=2, d_v=2$:
- **Queries ($Q$):**
  $$\begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 2.0 \\ 1.0 & 1.0 \end{bmatrix}$$
- **Keys ($K$):**
  $$\begin{bmatrix} 1.0 & 1.0 \\ 0.0 & 1.0 \\ 2.0 & 0.0 \end{bmatrix}$$
- **Values ($V$):**
  $$\begin{bmatrix} 1.0 & 2.0 \\ 3.0 & 0.0 \\ 0.0 & 4.0 \end{bmatrix}$$

**Masked Causal Attention Weights:**
$$\begin{bmatrix} 1.000 & 0.000 & 0.000 \\ 0.500 & 0.500 & 0.000 \\ 0.401 & 0.198 & 0.401 \end{bmatrix}$$

**Masked Output ($W_{\text{causal}} \times V$):**
$$\begin{bmatrix} 1.000 & 2.000 \\ 2.000 & 1.000 \\ 0.994 & 2.406 \end{bmatrix}$$

This manual step is numerically validated against PyTorch tensor operations in `tests/test_attention.py`.

---

## ⚠️ Known Limitations & Failure Modes

1. **Greedy Sampling Degeneracy:** Greedy decoding ($T=0.0$) collapses into repetitive loops (e.g., `"The shall the shall the shall..."`) due to periodic attractor states in argmax selection. *Mitigation:* Stochastic sampling with $T=0.8$ and Top-K ($k=40$).
2. **Domain Specialization / Semantic Mismatch:** Trained exclusively on Shakespeare, the model maps modern terms (e.g., "artificial intelligence") into archaic syntax and vocabulary. *Mitigation:* Pretraining on diverse modern corpora or instruction fine-tuning.
3. **Context Window Constraint:** With a 128-character context window, long-range discourse and character consistency across scenes cannot be preserved.

---

## 📄 Technical Report & Deliverables

- **Full Report (Markdown):** [`report/report.md`](report/report.md)
- **Publication PDF Report:** [`report/report.pdf`](report/report.pdf)
- **Google Colab Notebook:** [`colab_demo.ipynb`](colab_demo.ipynb)
- **Medium Article:** [Building a Small LLM from Scratch — Transformer Foundations, Causal Language Modeling & Scaled Dot-Product Attention](https://medium.com/@muhammad.umar26804/building-a-small-llm-from-scratch-a-deep-dive-into-transformer-foundations-causal-language-7b7fb702a878)

---

## 📜 License
MIT License. Open for educational and research use.
