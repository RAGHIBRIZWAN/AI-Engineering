# MiniGPT: Building a Small LLM from Scratch

A clean, modular, and fully tested implementation of a **decoder-only causal Transformer language model** built from first principles in PyTorch without using pretrained weights or high-level abstractions like Hugging Face `AutoModelForCausalLM`.

Trained on the **Tiny Shakespeare** dataset (~1.11 MB, 40,000 lines), this model demonstrates end-to-end understanding of scaled dot-product attention, multi-head projections, causal masking, positional encodings, and autoregressive generation.

---

## Repository Structure

```
transformer-gpt-assignment/
├── README.md                      # Complete setup, training, and inference guide
├── pyproject.toml                 # Package definition and build metadata
├── requirements.txt               # Pinned Python package dependencies
├── colab_demo.ipynb               # Interactive Google Colab notebook
├── configs/
│   └── tiny_gpt.yaml              # Hyperparameters & model configuration
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
│   ├── attention_visualization.py # Figures 1, 2, 3, 4, 7 visual generation
│   ├── ablation.py                # 3 controlled ablations (depth, heads, pos)
│   └── evaluate.py                # Quantitative metrics & 5 fixed prompt samples
├── outputs/
│   ├── figures/                   # All required figures (Figure 1 through 10)
│   ├── logs/                      # Training history & ablation JSON metrics
│   └── samples/                   # Generated text samples
├── checkpoints/
│   ├── best_model.pt              # Best model checkpoint (lowest val loss)
│   ├── final_model.pt             # Final model checkpoint
│   └── tokenizer.json             # Serialized character tokenizer vocabulary
└── report/
    ├── report.md                  # Comprehensive 10-section Technical Report
    └── report.pdf                 # Publication-quality compiled PDF report
```

---

## Quickstart & Setup

### 1. Prerequisites & Environment
Ensure you have Python 3.9+ installed:
```bash
# Clone the repository
git clone <repo-url>
cd transformer-gpt-assignment

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Unit Tests

All 10 architecture and numerical verification tests can be executed via:
```bash
python -m pytest tests/ -v
```

This verifies:
1. `test_attention.py`: Exact numeric equivalence between PyTorch attention and hand calculations; probability sums.
2. `test_mask.py`: Strict lower-triangular masking ($j > i \implies 0.0$ attention weight) and future-token perturbation invariance.
3. `test_shapes.py`: Dimensionality checks across embedding, MHA, FFN, and full MiniGPT forward passes.
4. `test_overfit.py`: Overfitting a single batch (loss drops from ~4.2 to < 0.25 within 100 steps).

---

## Training the Model

Train the baseline 4-layer, 4-head MiniGPT model on Tiny Shakespeare:
```bash
python -m src.train --config configs/tiny_gpt.yaml
```

**Training Highlights:**
- **Initial Loss:** ~4.20 (matches theoretical $-\ln(1/65) = 4.17$)
- **Final Validation Loss:** ~1.83
- **Perplexity:** ~6.2
- **Wall-Clock Time:** ~5-8 minutes on modern multi-core CPUs.
- Automatically saves `checkpoints/best_model.pt` and renders `outputs/figures/figure6_loss_curve.png`.

---

## Text Generation (CLI)

Generate text from any prompt using the CLI interface matching Handout Section 12.1:
```bash
python -m src.generate --prompt "The future of artificial intelligence" --max-new-tokens 80 --temperature 0.8
```

**Example Terminal Output:**
```
Prompt: The future of artificial intelligence
Generated: The future of artificial intelligence is an art unknown to these fair lords; but by my sword, thy counsel shall be heard...

model parameters: 0.8M (805,632 total parameters)
context length: 128
temperature: 0.8
```

---

## REST API Deployment (FastAPI)

Launch the REST API server:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Interactive Swagger API documentation is available at `http://localhost:8000/docs`.

### API Endpoints:
- `GET /`: Service metadata and total parameter count.
- `GET /health`: Health and device telemetry.
- `GET /info`: Architectural dimensions and vocabulary size.
- `POST /generate`:
  ```json
  {
    "prompt": "To be, or not to be",
    "max_new_tokens": 80,
    "temperature": 0.8,
    "top_k": 40
  }
  ```

---

## Experiments & Visualizations

### 1. Generate Figures (Attention Maps, Causal Mask, Positional Encoding)
```bash
python -m experiments.attention_visualization
```
Generates:
- `outputs/figures/figure1_hand_calculation.png` (Worked numeric attention)
- `outputs/figures/figure2_causal_mask.png` (GPT Causal vs BERT Bidirectional mask)
- `outputs/figures/figure3_multihead_shapes.png` (MHA Tensor-shape trace)
- `outputs/figures/figure4_positional_encoding.png` (Sinusoidal frequencies & heatmap)
- `outputs/figures/figure7_attention_map.png` (Trained model attention head heatmaps)

### 2. Controlled Ablation Studies
```bash
python -m experiments.ablation
```
Runs the 3 required ablations:
1. **Depth:** 2 vs. 4 vs. 6 Transformer blocks.
2. **Heads:** 2 vs. 4 vs. 8 Attention heads.
3. **Positional Encoding:** Sinusoidal vs. Learned.
Saves comparison plot to `outputs/figures/figure9_ablation_results.png`.

### 3. Quantitative Evaluation & 5 Fixed Prompts
```bash
python -m experiments.evaluate
```
Evaluates cross-entropy loss, perplexity, generation on 5 fixed prompts, sampling strategies, and documents failure modes.

---

## Hand Calculations (Handout Section 6)

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

---

## Known Limitations & Failure Modes

1. **Greedy Sampling Degeneracy:** Greedy decoding ($T=0.0$) tends to fall into repetitive cyclic attractors because argmax reinforces recurring loops. Mitigation: use temperature sampling ($T=0.8$) with Top-K ($K=40$).
2. **Domain Limitation:** The model was trained exclusively on Elizabethan English (Shakespeare). Modern prompts (e.g. "artificial intelligence") result in Shakespearean phrasing rather than factual modern content.
3. **Context Length Constraint:** With a context window of 128 tokens, the model cannot maintain multi-paragraph thematic coherence.

---

## License
MIT License. Open for educational and research use.
