# Tiny Shakespeare Dataset

## Overview
This directory contains the training and evaluation corpus used for training our miniature GPT causal language model from scratch.

- **File**: `input.txt`
- **Source**: [Andrej Karpathy's char-rnn / Hugging Face Tiny Shakespeare](https://huggingface.co/datasets/tiny_shakespeare)
- **Size**: ~1.11 MB (1,115,394 characters)
- **Content**: ~40,000 lines of Shakespeare plays (Coriolanus, Henry IV, Romeo and Juliet, etc.)
- **Vocabulary**: 65 unique characters (including alphanumeric characters, punctuation, whitespace, and newlines).

## Dataset Preprocessing & Splits
1. **Tokenization**: Deterministic character-level tokenization mapping each character to an integer ID $0 \le \text{ID} < 65$.
2. **Train/Validation Split**: Contiguous 90% training split (~1,003,854 characters) and 10% validation split (~111,540 characters).
3. **Sequence Construction**: Slices of length `context_length` (e.g., 128) paired with targets shifted by +1 token.
