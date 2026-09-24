"""
Dataset and Data Loading utilities for MiniGPT.

This module handles:
1. Loading the raw text corpus (Tiny Shakespeare).
2. Splitting data into training (90%) and validation (10%) splits.
3. Constructing contiguous sequences of fixed context length with shifted next-token targets:
     x: tokens[t : t + context_length]
     y: tokens[t + 1 : t + context_length + 1]
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
from src.tokenizer import CharacterTokenizer


class TextDataset(Dataset):
    """
    PyTorch Dataset providing contiguous token sequences with next-token targets.
    """

    def __init__(self, data: torch.Tensor, context_length: int):
        """
        Args:
            data: 1D torch.Tensor of token IDs.
            context_length: Length of input sequences.
        """
        self.data = data
        self.context_length = context_length

    def __len__(self) -> int:
        # Total number of full (context_length + 1) chunks available
        return max(0, len(self.data) - self.context_length)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input sequence of length context_length
        x = self.data[idx : idx + self.context_length]
        # Target sequence shifted by +1 token
        y = self.data[idx + 1 : idx + self.context_length + 1]
        return x, y


class CorpusManager:
    """
    Manages loading, tokenizing, and generating batches from a text file.
    """

    def __init__(
        self,
        filepath: str,
        tokenizer: Optional[CharacterTokenizer] = None,
        train_split: float = 0.9,
    ):
        """
        Args:
            filepath: Path to raw text file (e.g., 'data/input.txt').
            tokenizer: Pre-fitted tokenizer or None (in which case a new one is built).
            train_split: Fraction of text to use for training (default 0.9).
        """
        with open(filepath, "r", encoding="utf-8") as f:
            self.raw_text = f.read()

        if tokenizer is None:
            self.tokenizer = CharacterTokenizer.from_text(self.raw_text)
        else:
            self.tokenizer = tokenizer

        # Convert entire text to a 1D tensor of token IDs
        all_tokens = self.tokenizer.encode(self.raw_text)
        self.data_tensor = torch.tensor(all_tokens, dtype=torch.long)

        # Split into training and validation sets
        split_idx = int(len(self.data_tensor) * train_split)
        self.train_data = self.data_tensor[:split_idx]
        self.val_data = self.data_tensor[split_idx:]

        self.train_dataset = TextDataset(self.train_data, context_length=128)
        self.val_dataset = TextDataset(self.val_data, context_length=128)

    def get_batch(
        self,
        split: str,
        batch_size: int,
        context_length: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a random mini-batch of inputs x and targets y.

        Args:
            split: 'train' or 'val'.
            batch_size: Number of sequences in the batch.
            context_length: Sequence length.
            device: Target device ('cpu' or 'cuda').

        Returns:
            Tuple of (x, y) where:
              x has shape (batch_size, context_length)
              y has shape (batch_size, context_length)
        """
        data = self.train_data if split == "train" else self.val_data
        
        # High index limit ensuring we can slice (context_length + 1)
        max_idx = len(data) - context_length - 1
        if max_idx <= 0:
            raise ValueError(f"Data length ({len(data)}) is too short for context_length ({context_length}).")

        # Pick random starting indices
        random_indices = torch.randint(0, max_idx, (batch_size,))

        # Extract contiguous chunks
        x = torch.stack([data[i : i + context_length] for i in random_indices])
        y = torch.stack([data[i + 1 : i + context_length + 1] for i in random_indices])

        return x.to(device), y.to(device)
