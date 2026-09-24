"""
Character-Level Tokenizer for MiniGPT.

This module implements a deterministic character-level tokenizer.
Each unique character in the domain corpus is mapped to a unique integer ID.
Out-of-vocabulary characters are mapped to a special `<unk>` token.
"""

import json
from typing import List, Dict, Optional


class CharacterTokenizer:
    """
    A deterministic character-level tokenizer.
    
    Vocabulary:
    - Unique characters sorted lexicographically for exact reproducibility.
    - Optional special tokens: `<unk>` (unknown character) and `<pad>` (padding).
    """

    def __init__(self, chars: Optional[List[str]] = None, unk_token: str = "<unk>"):
        """
        Initialize the tokenizer.
        
        Args:
            chars: Optional list of unique characters. If provided, vocabulary is built immediately.
            unk_token: String representing unknown characters.
        """
        self.unk_token = unk_token
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        
        if chars is not None:
            self._build_vocab(chars)

    def _build_vocab(self, chars: List[str]):
        """Build deterministic character mappings from a list of characters."""
        # Sort characters to guarantee identical token IDs across runs
        unique_chars = sorted(list(set(chars)))
        
        # Build mappings
        self.char_to_id = {ch: i for i, ch in enumerate(unique_chars)}
        
        # Add unknown token at the end if not present
        if self.unk_token not in self.char_to_id:
            self.char_to_id[self.unk_token] = len(self.char_to_id)
            
        # Reverse mapping: ID -> character
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}
        self.unk_id = self.char_to_id[self.unk_token]

    @classmethod
    def from_text(cls, text: str, unk_token: str = "<unk>") -> "CharacterTokenizer":
        """
        Create and fit a tokenizer directly from raw text.
        
        Args:
            text: The text corpus to derive the vocabulary from.
            unk_token: The token representation for unknown characters.
        """
        chars = list(set(text))
        return cls(chars=chars, unk_token=unk_token)

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size including special tokens."""
        return len(self.char_to_id)

    def encode(self, text: str) -> List[int]:
        """
        Convert a text string into a list of integer token IDs.
        
        Args:
            text: Input string.
            
        Returns:
            List of integer token IDs.
        """
        return [self.char_to_id.get(ch, self.unk_id) for ch in text]

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert a list of integer token IDs back into a text string.
        
        Args:
            token_ids: List of integer IDs.
            
        Returns:
            Decoded string.
        """
        return "".join([self.id_to_char.get(i, self.unk_token) for i in token_ids])

    def save(self, filepath: str):
        """Save the tokenizer vocabulary to a JSON file."""
        data = {
            "unk_token": self.unk_token,
            "char_to_id": self.char_to_id,
            "vocab_size": self.vocab_size
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CharacterTokenizer":
        """Load tokenizer vocabulary from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        tokenizer = cls(unk_token=data["unk_token"])
        tokenizer.char_to_id = data["char_to_id"]
        # Convert string keys back to integers for id_to_char
        tokenizer.id_to_char = {int(i): ch for ch, i in tokenizer.char_to_id.items()}
        tokenizer.unk_id = tokenizer.char_to_id[tokenizer.unk_token]
        return tokenizer

    def __repr__(self) -> str:
        return f"CharacterTokenizer(vocab_size={self.vocab_size}, unk_token='{self.unk_token}')"
