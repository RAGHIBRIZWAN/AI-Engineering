"""
MiniGPT: Causal Language Model Architecture.

Integrates:
1. Token Embedding Layer: Maps discrete token IDs to d_model continuous vectors.
2. Positional Encoding: Sinusoidal or Learned position embeddings.
3. Stack of Transformer Decoder Blocks.
4. Final Layer Normalization.
5. Language Model Head (lm_head): Linear projection from d_model to vocab_size logits.
6. Target-shifted Cross-Entropy Loss computation.
7. Autoregressive text generation with temperature, top-k, and nucleus (top-p) sampling.
"""

import math
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.positional_encoding import SinusoidalPositionalEncoding, LearnedPositionalEncoding
from src.transformer_block import TransformerBlock


class MiniGPT(nn.Module):
    """
    A miniature GPT-style autoregressive causal language model.
    """

    def __init__(
        self,
        vocab_size: int = 65,
        context_length: int = 128,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = False,
        pos_encoding: str = "sinusoidal",
        tie_weights: bool = False,
    ):
        """
        Args:
            vocab_size: Vocabulary size.
            context_length: Maximum sequence length (block size).
            d_model: Dimensionality of token representations.
            n_heads: Number of attention heads.
            n_layers: Number of Transformer decoder layers.
            ffn_dim: Hidden dimension of FFN (defaults to 4 * d_model).
            dropout: Dropout probability.
            bias: Whether linear projections include bias.
            pos_encoding: 'sinusoidal' or 'learned'.
            tie_weights: Whether to tie token embedding and lm_head weights.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim if ffn_dim is not None else 4 * d_model
        self.dropout_p = dropout
        self.pos_encoding_type = pos_encoding
        self.tie_weights = tie_weights

        # 1. Token embedding table
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 2. Positional encoding
        if pos_encoding == "learned":
            self.pos_encoding = LearnedPositionalEncoding(d_model, max_len=context_length, dropout=dropout)
        else:
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=context_length, dropout=dropout)

        # 3. Stack of Transformer Decoder Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                ffn_dim=self.ffn_dim,
                max_context_length=context_length,
                dropout=dropout,
                bias=bias,
                causal=True,
            )
            for _ in range(n_layers)
        ])

        # 4. Final Layer Normalization
        self.ln_f = nn.LayerNorm(d_model)

        # 5. Language Model Head: Projects contextual states back to vocabulary logits
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Optional: Weight tying (sharing weights between token embedding and LM head)
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # Initialize weights with standard normal distribution scaled by 0.02
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Standard Gaussian initialization for Transformer weights (mean=0, std=0.02)."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[List[torch.Tensor]]]:
        """
        Forward pass for next-token prediction.

        Args:
            idx: Tensor of token IDs with shape (batch_size, seq_len).
            targets: Optional ground-truth next tokens with shape (batch_size, seq_len).
            return_attentions: Whether to return attention maps across all layers.

        Returns:
            logits: Unnormalized log probabilities of shape (batch_size, seq_len, vocab_size).
            loss: Token-level cross-entropy loss if targets is provided, else None.
            all_attentions: List of attention maps per layer if return_attentions is True.
        """
        batch_size, seq_len = idx.size()
        assert seq_len <= self.context_length, (
            f"Cannot forward sequence of length {seq_len}, block size is {self.context_length}"
        )

        # 1. Token embeddings: (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        tok_emb = self.token_embedding(idx)

        # 2. Add positional encoding
        x = self.pos_encoding(tok_emb)

        # 3. Pass through sequential Transformer blocks
        all_attentions = [] if return_attentions else None
        for block in self.blocks:
            x, attn = block(x, return_attention=return_attentions)
            if return_attentions:
                all_attentions.append(attn)

        # 4. Final layer normalization
        x = self.ln_f(x)

        # 5. Project to vocabulary logits: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)

        # 6. Compute Cross-Entropy Loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten batch and sequence dimensions for cross_entropy:
            # logits: (batch_size * seq_len, vocab_size)
            # targets: (batch_size * seq_len)
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss, all_attentions

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation loop.

        Args:
            idx: Prompt token IDs of shape (batch_size, seq_len).
            max_new_tokens: Maximum number of tokens to append.
            temperature: Softmax temperature (>0). Lower is greedier, higher is more diverse.
            top_k: If set, retain only the top k highest probability tokens.
            top_p: If set, retain tokens up to cumulative probability threshold (nucleus sampling).

        Returns:
            Tensor of token IDs with shape (batch_size, seq_len + max_new_tokens).
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Crop current context if it exceeds the maximum context length
            idx_cond = idx if idx.size(1) <= self.context_length else idx[:, -self.context_length:]

            # Forward model to get logits for the current sequence
            logits, _, _ = self(idx_cond)

            # Pluck logits at the final position: shape (batch_size, vocab_size)
            logits = logits[:, -1, :]

            # Greedy decoding if temperature is 0
            if temperature == 0.0 or temperature < 1e-6:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Scale logits by temperature
                logits = logits / temperature

                # Optional: Top-K filtering
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    # Replace anything lower than the k-th highest logit with -infinity
                    logits[logits < v[:, [-1]]] = float("-inf")

                # Optional: Top-P (nucleus) filtering
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift indices to the right to keep also the first token above threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    
                    # Scatter back to original indices
                    for b in range(logits.size(0)):
                        indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                        logits[b, indices_to_remove] = float("-inf")

                # Softmax to get probability distribution
                probs = F.softmax(logits, dim=-1)

                # Sample next token from categorical distribution
                next_token = torch.multinomial(probs, num_samples=1)

            # Append sampled token to running sequence
            idx = torch.cat((idx, next_token), dim=1)

        return idx

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Return the number of parameters in the model.
        
        Args:
            non_embedding: If True, excludes token embedding parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
        return n_params
