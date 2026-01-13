from typing import Optional

import torch

from src.modules.TransformerBlock import TransformerBlock


class Transformer(torch.nn.Module):
    def __init__(
            self,
            vocab_size: int,
            n_layers: int,
            hidden_dim: int,
            ff_dim: int,
            num_heads: int,
            head_dim: int,
            use_sliding_window_alternating: bool = False,
            window_size: int = 128,
            use_moe: bool = False,
            num_experts: int = 8,
            top_k: int = 2,
            num_kv_heads: Optional[int] = None
    ) -> None:
        """
        Args:
            vocab_size: Size of the vocabulary.
            n_layers: Number of transformer layers.
            hidden_dim: Hidden dimension.
            ff_dim: Feed-forward inner dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension per attention head.
            use_sliding_window_alternating: Use sliding window on every other layer
            window_size: Size of sliding window.
            use_moe: Whether to use Mixture of Experts.
            num_experts: Number of experts (if MoE).
            top_k: Number of experts per token (if MoE).
            num_kv_heads: Number of KV heads for GQA.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)

        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    ff_dim=ff_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    use_sliding_window=(use_sliding_window_alternating and (i % 2 == 1)),
                    window_size=window_size,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    top_k=top_k,
                    num_kv_heads=num_kv_heads
                )
                for i in range(n_layers)
            ]
        )

        self.final_norm = torch.nn.RMSNorm(hidden_dim)
        self.output_proj = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token indices of shape [batch, seq_len].

        Returns:
            Logits of shape [batch, seq_len, vocab_size].
        """
        assert len(x.shape) == 2, f"Expected 2D input, got shape {x.shape}"

        embeddings = self.embedding(x)
        transformer_block_output = embeddings
        for layer in self.layers:
            transformer_block_output = layer(transformer_block_output)
        normed_output = self.final_norm(transformer_block_output)
        logits = self.output_proj(normed_output)

        return logits
