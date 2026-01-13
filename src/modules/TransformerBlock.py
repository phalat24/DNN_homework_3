from typing import Optional

import torch

from src.modules.GroupedQueryAttention import GroupedQueryAttention
from src.modules.MixtureOfExperts import MixtureOfExperts
from src.modules.SWAttention import SWAttention
from src.modules.SwiGLUFeedForward import SwiGLUFeedForward


class TransformerBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            ff_dim: int,
            num_heads: int,
            head_dim: int,
            use_sliding_window: bool = False,
            window_size: int = 128,
            use_moe: bool = False,
            num_experts: int = 8,
            top_k: int = 2,
            num_kv_heads: Optional[int] = None
    ) -> None:
        super().__init__()

        if use_sliding_window:
            self.attention = SWAttention(
                hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                window_size=window_size
            )
        else:
            self.attention = GroupedQueryAttention(
                hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                num_kv_heads=num_kv_heads
            )

        if use_moe:
            self.ffn = MixtureOfExperts(
                hidden_dim=hidden_dim, inner_dim=ff_dim,
                num_experts=num_experts, top_k=top_k
            )
        else:
            self.ffn = SwiGLUFeedForward(
                hidden_dim=hidden_dim, inner_dim=ff_dim
            )

        self.attention_norm = torch.nn.RMSNorm(hidden_dim)
        self.ffn_norm = torch.nn.RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x))

        out = h + self.ffn(self.ffn_norm(h))

        return out
