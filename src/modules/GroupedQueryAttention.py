from typing import Optional
import torch
import torch.nn.functional as F
from src.modules.RotaryPositionalEmbedding import RotaryPositionalEmbedding


def calculate_attention(
        q: torch.Tensor,
        k: torch.Tensor,  # Expects [Batch, Num_Heads, Seq, Dim] (Expanded)
        v: torch.Tensor,  # Expects [Batch, Num_Heads, Seq, Dim] (Expanded)
        key_weights: torch.Tensor,
        scale: float,
        device: torch.device,
        mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Pure Scaled Dot-Product Attention kernel.
    Assumes q, k, v are already aligned (same number of heads) and rotated (RoPE).
    """

    w_reshaped = key_weights.view(1, -1, 1, 1)
    k_weighted = k * w_reshaped

    scores = torch.matmul(q, k_weighted.transpose(-2, -1)) * scale

    seq_len = q.shape[2]
    if mask is None:
        future_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
            diagonal=1
        )
        mask_tensor = torch.zeros((seq_len, seq_len), device=device, dtype=q.dtype)
        mask_tensor = mask_tensor.masked_fill(future_mask, float('-inf'))
    else:
        mask_tensor = mask

    scores = scores + mask_tensor

    attn_weights = F.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, v)

    return output


class GroupedQueryAttention(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            head_dim: int,
            num_kv_heads: Optional[int] = None
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.scale = head_dim ** -0.5

        assert num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_queries_per_kv = num_heads // self.num_kv_heads

        self.rope = RotaryPositionalEmbedding(head_dim=head_dim)

        self.q_projection = torch.nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_projection = torch.nn.Linear(hidden_dim, self.num_kv_heads * head_dim, bias=False)
        self.v_projection = torch.nn.Linear(hidden_dim, self.num_kv_heads * head_dim, bias=False)

        self.output_projection = torch.nn.Linear(num_heads * head_dim, hidden_dim, bias=False)
        self.key_weights = torch.nn.Parameter(torch.ones(num_heads))

    def _project_qkv_and_rope_qk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape

        q = self.q_projection(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_projection(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_projection(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        return q, k, v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 3
        batch, seq_len, _ = x.shape

        q, k, v = self._project_qkv_and_rope_qk(x)

        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        attention_output = calculate_attention(
            q=q,
            k=k,
            v=v,
            key_weights=self.key_weights,
            scale=self.scale,
            device=x.device
        )

        attention_output_transformed = attention_output.transpose(1, 2).contiguous().view(batch, seq_len,
                                                                                          self.hidden_dim)
        return self.output_projection(attention_output_transformed)
