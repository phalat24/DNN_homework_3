import torch

from src.modules.GroupedQueryAttention import GroupedQueryAttention, calculate_attention


def create_sliding_window_mask(
        seq_len: int,
        window_size: int,
        device: torch.device,
        dtype: torch.dtype
) -> torch.Tensor:
    """
    Creates a sliding window mask efficiently.
    """

    ones = torch.ones((seq_len, seq_len), device=device, dtype=torch.bool)
    mask_combined = torch.triu(ones, diagonal=1) | torch.tril(ones, diagonal=-(window_size + 1))

    return torch.zeros((seq_len, seq_len), device=device, dtype=dtype).masked_fill(mask_combined, float('-inf'))


def calculate_sliding_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_weights: torch.Tensor,
        scale: float,
        device: torch.device,
        window_size: int
) -> torch.Tensor:
    """
    Computes Scaled Dot-Product Attention with a Sliding Window Mask constraint.

    This function acts as a wrapper around the core attention kernel, injecting
    a banded mask that restricts attention to the range [i - window_size, i].

    Args:
        q: Query tensor of shape [batch, num_heads, seq_len, head_dim].
           Must have RoPE applied.
        k: Key tensor of shape [batch, num_heads, seq_len, head_dim].
           Must be ALREADY expanded (via GQA) to match num_heads.
           Must have RoPE applied.
        v: Value tensor of shape [batch, num_heads, seq_len, head_dim].
           Must be ALREADY expanded (via GQA) to match num_heads.
        key_weights: Per-head scalar weights of shape [num_heads].
        scale: Scaling factor (typically 1 / sqrt(head_dim)).
        device: Device to create the mask on.
        window_size: The lookback window size (w). Attention beyond w steps back is masked.

    Returns:
        Output tensor of shape [batch, num_heads, seq_len, head_dim].
    """

    batch, num_heads, seq_len, head_dim = q.shape

    sliding_window_mask = create_sliding_window_mask(
        seq_len=seq_len,
        window_size=window_size,
        device=device,
        dtype=q.dtype
    )

    return calculate_attention(
        q=q,
        k=k,
        v=v,
        key_weights=key_weights,
        scale=scale,
        device=device,
        mask=sliding_window_mask
    )


class SWAttention(GroupedQueryAttention):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            head_dim: int,
            window_size: int
    ) -> None:
        """
        Args:
            hidden_dim: Input/output dimension.
            num_heads: Number of attention heads.
            head_dim: Dimension of each head.
            window_size: Size of the sliding window.
        """
        super().__init__(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_heads  # Standard MHA for sliding window
        )
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim].

        Returns:
            Output tensor of shape [batch, seq_len, hidden_dim].
        """
        assert len(x.shape) == 3
        batch, seq_len, _ = x.shape

        q, k, v = self._project_qkv_and_rope_qk(x)

        attention_output = calculate_sliding_attention(
            q=q,
            k=k,
            v=v,
            key_weights=self.key_weights,
            scale=self.scale,
            device=x.device,
            window_size=self.window_size
        )

        attention_transformed = attention_output.transpose(1, 2).contiguous().view(batch, seq_len,
                                                                                   self.num_heads * self.head_dim)
        return self.output_projection(attention_transformed)
