import torch


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        """
        Args:
            head_dim: Dimension of each attention head (must be even).
            max_seq_len: Maximum sequence length to precompute embeddings for.
            base: Base for computing rotation frequencies.

        WARNING: YOUR IMPLEMENTATION MUST PRECOMPUTE THE EMBEDDINGS
        """
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.head_dim = head_dim
        self.base = base

        cache_shape = (max_seq_len, head_dim)

        self.register_buffer("cos_cache", torch.zeros(cache_shape), persistent=True)
        self.register_buffer("sin_cache", torch.zeros(cache_shape), persistent=True)

        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len: int) -> None:
        theta_numerator = torch.arange(0, self.head_dim, 2).float()
        theta = 1.0 / (self.base ** (theta_numerator / self.head_dim))

        seq_idx = torch.arange(seq_len).float()

        idx_theta = torch.outer(seq_idx, theta)

        freqs_complex = torch.cat((idx_theta, idx_theta), dim=-1)

        self.cos_cache.copy_(freqs_complex.cos())
        self.sin_cache.copy_(freqs_complex.sin())

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, num_heads, seq_len, head_dim].
            start_pos: Starting position index.
        """
        batch, heads, seq_len, head_dim = x.shape

        idx_end = start_pos + seq_len
        cos = self.cos_cache[start_pos:idx_end]
        sin = self.sin_cache[start_pos:idx_end]

        cos = cos.view(1, 1, seq_len, head_dim)
        sin = sin.view(1, 1, seq_len, head_dim)

        x1 = x[..., :head_dim // 2]
        x2 = x[..., head_dim // 2:]

        rotated_x = torch.cat((-x2, x1), dim=-1)

        return (x * cos) + (rotated_x * sin)