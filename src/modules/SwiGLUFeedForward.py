import torch


class SwiGLUFeedForward(torch.nn.Module):
    def __init__(self, hidden_dim: int, inner_dim: int) -> None:
        """
        Args:
            hidden_dim: Dimension of input and output tensors.
            inner_dim: Dimension of the intermediate (inner) representation.
        """
        super().__init__()

        # timy computation optimization
        # Instead of two separate layers we project to 2x size immediately.
        self.gate_up_proj = torch.nn.Linear(hidden_dim, 2 * inner_dim, bias=False)
        self.output_proj = torch.nn.Linear(inner_dim, hidden_dim, bias=False)

        self.activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim].

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_dim].
        """
        assert len(x.shape) == 3, f"Expected 3D tensor, got shape {x.shape}"

        fused_output = self.gate_up_proj(x)
        # retrieve gate and value by splitting the last dimension
        gate, value = fused_output.chunk(2, dim=-1)

        hidden = self.activation(gate) * value

        return self.output_proj(hidden)
