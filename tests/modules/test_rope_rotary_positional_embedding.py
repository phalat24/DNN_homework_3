import unittest

import torch

from src.modules.RotaryPositionalEmbedding import RotaryPositionalEmbedding

class MyTestCase(unittest.TestCase):
    @torch.no_grad()
    def test_rope(self) -> None:
        """Test RoPE applies correct rotations."""
        head_dim = 4
        max_seq_len = 8
        batch, num_heads, seq_len = 2, 2, 4

        rope = RotaryPositionalEmbedding(head_dim, max_seq_len)
        x = torch.ones(batch, num_heads, seq_len, head_dim)

        result = rope(x)

        expected = torch.tensor(
            [[[[ 1.0000,  1.0000,  1.0000,  1.0000],
              [-0.3012,  0.9900,  1.3818,  1.0099],
              [-1.3254,  0.9798,  0.4932,  1.0198],
              [-1.1311,  0.9696, -0.8489,  1.0295]],

             [[ 1.0000,  1.0000,  1.0000,  1.0000],
              [-0.3012,  0.9900,  1.3818,  1.0099],
              [-1.3254,  0.9798,  0.4932,  1.0198],
              [-1.1311,  0.9696, -0.8489,  1.0295]]],


            [[[ 1.0000,  1.0000,  1.0000,  1.0000],
              [-0.3012,  0.9900,  1.3818,  1.0099],
              [-1.3254,  0.9798,  0.4932,  1.0198],
              [-1.1311,  0.9696, -0.8489,  1.0295]],

             [[ 1.0000,  1.0000,  1.0000,  1.0000],
              [-0.3012,  0.9900,  1.3818,  1.0099],
              [-1.3254,  0.9798,  0.4932,  1.0198],
              [-1.1311,  0.9696, -0.8489,  1.0295]]]]
        )

        assert result.shape == x.shape, f"Shape mismatch: {result.shape} vs {x.shape}"
        assert torch.allclose(result, expected, atol=1e-4), "Error in ROPE"

