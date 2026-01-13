import unittest

import torch

from src.modules.GroupedQueryAttention import calculate_attention
from src.modules.RotaryPositionalEmbedding import RotaryPositionalEmbedding


class MyTestCase(unittest.TestCase):
    @torch.no_grad()
    def test_calculate_attention(self) -> None:
        """Test the calculate_attention function independently of module weights."""
        torch.manual_seed(42)
        batch, seq_len = 2, 4
        num_heads, num_kv_heads, head_dim = 4, 2, 4

        q = torch.randn(batch, num_heads, seq_len, head_dim)
        k = torch.randn(batch, num_kv_heads, seq_len, head_dim)
        v = torch.randn(batch, num_kv_heads, seq_len, head_dim)

        key_weights = torch.randn(num_heads)
        rope = RotaryPositionalEmbedding(head_dim)
        scale = head_dim ** -0.5

        num_queries_per_kv = num_heads // num_kv_heads

        output = calculate_attention(
            q=rope(q),
            k=rope(k).repeat_interleave(num_queries_per_kv, dim=1),
            v=v.repeat_interleave(num_queries_per_kv, dim=1),
            key_weights=key_weights,
            scale=scale,
            device=q.device
        )

        assert output.shape == (batch, num_heads, seq_len, head_dim), \
            f"Wrong output shape: {output.shape}, expected {(batch, num_heads, seq_len, head_dim)}"

        expected = torch.tensor(
            [[[[1.7744, -0.9216, 0.9624, -0.3370],
               [0.3143, -0.2881, 0.7230, 0.4999],
               [0.3844, 0.4978, 0.3157, 0.0305],
               [0.3285, 0.6624, 0.5529, 0.1313]],

              [[1.7744, -0.9216, 0.9624, -0.3370],
               [-0.0153, -0.1452, 0.6690, 0.6888],
               [0.1277, 0.6668, 0.2440, 0.1472],
               [0.3011, 0.7186, 0.5328, 0.1269]],

              [[-0.8146, -1.0212, -0.4949, -0.5923],
               [-0.2359, -0.1480, -0.2879, -1.6233],
               [-0.0641, 0.5656, -0.6690, -1.0527],
               [-0.0747, -0.1315, -0.0178, 0.9354]],

              [[-0.8146, -1.0212, -0.4949, -0.5923],
               [-0.6315, -0.7449, -0.4294, -0.9186],
               [-0.6552, -0.6012, -0.6129, -0.5296],
               [0.1580, 0.3066, -0.0132, -1.6795]]],

             [[[-0.0045, 1.6668, 0.1539, -1.0603],
               [-0.2895, 0.8726, 0.2773, 0.4695],
               [-0.2155, 0.2792, -0.5029, -0.0560],
               [-0.1606, 0.2407, -0.3138, 0.0749]],

              [[-0.0045, 1.6668, 0.1539, -1.0603],
               [-0.2733, 0.9177, 0.2703, 0.3825],
               [-0.2363, 0.3177, -0.4039, 0.0707],
               [-0.1465, 0.1215, -0.3514, 0.1096]],

              [[-0.9291, 0.2762, -0.5389, 0.4626],
               [-0.9150, 0.2015, -0.4932, 0.7090],
               [-0.5811, 0.0459, -0.2781, 0.7316],
               [-0.2906, 0.2335, -0.0824, 0.5158]],

              [[-0.9291, 0.2762, -0.5389, 0.4626],
               [-0.8976, 0.1089, -0.4365, 1.0148],
               [-0.1522, -0.1901, 0.0176, 0.8908],
               [0.8761, -0.5560, 0.6310, 0.6123]]]]
        )

        assert torch.allclose(output, expected, atol=1e-4), \
            f"calculate_attention output values mismatch"
