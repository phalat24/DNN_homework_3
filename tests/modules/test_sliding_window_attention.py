import unittest

import torch

from src.modules.RotaryPositionalEmbedding import RotaryPositionalEmbedding
from src.modules.SWAttention import calculate_sliding_attention


class MyTestCase(unittest.TestCase):

    @torch.no_grad()
    def test_calculate_sliding_attention(self) -> None:
        """Test the calculate_sliding_attention function independently of module weights."""
        torch.manual_seed(42)
        batch, seq_len = 2, 4
        num_heads, head_dim = 4, 4
        window_size = 2

        q = torch.randn(batch, num_heads, seq_len, head_dim)
        k = torch.randn(batch, num_heads, seq_len, head_dim)
        v = torch.randn(batch, num_heads, seq_len, head_dim)

        key_weights = torch.randn(num_heads)
        rope = RotaryPositionalEmbedding(head_dim)
        scale = head_dim ** -0.5

        output = calculate_sliding_attention(
            q=rope(q),
            k=rope(k),
            v=v,
            key_weights=key_weights,
            scale=scale,
            device=q.device,
            window_size=window_size
        )

        assert output.shape == (batch, num_heads, seq_len, head_dim), \
            f"Wrong output shape: {output.shape}, expected {(batch, num_heads, seq_len, head_dim)}"

        expected = torch.tensor(
            [[[[-6.8548e-01, 5.6356e-01, -1.5072e+00, -1.6107e+00],
               [-7.5833e-01, 5.5151e-01, -1.3803e+00, -1.3910e+00],
               [-7.5970e-01, 5.4425e-01, -1.3694e+00, -1.4018e+00],
               [-1.0289e+00, -1.5047e-03, 1.3449e-01, 8.8395e-02]],

              [[-1.3793e+00, 6.2580e-01, -2.5850e+00, -2.4000e-02],
               [-1.0804e+00, 2.9937e-01, -1.5638e+00, -4.5193e-03],
               [-3.7572e-01, 9.0874e-01, -9.8827e-01, 3.2158e-01],
               [5.5610e-01, 9.3138e-01, 8.2518e-01, 3.6249e-01]],

              [[9.7329e-01, -1.0151e+00, -5.4192e-01, -4.4102e-01],
               [3.7820e-01, -6.0546e-01, -6.2194e-01, -2.5908e-01],
               [7.5603e-01, -4.5413e-01, -2.9462e-01, -6.9975e-02],
               [5.6501e-01, 6.4487e-02, 4.0517e-01, 4.1787e-01]],

              [[4.0380e-01, -7.1398e-01, 8.3373e-01, -9.5855e-01],
               [4.2490e-01, 1.1594e-01, -4.9589e-01, -1.0976e+00],
               [3.5349e-01, -4.0529e-01, -6.6044e-01, -1.1089e+00],
               [-2.1912e-01, -6.5963e-01, 1.6555e-01, -1.0503e+00]]],

             [[[4.3344e-01, -7.1719e-01, 1.0554e+00, -1.4534e+00],
               [4.4607e-01, -2.8344e-01, 6.3300e-01, -8.4259e-01],
               [4.2691e-01, 3.2082e-01, -4.8548e-01, -5.2133e-01],
               [3.8897e-01, 6.5369e-01, -1.5100e+00, -7.1436e-01]],

              [[8.8538e-01, 1.8244e-01, 7.8638e-01, -5.7920e-02],
               [7.1542e-01, -2.9334e-01, 1.0705e-01, -3.1831e-04],
               [6.7078e-01, 7.1021e-01, 4.8379e-02, 5.4688e-01],
               [7.3988e-01, 2.3339e-01, -3.6269e-01, 3.0450e-01]],

              [[-7.9394e-01, 3.7523e-01, 8.7910e-02, -1.2415e+00],
               [-5.1264e-01, -3.4904e-01, -2.9172e-01, 6.7694e-01],
               [4.9596e-01, 5.8218e-01, -1.2478e-01, 1.5970e-01],
               [6.7310e-02, 1.5020e-01, -2.8622e-01, 1.1465e+00]],

              [[-2.1844e-01, 1.6630e-01, 2.1442e+00, 1.7046e+00],
               [9.8019e-02, 4.3332e-01, 8.2743e-01, 1.1330e+00],
               [2.0074e-02, -5.5385e-02, 2.2436e-01, 9.4053e-01],
               [-1.2419e-01, 1.2867e-01, -6.4606e-01, 2.5874e-01]]]]
        )

        assert torch.allclose(output, expected, atol=1e-4), \
            f"calculate_sliding_attention output values mismatch"
