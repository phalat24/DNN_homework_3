import torch
import unittest

from src.modules.MixtureOfExperts import Router


class MyTestCase(unittest.TestCase):
    @torch.no_grad()
    def test_router_topk_shapes_and_values(self) -> None:
        """Router should return (top-k probs, top-k indices)."""
        torch.manual_seed(0)

        batch, seq_len, hidden_dim = 2, 3, 16
        num_experts, top_k = 8, 2

        router = Router(hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k)

        x = torch.randn(batch, seq_len, hidden_dim)

        top_probs, idx = router(x)

        # Shapes
        assert top_probs.shape == (batch, seq_len, top_k)
        assert idx.shape == (batch, seq_len, top_k)

        # Reference: compute logits -> topk -> softmax over topk logits
        full_logits = router.routing_proj(x)
        top_logits, expected_idx = torch.topk(full_logits, top_k, dim=-1)
        expected_probs = torch.softmax(top_logits, dim=-1)

        # Indices must match
        assert torch.equal(idx, expected_idx)

        # Probabilities must match (float compare)
        assert torch.allclose(top_probs, expected_probs, atol=1e-6)

        # Index range
        assert int(idx.min()) >= 0 and int(idx.max()) < num_experts

        # Softmax normalization
        sums = top_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(batch, seq_len), atol=1e-6)
