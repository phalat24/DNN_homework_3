import unittest

import torch

from src.modules.GroupedQueryAttention import GroupedQueryAttention
from src.modules.MixtureOfExperts import MixtureOfExperts
from src.modules.SWAttention import SWAttention
from src.modules.Transformer import Transformer


class MyTestCase(unittest.TestCase):
    @torch.no_grad()
    def test_transformer(self) -> None:
        torch.manual_seed(42)
        batch, seq_len = 2, 8
        vocab_size, n_layers, hidden_dim = 100, 2, 32
        ff_dim, num_heads, head_dim = 64, 4, 8

        model = Transformer(
            vocab_size=vocab_size,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            ff_dim=ff_dim,
            num_heads=num_heads,
            head_dim=head_dim
        )

        x = torch.randint(0, vocab_size, (batch, seq_len))
        output = model(x)

        # Test output shape
        assert output.shape == (batch, seq_len, vocab_size), f"Wrong shape: {output.shape}"

        # Test that model has correct number of layers
        assert len(model.layers) == n_layers, f"Model should have {n_layers} layers"

        # Test that model has embedding and output projection
        assert hasattr(model, 'embedding'), "Model should have embedding layer"
        assert hasattr(model, 'output_proj'), "Model should have output projection"
        assert isinstance(model.embedding, torch.nn.Embedding), "embedding should be nn.Embedding"

        # Test that model has final normalization
        assert hasattr(model, 'final_norm'), "Model should have final_norm"
        assert isinstance(model.final_norm, torch.nn.RMSNorm), "final_norm should be RMSNorm"

        # Test embedding size
        assert model.embedding.num_embeddings == vocab_size, \
            f"Embedding should have {vocab_size} tokens"
        assert model.embedding.embedding_dim == hidden_dim, \
            f"Embedding dimension should be {hidden_dim}"

        # Test output projection size
        assert model.output_proj.out_features == vocab_size, \
            f"Output projection should project to {vocab_size} dimensions"

        # Test with sliding window alternating
        model_sw = Transformer(
            vocab_size=vocab_size,
            n_layers=4,
            hidden_dim=hidden_dim,
            ff_dim=ff_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            use_sliding_window_alternating=True,
            window_size=4
        )
        output_sw = model_sw(x)
        assert output_sw.shape == (batch, seq_len, vocab_size)

        # Check that alternating layers use sliding window
        assert isinstance(model_sw.layers[1].attention, SWAttention), \
            "Layer 1 should use SWAttention (alternating pattern)"
        assert isinstance(model_sw.layers[3].attention, SWAttention), \
            "Layer 3 should use SWAttention (alternating pattern)"
        assert isinstance(model_sw.layers[0].attention, GroupedQueryAttention), \
            "Layer 0 should use GQA (not sliding window)"

        # Test with MoE
        model_moe = Transformer(
            vocab_size=vocab_size,
            n_layers=2,
            hidden_dim=hidden_dim,
            ff_dim=ff_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            use_moe=True,
            num_experts=4,
            top_k=2
        )
        output_moe = model_moe(x)
        assert output_moe.shape == (batch, seq_len, vocab_size)
        assert isinstance(model_moe.layers[0].ffn, MixtureOfExperts), \
            "All layers should use MoE when use_moe=True"

        # Test with GQA
        model_gqa = Transformer(
            vocab_size=vocab_size,
            n_layers=2,
            hidden_dim=hidden_dim,
            ff_dim=ff_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=2
        )
        output_gqa = model_gqa(x)
        assert output_gqa.shape == (batch, seq_len, vocab_size)

        # Test determinism
        torch.manual_seed(42)
        model2 = Transformer(
            vocab_size=vocab_size,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            ff_dim=ff_dim,
            num_heads=num_heads,
            head_dim=head_dim
        )
        x2 = torch.randint(0, vocab_size, (batch, seq_len))
        output2 = model2(x2)
        assert torch.allclose(output, output2, atol=1e-5), "Transformer should be deterministic"

        # Test that logits are different for different inputs
        x_different = torch.randint(0, vocab_size, (batch, seq_len))
        output_different = model(x_different)
        assert not torch.allclose(output, output_different), \
            "Different inputs should produce different outputs"
