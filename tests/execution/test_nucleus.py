import unittest

import torch

from src.execution.TokenChoiceStrategy import TokenChoiceAdvanced


class MyTestCase(unittest.TestCase):
    @torch.no_grad()
    def test_nucleus(self) -> None:
        def test_dist_topp() -> None:
            # Equal probs - should remain equal with top_p=1.0
            logits = torch.tensor([[[1.0, 1.0, 1.0]]])
            sampler = TokenChoiceAdvanced(top_p=1.0, t=1.0)
            res = sampler.get_dist_after_with_temp_and_topp(logits)

            assert torch.abs(res - torch.tensor([1 / 3, 1 / 3, 1 / 3])).sum() <= 1e-4

            # top_p=0.0 should select only the most probable (Rank 0)
            logits = torch.tensor([[[2.0, 3.0, 1.0]]])
            sampler = TokenChoiceAdvanced(top_p=0.0, t=1.0)
            res = sampler.get_dist_after_with_temp_and_topp(logits)

            assert torch.abs(res - torch.tensor([0.0, 1.0, 0.0])).sum() <= 1e-4

            # top_p=0.6: Prob(3.0) ~0.66 > 0.6. Should satisfy threshold alone.
            logits = torch.tensor([[[2.0, 3.0, 1.0]]])
            sampler = TokenChoiceAdvanced(top_p=0.6, t=1.0)
            res = sampler.get_dist_after_with_temp_and_topp(logits)

            assert torch.abs(res - torch.tensor([0.0, 1.0, 0.0])).sum() <= 1e-4

            # top_p=0.71: Prob(3.0)~0.66. Not enough. Need Prob(2.0)~0.24.
            # Keep [3.0, 2.0]. Renormalize.
            logits = torch.tensor([[[1.0, 3.0, 2.0]]])
            sampler = TokenChoiceAdvanced(top_p=0.71, t=1.0)
            res = sampler.get_dist_after_with_temp_and_topp(logits)

            assert torch.abs(res - torch.tensor([0.0, 0.7311, 0.2689])).sum() <= 1e-2

            # top_p=1.0: Keep all
            logits = torch.tensor([[[1.0, 3.0, 2.0]]])
            sampler = TokenChoiceAdvanced(top_p=1.0, t=1.0)
            res = sampler.get_dist_after_with_temp_and_topp(logits)

            assert torch.abs(res - torch.tensor([0.0900, 0.6652, 0.2447])).sum() <= 1e-2

        def test_temperature() -> None:
            # High temperature -> more uniform
            logits = torch.tensor([[[1.0, 1.0, 1.0]]])
            sampler = TokenChoiceAdvanced(top_p=1.0, t=3.0)
            res = sampler.get_dist_after_with_temp_and_topp(logits)

            assert torch.abs(res - torch.tensor([1 / 3, 1 / 3, 1 / 3])).sum() <= 1e-4

            logits = torch.tensor([[[1.0, 3.0, 2.0]]])
            sampler = TokenChoiceAdvanced(top_p=1.0, t=3.0)
            res = sampler.get_dist_after_with_temp_and_topp(logits)

            assert torch.abs(res - torch.tensor([0.2302, 0.4484, 0.3213])).sum() <= 1e-2

            # Low temperature -> more peaked
            logits = torch.tensor([[[1.0, 3.0, 2.0]]])
            sampler = TokenChoiceAdvanced(top_p=1.0, t=1 / 3)
            res = sampler.get_dist_after_with_temp_and_topp(logits)

            assert torch.abs(res - torch.tensor([0.0024, 0.9503, 0.0473])).sum() <= 1e-2

            # Low Temp + Top P -> Extremely peaked (usually 1-hot)
            logits = torch.tensor([[[1.0, 3.0, 2.0]]])
            sampler = TokenChoiceAdvanced(top_p=0.94, t=1 / 3)
            res = sampler.get_dist_after_with_temp_and_topp(logits)

            assert torch.abs(res - torch.tensor([0.0, 1.0, 0.0])).sum() <= 1e-4

        def test_batching() -> None:
            logits = torch.tensor([
                [[1.0, 3.0, 2.0], [1.0, 4.0, 8.0]],
                [[8.0, 4.0, 1.0], [3.0, 1.0, 2.0]]
            ])
            sampler = TokenChoiceAdvanced(top_p=0.7, t=1.0)
            res = sampler.get_dist_after_with_temp_and_topp(logits)

            expected = torch.tensor([
                [[0.0, 0.7311, 0.2689], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0], [0.7311, 0.0, 0.2689]],
            ])
            assert torch.abs(res - expected).sum() <= 1e-2

        test_dist_topp()
        test_temperature()
        test_batching()
        print("All Nucleus Sampling tests passed.")

    test_nucleus()
