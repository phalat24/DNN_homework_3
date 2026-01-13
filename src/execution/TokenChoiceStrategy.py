import torch
import torch.nn.functional as F


class TokenChoiceAdvanced:
    def __init__(self, top_p: float, t: float) -> None:
        self.top_p = top_p
        self.t = t

    @torch.no_grad()
    def get_dist_after_with_temp_and_topp(
            self, model_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply temperature scaling and top-p (nucleus) sampling.
        """
        assert len(model_logits.shape) == 3

        scaled_logits = model_logits / self.t
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)

        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > self.top_p

        # shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()

        # always keep the first token
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )

        scaled_logits[indices_to_remove] = float('-inf')

        return F.softmax(scaled_logits, dim=-1)

    @torch.no_grad()
    def __call__(
            self, model_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Select next token using temperature and top-p sampling.
        """
        probs = self.get_dist_after_with_temp_and_topp(
            model_logits=model_logits[:, -1:, :]
        )

        probs = probs.squeeze(1)

        dist = torch.distributions.Categorical(probs=probs)

        return dist.sample().unsqueeze(1)
