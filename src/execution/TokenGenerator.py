import torch
from typing import Callable


class TokenGenerator:
    def __init__(self, model: torch.nn.Module, device: torch.device,
                 token_choice_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.model = model
        self.device = device
        self.token_choice_fn = token_choice_fn

    @staticmethod
    def token_choice_greedy(model_logits: torch.Tensor) -> torch.Tensor:
        """
        Select the most likely token (greedy decoding).

        Args:
            model_logits: Logits of shape [batch, seq_len, vocab_size].

        Returns:
            Selected token indices of shape [batch, 1].
        """
        assert len(model_logits.shape) == 3
        return torch.argmax(model_logits[:, -1:, :], dim=-1)

    @torch.no_grad()
    def generate(
            self,
            model_input: torch.Tensor,
            gen_length: int,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.

        Args:
            model_input: Initial token sequence of shape [batch, seq_len].
            gen_length: Number of tokens to generate.

        Returns:
            Generated tokens of shape [batch, gen_length] (without the input).
        """
        assert len(model_input.shape) == 2
        self.model.eval()

        current_seq = model_input.to(self.device)
        output_tokens = []

        for _ in range(gen_length):
            token_logits = self.model(current_seq)
            token_choice = self.token_choice_fn(token_logits)

            output_tokens.append(token_choice)
            current_seq = torch.cat([current_seq, token_choice], dim=-1)

        return torch.cat(output_tokens, dim=-1)
