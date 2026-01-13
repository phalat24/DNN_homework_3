import torch
import torch.nn.functional as F

from src.modules.SwiGLUFeedForward import SwiGLUFeedForward


class Router(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.routing_proj = torch.nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            routing_weights: PROBABILITIES [batch, seq_len, top_k] (Sum to 1).
            expert_indices: Indices [batch, seq_len, top_k].
        """
        assert len(x.shape) == 3

        logits = self.routing_proj(x)

        top_logits, expert_indices = torch.topk(logits, self.top_k, dim=-1)

        top_weights = F.softmax(top_logits, dim=-1)

        return top_weights, expert_indices

    def get_weight_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Debug helper: Returns sparse weight vector [Batch, Seq, Num_Experts].
        """
        batch, seq_len, _ = x.shape
        top_weights, expert_indices = self.forward(x)

        full_weights = torch.zeros(
            (batch, seq_len, self.num_experts),
            device=x.device,
            dtype=x.dtype
        )
        return full_weights.scatter_(dim=-1, index=expert_indices, src=top_weights)


class MixtureOfExperts(torch.nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            inner_dim: int,
            num_experts: int = 8,
            top_k: int = 2
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.experts = torch.nn.ModuleList([
            SwiGLUFeedForward(hidden_dim, inner_dim) for _ in range(num_experts)
        ])

        self.router = Router(hidden_dim, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim].

        Returns:
            Output tensor of shape [batch, seq_len, hidden_dim].
        """
        assert x.ndim == 3

        routing_weights, selected_expert_indices = self.router(x)  # [B,S,K], [B,S,K]

        final_output = torch.zeros_like(x)

        for expert_id in range(self.num_experts):
            weight_mask = torch.eq(selected_expert_indices, expert_id)
            expert_weights = torch.where(
                weight_mask,
                routing_weights,
                torch.zeros_like(routing_weights)
            ).sum(dim=-1)

            if not expert_weights.any():
                continue

            expert_out = self.experts[expert_id](x)  # [B,S,H]

            final_output = final_output + expert_out * expert_weights.unsqueeze(-1)

        return final_output
