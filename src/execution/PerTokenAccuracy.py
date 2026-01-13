import numpy as np
import torch
from torch.utils.data import DataLoader


class PerTokenAccuracy:
    def __init__(self, data_loader: DataLoader, device: torch.device) -> None:
        """
        Utility to compute per-token accuracy over a dataset.

        Args:
            data_loader: Data loader yielding (x, y) pairs where x: [B, S], y: [B, S].
            device: Device to run computation on.
        """
        self.data_loader = data_loader
        self.device = device

    @torch.no_grad()
    def calc_per_token_acc(self, model: torch.nn.Module) -> np.ndarray:
        """
        Calculate accuracy for each position in the sequence for the dataset provided
        to the constructor.

        Args:
            model: The transformer model.

        Returns:
            Array of shape [seq_len] with accuracy at each position.
        """
        model.eval()

        total_correct = None
        total_counts = None

        for x, y in self.data_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits = model(x)
            preds = logits.argmax(dim=-1)

            correct_mask = (preds == y).float()

            batch_correct = correct_mask.sum(dim=0)
            batch_counts = torch.full_like(batch_correct, x.shape[0])

            if total_correct is None:
                total_correct = batch_correct
                total_counts = batch_counts
            else:
                total_correct += batch_correct
                total_counts += batch_counts

        if total_correct is None:
            return np.array([])

        return (total_correct / total_counts).cpu().numpy()
