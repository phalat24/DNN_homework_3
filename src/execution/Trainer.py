import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from src.modules.Transformer import Transformer


class Trainer:
    def __init__(
            self,
            vocab_size: int,
            device: torch.device,
            train_loader: DataLoader,
            test_loader: DataLoader,
            model: Transformer,
            loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
    ) -> None:
        self.vocab_size = vocab_size
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.model = model

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.loss_fn = loss_fn

    @torch.no_grad()
    def eval_acc(self, dataloader: DataLoader) -> torch.Tensor:
        self.model.eval()
        sum_acc = 0
        num_examples = 0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            model_out = self.model(x)

            acc = (torch.argmax(model_out, dim=-1) == y).to(torch.float32).sum()
            sum_acc += acc
            num_examples += model_out.shape[0] * model_out.shape[1]

        return sum_acc / num_examples

    def train(self, num_epochs: int) -> None:
        for epoch in range(num_epochs):
            if epoch == 0:
                acc = self.eval_acc(self.test_loader)
                print(f"{epoch}: Avg eval accuracy {acc}")

            self.model.train()
            for i, (x, y) in tqdm(enumerate(self.train_loader)):
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                model_out = self.model(x)

                loss = self.calculate_loss(
                    model_out=model_out,
                    targets=y,
                    loss_fn=self.loss_fn
                )

                loss.backward()
                self.optimizer.step()

            acc = self.eval_acc(self.test_loader)
            print(f"{epoch}: Avg eval accuracy {acc}")

    @staticmethod
    def calculate_loss(
            model_out: torch.Tensor,
            targets: torch.Tensor,
            loss_fn: torch.nn.Module,
    ) -> torch.Tensor:
        """
        Args:
            model_out: Output tensor of shape [batch, seq_len, vocab_size].
            targets: Target tensor of shape [batch, seq_len].
            loss_fn: Loss function to use.

        Returns:
            Cross-entropy loss.
        """
        batch, seq_len, vocab_size = model_out.shape
        model_out_flat = model_out.view(batch * seq_len, vocab_size)
        targets_flat = targets.view(batch * seq_len)

        loss = loss_fn(model_out_flat, targets_flat)

        return loss
