import torch
from torch import nn

################################################################################
# FocalLoss
################################################################################


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma: float = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probas: torch.Tensor = pred.softmax(dim=1)
        loss: torch.Tensor = -(
            target * ((1 - probas) ** self.gamma) * (probas.log())
        ).mean()
        return loss
