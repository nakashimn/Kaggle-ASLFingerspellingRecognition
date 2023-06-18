import os
import pathlib
import sys
import traceback
from typing import Any

import numpy as np
import timm
import torch
from numpy.typing import NDArray
from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
from augmentations import Augmentation, LabelSmoothing, Mixup


################################################################################
# EfficientNet
################################################################################
class EfficientNetModel(LightningModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        # const
        self.config: dict[str, Any] = config
        self.encoder, self.fc = self._create_model()
        self.criterion: nn.Module = eval(config["loss"]["name"])(
            **self.config["loss"]["params"]
        )

        # augmentation
        self.mixup: Augmentation = Mixup(config["mixup"]["alpha"])
        self.label_smoothing: Augmentation = LabelSmoothing(
            config["label_smoothing"]["eps"], config["num_class"]
        )

        # variables
        self.training_step_outputs: list[dict[str, Any]] = []
        self.validation_step_outputs: list[dict[str, Any]] = []
        self.val_probs: NDArray | float = np.nan
        self.val_labels: NDArray | float = np.nan
        self.min_loss: float = np.nan

    def _create_model(self) -> tuple[nn.Sequential, nn.Sequential]:
        # basemodel
        base_model = timm.create_model(
            self.config["base_model_name"],
            pretrained=True,
            num_classes=0,
            global_pool="",
            in_chans=3,
        )
        layers: list = list(base_model.children())[:-2]
        encoder: nn.Sequential = nn.Sequential(*layers)
        # linear
        fc: nn.Sequential = nn.Sequential(
            nn.Linear(
                encoder[-1].num_features * 7, self.config["fc_mid_dim"], bias=True
            ),
            nn.ReLU(),
            nn.Linear(self.config["fc_mid_dim"], self.config["num_class"], bias=True),
        )
        return encoder, fc

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = input_data
        x = self.encoder(x)
        x = x.mean(dim=2)
        x = x.flatten(start_dim=1)
        out: torch.Tensor = self.fc(x)
        return out

    def training_step(
        self, batch: torch.Tensor | tuple[torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        img, labels = batch
        img, labels = self.mixup(img, labels)
        labels = self.label_smoothing(labels)
        logits = self.forward(img)
        loss: torch.Tensor = self.criterion(logits, labels)
        logit: torch.Tensor = logits.detach()
        label: torch.Tensor = labels.detach()
        outputs: dict[str, torch.Tensor] = {
            "loss": loss,
            "logit": logit,
            "label": label,
        }
        self.training_step_outputs.append(outputs)
        return outputs

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        img, labels = batch
        logits: torch.Tensor = self.forward(img)
        loss: torch.Tensor = self.criterion(logits, labels)
        logit: torch.Tensor = logits.detach()
        prob: torch.Tensor = logits.softmax(dim=1).detach()
        label: torch.Tensor = labels.detach()
        outputs: dict[str, torch.Tensor] = {
            "loss": loss,
            "logit": logit,
            "prob": prob,
            "label": label,
        }
        self.validation_step_outputs.append(outputs)
        return outputs

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        img: torch.Tensor = batch
        logits: torch.Tensor = self.forward(img)
        prob: torch.Tensor = logits.softmax(dim=1).detach()
        return {"prob": prob}

    def on_train_epoch_end(self) -> None:
        logits: torch.Tensor = torch.cat(
            [out["logit"] for out in self.training_step_outputs]
        )
        labels: torch.Tensor = torch.cat(
            [out["label"] for out in self.training_step_outputs]
        )
        metrics: torch.Tensor = self.criterion(logits, labels)
        self.min_loss = np.nanmin(
            [self.min_loss, metrics.detach().cpu().numpy()]
        )
        self.log(f"train_loss", metrics)

        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        logits: torch.Tensor = torch.cat(
            [out["logit"] for out in self.validation_step_outputs]
        )
        probs: torch.Tensor = torch.cat(
            [out["prob"] for out in self.validation_step_outputs]
        )
        labels: torch.Tensor = torch.cat(
            [out["label"] for out in self.validation_step_outputs]
        )
        metrics: torch.Tensor = self.criterion(logits, labels)
        self.log(f"val_loss", metrics)

        self.val_probs = probs.detach().cpu().numpy()
        self.val_labels = labels.detach().cpu().numpy()

        return super().on_validation_epoch_end()

    def configure_optimizers(self) -> tuple[list, list]:
        optimizer = eval(self.config["optimizer"]["name"])(
            self.parameters(), **self.config["optimizer"]["params"]
        )
        scheduler = eval(self.config["scheduler"]["name"])(
            optimizer, **self.config["scheduler"]["params"]
        )
        return [optimizer], [scheduler]
