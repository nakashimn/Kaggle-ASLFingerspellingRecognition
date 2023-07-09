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
from transformers import T5Config, T5ForConditionalGeneration

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

################################################################################
# T5 for ASL
################################################################################
class T5forASLModel(LightningModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        # const
        self.config: dict[str, Any] = config
        models: tuple[nn.Module] = self._create_model()
        self.linear: nn.Module = models[0]
        self.model: nn.Module = models[1]
        self.input_embeddings: nn.Module = self.model.get_input_embeddings()
        self.output_embeddings: nn.Module = self.model.get_output_embeddings()
        self.bos_embed: torch.Tensor = self.input_embeddings(
            torch.tensor([[self.config["special_token_ids"]["bos_token_id"]]])
        ).detach()
        self.end_token_ids: torch.Tensor = torch.tensor([
            self.config["special_token_ids"]["eos_token_id"],
        ])
        self.example_input_array = torch.rand([1, 10, 84])

        # variables
        self.training_step_outputs: list[dict[str, Any]] = []
        self.validation_step_outputs: list[dict[str, Any]] = []
        self.min_loss: float = np.nan

    def _create_model(self) -> tuple[nn.Module]:
        # linear
        linear = nn.Linear(
            in_features=self.config["dim_input"],
            out_features=self.config["T5_config"]["d_model"],
            bias=True,
        )
        # basemodel
        t5_config = T5Config(**self.config["T5_config"])
        base_model = T5ForConditionalGeneration(t5_config)
        return linear, base_model

    def forward(
            self,
            input_embeds: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = input_embeds.nan_to_num(self.config["fillna_val"])
        x = self.linear(input_embeds)
        if labels is not None:
            loss: torch.Tensor = self._training(x, attention_mask, labels)
            return loss
        else:
            result_ids: torch.Tensor = self._inference(x, attention_mask)
            return result_ids

    def _training(
            self,
            input_embeds: torch.Tensor,
            attention_mask: torch.Tensor,
            labels: torch.Tensor
    ) -> tuple[torch.Tensor]:
        results = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return results.loss, results.logits

    def _inference(
            self,
            input_embeds: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_hidden_states: torch.Tensor = self._encoding_for_inference(
            input_embeds, attention_mask)
        result_ids: torch.Tensor = self._decoding_for_inference(
            encoder_hidden_states, attention_mask)
        return result_ids

    def _encoding_for_inference(
            self,
            input_embeds: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_hidden_states: torch.Tensor = self.model.encoder(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict="pt",
        ).last_hidden_state
        return encoder_hidden_states

    def _decoding_for_inference(
            self,
            encoder_hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size: int = encoder_hidden_states.shape[0]
        bos_embeds: torch.Tensor = self.bos_embed.repeat([batch_size, 1, 1])

        decoder_inputs_embeds: torch.Tensor = bos_embeds
        for _ in range(self.config["max_length"]):
            rslt_embeds: torch.Tensor = self.model.decoder(
                inputs_embeds=decoder_inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                return_dict="pt",
            ).last_hidden_state
            result_ids: torch.Tensor = self.output_embeddings(rslt_embeds).argmax(dim=2)

            # check inference should be finished
            if self._all_have_end_token_id(result_ids):
                break

            # update decoder input
            decoder_inputs_embeds = torch.concat([bos_embeds, rslt_embeds], dim=1)

        return result_ids

    def _all_have_end_token_id(self, result_ids: torch.Tensor) -> bool:
        each_has_end_token_id: torch.Tensor = torch.isin(
            result_ids, self.end_token_ids).any(dim=1)
        return each_has_end_token_id.all()

    def training_step(
            self, batch: torch.Tensor | tuple[torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        input_embeds, attention_mask, labels = batch
        loss, _ = self.forward(input_embeds, attention_mask, labels)
        self.training_step_outputs.append({"loss": loss.detach()})
        self.log(f"train_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss

    def validation_step(
            self, batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        input_embeds, attention_mask, labels = batch
        loss, logits = self.forward(input_embeds, attention_mask, labels)
        self.validation_step_outputs.append({"loss": loss.detach(), "logit": logits.detach()})
        self.log(f"val_loss", loss, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return loss

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        input_embeds: torch.Tensor = batch
        result_ids: torch.Tensor = self.forward(input_embeds)
        return {"outputs": result_ids}

    def on_train_epoch_end(self) -> None:
        loss: float = np.mean(
            [out["loss"].cpu().numpy() for out in self.training_step_outputs]
        )
        self.min_loss = np.nanmin([self.min_loss, loss])
        self.training_step_outputs.clear()
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        loss: float = np.mean(
            [out["loss"].cpu().numpy() for out in self.validation_step_outputs]
        )
        logits: torch.Tensor = torch.cat(
            [out["logit"] for out in self.validation_step_outputs]
        )
        self.validation_step_outputs.clear()
        return super().on_validation_epoch_end()

    def configure_optimizers(self) -> tuple[list, list]:
        optimizer = eval(self.config["optimizer"]["name"])(
            self.parameters(), **self.config["optimizer"]["params"]
        )
        scheduler = eval(self.config["scheduler"]["name"])(
            optimizer, **self.config["scheduler"]["params"]
        )
        return [optimizer], [scheduler]
