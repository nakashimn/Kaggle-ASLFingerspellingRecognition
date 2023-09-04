import os
import pathlib
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd
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
from validations import LevenshteinDistance


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
        self.linear_in, self.model, self.linear_out = self._create_model()
        self.input_embeddings: nn.Module = self.model.get_input_embeddings()
        self.output_embeddings: nn.Module = self.model.get_output_embeddings()
        self.bos_embed: torch.Tensor = self.input_embeddings(
            torch.tensor([[self.config["special_token_ids"]["bos_token_id"]]])
        ).detach()
        self.end_token_ids: torch.Tensor = torch.tensor([
            self.config["special_token_ids"]["eos_token_id"]
        ]).detach()
        self.levenshtein_distance = LevenshteinDistance(
            self.config["metrics"]["levenshtein_distance"]
        )

        # variables
        self.training_step_outputs: list[dict[str, Any]] = []
        self.validation_step_outputs: list[dict[str, Any]] = []
        self.min_loss: float = np.nan

    def cuda(self):
        self.bos_embed = self.bos_embed.cuda()
        self.end_token_ids = self.end_token_ids.cuda()
        return super().cuda()

    def cpu(self):
        self.bos_embed = self.bos_embed.cpu()
        self.end_token_ids = self.end_token_ids.cpu()
        return super().cpu()

    def to(self, device):
        self.bos_embed = self.bos_embed.to(device)
        self.end_token_ids = self.end_token_ids.to(device)
        return super().to(device)

    def _create_model(self) -> tuple[nn.Module]:
        # linear
        linear_in = nn.Linear(
            in_features=self.config["dim_input"],
            out_features=self.config["T5_config"]["d_model"],
            bias=True,
            device=self.device,
        )
        # basemodel
        t5_config = T5Config(**self.config["T5_config"])
        base_model = T5ForConditionalGeneration(t5_config)
        linear_out = nn.Linear(
            in_features=self.config["T5_config"]["d_model"],
            out_features=self.config["T5_config"]["dim_output"],
            bias=True,
            device=self.device,
        )
        return linear_in, base_model, linear_out

    def forward(
            self,
            input_embeds: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor] | torch.Tensor:
        input_embeds[torch.isnan(input_embeds)] = self.config["fillna_val"]
        x = self.linear_in(input_embeds)

        # training
        if labels is not None:
            outputs = self._training(x, attention_mask, labels)
            return outputs

        # inference
        else:
            output_embeds: torch.Tensor = self._inference(x, attention_mask)
            return output_embeds

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
        output_enbeds: torch.Tensor = self._decoding_for_inference(
            encoder_hidden_states, attention_mask)
        return output_enbeds

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

            # update decoder input
            decoder_inputs_embeds = torch.concat([bos_embeds, rslt_embeds], dim=1)
        output_embeds: torch.Tensor = rslt_embeds.detach()
        return output_embeds

    def training_step(
            self, batch: torch.Tensor | tuple[torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        input_embeds, attention_mask, labels = batch
        print(input_embeds)
        loss, _ = self.forward(input_embeds, attention_mask, labels)
        self.training_step_outputs.append({"loss": loss.detach()})
        self.log(
            f"train_loss",loss, prog_bar=True, logger=True,
            on_epoch=True, on_step=True,
        )
        return loss

    def validation_step(
            self, batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        input_embeds, attention_mask, labels = batch
        loss, logits = self.forward(input_embeds, attention_mask, labels)
        self.validation_step_outputs.append(
            {"loss": loss.detach(), "logit": logits.detach()}
        )
        result_ids: NDArray = logits.detach().argmax(dim=2).cpu().numpy()

        ld_score: float = self.levenshtein_distance.calc(
            result_ids,
            labels.detach().cpu().numpy()
        ).mean()
        self.log(
            f"val_loss", loss, prog_bar=True, logger=True,
            on_epoch=True, on_step=True,
        )
        self.log(
            f"levenshtein_distance", ld_score, prog_bar=True, logger=True,
            on_epoch=True, on_step=True,
        )
        return loss

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        input_embeds: torch.Tensor = batch
        rslt_embeds: torch.Tensor = self.forward(input_embeds)
        return {"outputs": rslt_embeds}

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

class DummyASLModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()

        # const
        self.model = self._create_model()

    def cuda(self):
        return super().cuda()

    def cpu(self):
        return super().cpu()

    def to(self, device):
        return super().to(device)

    def _create_model(self) -> tuple[nn.Module]:
        # linear
        model = nn.Transformer(d_model=184, device=self.device)
        return model

    def forward(
            self,
            input_embeds: torch.Tensor
    ) -> torch.Tensor:
        input_embeds[torch.isnan(input_embeds)] = 0.0
        output_embeds = self.model(input_embeds)
        return output_embeds


################################################################################
# Transformer(Scratch) for ASL
################################################################################
class TransformerforASLModel(LightningModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        # const
        self.config: dict[str, Any] = config
        self.preprocessor: nn.Module = Preprocessor(**config["Preprocessor"])
        self.transformer: nn.Module = Transformer(**config["Transformer"])
        self.criterion: nn.Module = eval(config["loss"]["name"])(
            **self.config["loss"]["params"]
        )
        self.levenshtein_distance: LevenshteinDistance = LevenshteinDistance(
            self.config["metrics"]["levenshtein_distance"]
        )

        # variables
        self.training_step_outputs: list[dict[str, Any]] = []
        self.validation_step_outputs: list[dict[str, Any]] = []
        self.min_loss: float = np.nan

    def forward(
            self,
            input_embeds: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor] | torch.Tensor:
        x: torch.Tensor = self.preprocessor(input_embeds)
        outputs: torch.Tensor = self.transformer(x, labels, attention_mask)
        return outputs

    def training_step(
            self, batch: torch.Tensor | tuple[torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        input_embeds, attention_mask, labels = batch
        logits: torch.Tensor = self.forward(input_embeds, attention_mask, labels)
        loss: torch.Tensor = self.criterion(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
        )
        self.training_step_outputs.append({"loss": loss.detach()})
        self.log(
            f"train_loss", loss, prog_bar=True, logger=True,
            on_epoch=True, on_step=True,
        )
        return loss

    def validation_step(
            self, batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        input_embeds, attention_mask, labels = batch
        logits: torch.Tensor = self.forward(input_embeds, attention_mask, labels)
        loss: torch.Tensor = self.criterion(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1),
        )
        self.validation_step_outputs.append(
            {"loss": loss.detach(), "logit": logits.detach()}
        )
        result_ids: NDArray = logits.detach().argmax(dim=2).cpu().numpy()

        ld_score: float = self.levenshtein_distance.calc(
            result_ids,
            labels.detach().cpu().numpy()
        ).mean()
        self.log(
            f"val_loss", loss, prog_bar=True, logger=True,
            on_epoch=True, on_step=True,
        )
        self.log(
            f"levenshtein_distance", ld_score, prog_bar=True, logger=True,
            on_epoch=True, on_step=True,
        )
        return loss

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        input_embeds: torch.Tensor = batch
        rslt_embeds: torch.Tensor = self.forward(input_embeds)
        return {"outputs": rslt_embeds}

    def on_train_epoch_end(self) -> None:
        loss: float = np.mean(
            [out["loss"].cpu().numpy() for out in self.training_step_outputs]
        )
        self.min_loss = np.nanmin([self.min_loss, loss])
        self.training_step_outputs.clear()
        return super().on_train_epoch_end()

    # def on_validation_epoch_end(self) -> None:
    #     loss: float = np.mean(
    #         [out["loss"].cpu().numpy() for out in self.validation_step_outputs]
    #     )
    #     logits: torch.Tensor = torch.cat(
    #         [out["logit"] for out in self.validation_step_outputs]
    #     )
    #     self.validation_step_outputs.clear()
    #     return super().on_validation_epoch_end()

    def configure_optimizers(self) -> tuple[list, list]:
        optimizer = eval(self.config["optimizer"]["name"])(
            self.parameters(), **self.config["optimizer"]["params"]
        )
        scheduler = eval(self.config["scheduler"]["name"])(
            optimizer, **self.config["scheduler"]["params"]
        )
        return [optimizer], [scheduler]

class Preprocessor(nn.Module):
    def __init__(
            self,
            norm_factor_path: str,
            fillna_val: float = 0.0,
            device: str = "cpu",
    ) -> None:
        super().__init__()
        self.norm_factor_path: str = norm_factor_path
        self.fillna_val: float = fillna_val
        self.device: str = device
        self.normalization: nn.Module = self._create_normalizer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.normalization(x)
        x[torch.isnan(x)] = self.fillna_val
        return x

    def _create_normalizer(self) -> nn.Module:
        means, stds = self._read_norm_factor()
        normalizer: nn.Module = Normalization(means, stds, device=self.device)
        return normalizer

    def _read_norm_factor(self) -> tuple[torch.Tensor]:
        norm_factor: pd.DataFrame = pd.read_csv(
            self.norm_factor_path, index_col=0)
        means: torch.Tensor = torch.tensor(norm_factor.loc["mean"]).to(self.device)
        stds: torch.Tensor = torch.tensor(norm_factor.loc["std"]).to(self.device)
        return means, stds


class Normalization(nn.Module):
    def __init__(
            self,
            means: torch.Tensor,
            stds: torch.Tensor,
            device: str = "cpu",
    ) -> None:
        super().__init__()
        self.means: torch.Tensor = means.float().to(device)
        self.inv_stds: torch.Tensor = (1 / stds).float().to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(x - self.means, self.inv_stds).detach()

################################################################################
# Transformer(Scratch)
#
# Transformer
#  └ TransformerEncoder/TransformerDecoder
#     └ Transformer*Layer
#        ├ AddPositionalEncoding
#        ├ MultiHeadAttention
#        │  └ ScaledDocProductAttention
#        └ FeedForward
################################################################################
class Transformer(nn.Module):
    def __init__(
        self,
        dim_model: int,
        dim_output: int,
        dim_ff: int,
        vocab_size: int,
        bos_idx: int,
        pad_idx: int,
        max_len_encoder: int,
        max_len_decoder: int,
        n_encoder_layer: int,
        n_decoder_layer: int,
        n_head: int,
        dropout_rate: float = 0.0,
        layer_norm_eps: float = 1e-7,
        prenorm: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.dim_model: int = dim_model
        self.dim_output: int = dim_output
        self.dim_ff: int = dim_ff
        self.vocab_size: int = vocab_size
        self.bos_idx: int = bos_idx
        self.pad_idx: int = pad_idx
        self.max_len_encoder: int = max_len_encoder
        self.max_len_decoder: int = max_len_decoder
        self.n_encoder_layer: int = n_encoder_layer
        self.n_decoder_layer: int = n_decoder_layer
        self.n_head: int = n_head
        self.dropout_rate: float = dropout_rate
        self.layer_norm_eps: float = layer_norm_eps
        self.prenorm: bool = prenorm
        self.device: str = device

        self.encoder: nn.Module = TransformerEncoder(
            dim_model, dim_ff, max_len_encoder, n_encoder_layer, n_head,
            dropout_rate, layer_norm_eps, prenorm, device,
        )

        self.embedding: nn.Module = nn.Embedding(
            vocab_size, dim_model, pad_idx, device=device)
        self.decoder: nn.Module = TransformerDecoder(
            dim_model, dim_ff, max_len_decoder, n_decoder_layer, n_head,
            dropout_rate, layer_norm_eps, prenorm, device,
        )

        self.linear: nn.Module = nn.Linear(dim_model, dim_output, device=device)

        # const
        self.bos: torch.Tensor = torch.tensor([[bos_idx]]).detach().to(device)
        self.bos_embeds: torch.Tensor = self.embedding(self.bos).detach().to(device)

    def forward(
        self,
        src: torch.Tensor,
        tgt_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # encoder
        src = self._encoder(src, attention_mask)

        # decoder
        if tgt_ids is None:
            # inference
            return self._inference(src, attention_mask)
        # train
        return self._train(src, tgt_ids, attention_mask)

    def _encoder(
        self,
        src: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        src_pad_mask: torch.Tensor | None = self._extend_mask(
            attention_mask, src.size(1))
        src = self.encoder(src, src_pad_mask)
        return src

    def _inference(
        self,
        src: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        dec_hidden_state_init: torch.Tensor = self.bos_embeds.repeat(
            src.size(0), 1, 1).detach()
        dec_hidden_state: torch.Tensor = dec_hidden_state_init
        for _ in range(self.max_len_decoder):
            src_pad_mask: torch.Tensor | None = self._extend_mask(
                attention_mask, dec_hidden_state.size(1))
            dec_output: torch.Tensor = self.decoder(
                src, dec_hidden_state, src_pad_mask)
            dec_hidden_state: torch.Tensor = torch.cat(
                [dec_hidden_state_init, dec_output], dim=1)
        return self.linear(dec_output)

    def _train(
        self,
        src: torch.Tensor,
        tgt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_pad_mask: torch.Tensor = self._extend_mask(attention_mask, tgt_ids.size(1))
        subsequent_mask: torch.Tensor = self._subsequent_mask(tgt_ids)
        tgt_pad_mask: torch.Tensor = self._pad_mask(tgt_ids)
        mask_self_attn: torch.Tensor = torch.logical_or(subsequent_mask, tgt_pad_mask)

        tgt: torch.Tensor = self.embedding(tgt_ids)
        dec_output: torch.Tensor = self.decoder(src, tgt, src_pad_mask, mask_self_attn)
        return self.linear(dec_output)

    def _extend_mask(
        self,
        mask: torch.Tensor | None,
        seq_len: int,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        mask = mask.unsqueeze(dim=1)
        mask = mask.repeat(1, seq_len, 1)
        return mask.detach().to(self.device)

    def _pad_mask(self, ids: torch.Tensor) -> torch.Tensor:
        mask: torch.Tensor = ids.eq(self.pad_idx)
        mask = self._extend_mask(mask, ids.size(1))
        return mask.detach().to(self.device)

    def _subsequent_mask(self, ids: torch.Tensor) -> torch.Tensor:
        batch_size: int = ids.size(0)
        max_len: int = ids.size(1)
        mask: torch.Tensor = torch.tril(
            torch.ones(batch_size, max_len, max_len)).eq(0)
        return mask.detach().to(self.device)

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim_inout: int,
        dim_ff: int,
        max_len: int,
        n_layer: int,
        n_head: int,
        dropout_rate: float = 0.0,
        layer_norm_eps: float = 1e-7,
        prenorm: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.positional_encoding: nn.Module = AddPositionalEncoding(
            dim_inout, max_len, device)
        self.encoder_layers: nn.Module = nn.ModuleList(
            [TransformerEncoderLayer(
                 dim_inout, dim_ff, n_head, dropout_rate,
                 layer_norm_eps, prenorm, device)
             for _ in range(n_layer)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.positional_encoding(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_inout: int,
        dim_ff: int,
        heads_num: int,
        dropout_rate: float = 0.0,
        layer_norm_eps: float = 1e-7,
        prenorm: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.prenorm: bool = prenorm
        self.multi_head_attention: nn.Module = MultiHeadAttention(
            dim_inout, heads_num, device)
        self.dropout_self_attention: nn.Module = nn.Dropout(dropout_rate)
        self.layer_norm_self_attention: nn.Module = nn.LayerNorm(
            dim_inout, eps=layer_norm_eps, device=device)
        self.ffn: nn.Module = FeedForward(dim_inout, dim_ff, device)
        self.dropout_ffn: nn.Module = nn.Dropout(dropout_rate)
        self.layer_norm_ffn: nn.Module = nn.LayerNorm(
            dim_inout, eps=layer_norm_eps, device=device)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self._multi_head_attention(x, mask)
        x = self._ffn(x)
        return x

    def _multi_head_attention(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tmp: torch.Tensor = x
        if self.prenorm:
            tmp = self.layer_norm_self_attention(tmp)
        tmp = self.multi_head_attention(tmp, tmp, tmp, mask)
        tmp = self.dropout_self_attention(tmp) + x
        if not self.prenorm:
            tmp = self.layer_norm_self_attention(tmp)
        return tmp

    def _ffn(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        tmp: torch.Tensor = x
        if self.prenorm:
            tmp = self.layer_norm_ffn(tmp)
        tmp = self.ffn(tmp)
        tmp = self.dropout_ffn(tmp) + x
        if not self.prenorm:
            tmp = self.layer_norm_ffn(tmp)
        return tmp

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        dim_inout: int,
        dim_ff: int,
        max_len: int,
        n_layer: int,
        n_head: int,
        dropout_rate: float = 0.0,
        layer_norm_eps: float = 1e-7,
        prenorm: bool = False,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.positional_encoding: nn.Module = AddPositionalEncoding(
            dim_inout, max_len, device)
        self.decoder_layers: nn.Module = nn.ModuleList(
            [TransformerDecoderLayer(
                dim_inout, dim_ff, n_head, dropout_rate,
                layer_norm_eps, prenorm, device)
             for _ in range(n_layer)]
        )

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        mask_src_tgt: torch.Tensor | None = None,
        mask_tgt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt = self.positional_encoding(tgt)
        for decoder_layer in self.decoder_layers:
            tgt = decoder_layer(src, tgt, mask_src_tgt, mask_tgt)
        return tgt

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        dim_inout: int,
        dim_ff: int,
        n_head: int,
        dropout_rate: float = 0.0,
        layer_norm_eps: float = 1e-7,
        prenorm: bool = False,
        device: str = "cpu",
    ):
        super().__init__()
        self.prenorm: bool = prenorm
        # self attention block
        self.self_attention: nn.Module = MultiHeadAttention(dim_inout, n_head, device)
        self.dropout_self_attention: nn.Module = nn.Dropout(dropout_rate)
        self.layer_norm_self_attention: nn.Module = nn.LayerNorm(
            dim_inout, eps=layer_norm_eps, device=device)

        # src tgt attention block
        self.src_tgt_attention: nn.Module = MultiHeadAttention(
            dim_inout, n_head, device)
        self.dropout_src_tgt_attention: nn.Module = nn.Dropout(dropout_rate)
        self.layer_norm_src_tgt_attention: nn.Module = nn.LayerNorm(
            dim_inout, eps=layer_norm_eps, device=device)

        # feedforward block
        self.ffn: nn.Module = FeedForward(dim_inout, dim_ff, device)
        self.dropout_ffn: nn.Module = nn.Dropout(dropout_rate)
        self.layer_norm_ffn: nn.Module = nn.LayerNorm(
            dim_inout, eps=layer_norm_eps, device=device)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        mask_src_tgt: torch.Tensor | None = None,
        mask_tgt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt = self._self_attention_block(tgt, mask_tgt)
        x: torch.Tensor = self._src_tgt_attention_block(src, tgt, mask_src_tgt)
        x = self._feedforward_block(x)
        return x

    def _self_attention_block(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        tmp: torch.Tensor = x
        if self.prenorm:
            tmp = self.layer_norm_self_attention(tmp)
        tmp = self.self_attention(tmp, tmp, tmp, mask)
        tmp = self.dropout_self_attention(tmp) + x
        if not self.prenorm:
            tmp = self.layer_norm_self_attention(tmp)
        return tmp

    def _src_tgt_attention_block(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        tmp: torch.Tensor = tgt
        if self.prenorm:
            tmp = self.layer_norm_src_tgt_attention(tmp)
        tmp = self.src_tgt_attention(tmp, src, src, mask)
        tmp = self.dropout_src_tgt_attention(tmp) + tgt
        if not self.prenorm:
            tmp = self.layer_norm_src_tgt_attention(tmp)
        return tmp

    def _feedforward_block(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        tmp: torch.Tensor = x
        if self.prenorm:
            tmp = self.layer_norm_ffn(tmp)
        tmp = self.ffn(tmp)
        tmp = self.dropout_ffn(tmp) + x
        if not self.prenorm:
            tmp = self.layer_norm_ffn(tmp)
        return tmp

class AddPositionalEncoding(nn.Module):
    def __init__(
        self, dim: int, max_len: int, device: str = "cpu"
    ) -> None:
        super().__init__()
        self.dim: int = dim
        self.max_len: int = max_len
        positional_encoding_weight: torch.Tensor = self._initialize_weight().to(device)
        self.register_buffer("positional_encoding_weight", positional_encoding_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len: int = x.size(1)
        return x + self.positional_encoding_weight[:seq_len, :].unsqueeze(0)

    def _get_positional_encoding(self, pos: int, i: int) -> float:
        w: float = pos / (10000 ** (((2 * i) // 2) / self.dim))
        if i % 2 == 0:
            return np.sin(w)
        else:
            return np.cos(w)

    def _initialize_weight(self) -> torch.Tensor:
        positional_encoding_weight: list[list[float]] = [
            [self._get_positional_encoding(pos, i) for i in range(1, self.dim + 1)]
            for pos in range(1, self.max_len + 1)
        ]
        return torch.tensor(positional_encoding_weight).float()

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_inout: int, n_head: int, device: str = "cpu") -> None:
        super().__init__()
        self.dim_inout: int = dim_inout
        self.n_head: int = n_head
        self.dim_k: int = dim_inout // n_head
        self.dim_v: int = dim_inout // n_head

        self.W_k: torch.Tensor = nn.Parameter(
            torch.randn([n_head, dim_inout, self.dim_k]).to(device))
        self.W_q: torch.Tensor = nn.Parameter(
            torch.randn([n_head, dim_inout, self.dim_k]).to(device))
        self.W_v: torch.Tensor = nn.Parameter(
            torch.randn([n_head, dim_inout, self.dim_v]).to(device))

        self.scaled_dot_product_attention: nn.Module = ScaledDotProductAttention(
            self.dim_k)

        self.linear: nn.Module = nn.Linear(
            n_head * self.dim_v, dim_inout, device=device)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qs: list[torch.Tensor] = [
            torch.matmul(q, self.W_q[i]) for i in range(self.n_head)]
        ks: list[torch.Tensor] = [
            torch.matmul(k, self.W_k[i]) for i in range(self.n_head)]
        vs: list[torch.Tensor] = [
            torch.matmul(v, self.W_v[i]) for i in range(self.n_head)]

        attn: torch.Tensor = torch.cat(
            [self.scaled_dot_product_attention(qs[i], ks[i], vs[i], mask)
             for i in range(self.n_head)], dim=2)
        out: torch.Tensor = self.linear(attn)

        return out

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim_key: int) -> None:
        super().__init__()
        self.dim_key: int = dim_key

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scalar: float = np.sqrt(self.dim_key)
        attn: torch.Tensor = torch.matmul(q, torch.transpose(k, 1, 2)) / scalar

        if mask is not None:
            attn = attn.data.masked_fill_(mask, -torch.finfo(torch.float).max)

        attn = nn.functional.softmax(attn, dim=2)
        return torch.matmul(attn, v)

class FeedForward(nn.Module):
    def __init__(self, dim_inout: int, dim_ff: int, device: str = "cpu") -> None:
        super().__init__()
        self.linear1: nn.Module = nn.Linear(dim_inout, dim_ff, device=device)
        self.relu: nn.Module = nn.ReLU()
        self.linear2: nn.Module = nn.Linear(dim_ff, dim_inout, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

################################################################################
# TransformerEncoder for ASL
################################################################################
class TfEncoderforASLModel(LightningModule):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()

        # const
        self.config: dict[str, Any] = config
        self.preprocessor: nn.Module = Preprocessor(**config["Preprocessor"])
        self.encoder: nn.Module = TransformerEncoder(**config["TfEncoder"])
        self.linear: nn.Module = nn.Linear(
            config["TfEncoder"]["dim_inout"], config["dim_output"], bias=True,
        )
        self.criterion: nn.Module = nn.CTCLoss(**self.config["loss"]["params"])
        self.levenshtein_distance: LevenshteinDistance = LevenshteinDistance(
            self.config["metrics"]["levenshtein_distance"]
        )

        # variables
        self.training_step_outputs: list[dict[str, Any]] = []
        self.validation_step_outputs: list[dict[str, Any]] = []
        self.min_loss: float = np.nan

    def forward(
            self,
            input_embeds: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor] | torch.Tensor:
        x: torch.Tensor = self.preprocessor(input_embeds)
        src_pad_mask: torch.Tensor | None = self._extend_mask(
            attention_mask, x.size(1))
        x = self.encoder(x, src_pad_mask)
        outputs: torch.Tensor = self.linear(x)
        return outputs

    def training_step(
            self, batch: torch.Tensor | tuple[torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        input_embeds, attention_mask, labels = batch
        logits: torch.Tensor = self.forward(input_embeds, attention_mask)
        # calc CTC loss
        input_lengths: torch.Tensor = (~attention_mask.detach()).sum(dim=1)
        target_lengths: torch.Tensor = (
            labels != self.config["blank_token_id"]).clone().detach().sum(dim=1)
        loss: torch.Tensor = self.criterion(
            logits.log_softmax(dim=2).transpose(0, 1),
            labels, input_lengths, target_lengths,
        )
        self.training_step_outputs.append({"loss": loss.detach()})
        self.log(
            f"train_loss", loss.detach(),
            prog_bar=True, logger=True, on_epoch=True, on_step=True,
        )
        return loss

    def validation_step(
            self, batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        input_embeds, attention_mask, labels = batch
        logits: torch.Tensor = self.forward(input_embeds, attention_mask)
        input_lengths: torch.Tensor = (~attention_mask.detach()).sum(dim=1)
        target_lengths: torch.Tensor = (
            labels != self.config["blank_token_id"]).clone().detach().sum(dim=1)
        loss: torch.Tensor = self.criterion(
            logits.log_softmax(dim=2).transpose(0, 1),
            labels, input_lengths, target_lengths,
        )
        self.validation_step_outputs.append(
            {"loss": loss.detach(), "logit": logits.detach(), "label": labels.detach()}
        )
        self.log(
            f"val_loss", loss.detach(),
            prog_bar=True, logger=True, on_epoch=True, on_step=False,
        )

        # CTC Postprocess
        result_ids: NDArray = logits.detach().argmax(dim=2).cpu()
        valids: torch.Tensor = (result_ids.diff(dim=1) != 0)
        masks: torch.Tensor = torch.concat(
            [torch.ones([logits.size(0), 1], dtype=bool), valids], dim=1)
        result_ids = torch.where(masks, result_ids, self.config["blank_token_id"])
        # calc LevenshteinDistance
        ld_score: float = self.levenshtein_distance.calc(
            result_ids.numpy(), labels.detach().cpu().numpy()).mean()
        self.log(
            f"levenshtein_distance", ld_score,
            prog_bar=True, logger=True, on_epoch=True, on_step=False,
        )

        return loss

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        input_embeds: torch.Tensor = batch
        rslt_embeds: torch.Tensor = self.forward(input_embeds)
        return {"outputs": rslt_embeds}

    def _extend_mask(
        self,
        mask: torch.Tensor | None,
        seq_len: int,
    ) -> torch.Tensor | None:
        if mask is None:
            return None
        mask = mask.unsqueeze(dim=1)
        mask = mask.repeat(1, seq_len, 1)
        return mask.detach().to(self.device)

    def on_train_epoch_end(self) -> None:
        loss: float = np.mean(
            [out["loss"].cpu().numpy() for out in self.training_step_outputs]
        )
        self.min_loss = np.nanmin([self.min_loss, loss])
        self.training_step_outputs.clear()
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        if len(self.validation_step_outputs) == 0:
            return super().on_validation_epoch_end()
        logit = self.validation_step_outputs[0]["logit"][0]
        label = self.validation_step_outputs[0]["label"][0]
        result_ids: NDArray = logit.detach().argmax(dim=1).cpu()
        pred_sample: str = self.levenshtein_distance._convert_ids_to_chars(
            result_ids.numpy())
        label_sample: str = self.levenshtein_distance._convert_ids_to_chars(
            label.cpu().numpy())
        self.validation_step_outputs.clear()
        print(f"\npred: {pred_sample[:10]} / label: {label_sample[:10]}\n")
        return super().on_validation_epoch_end()

    def configure_optimizers(self) -> tuple[list, list]:
        optimizer = eval(self.config["optimizer"]["name"])(
            self.parameters(), **self.config["optimizer"]["params"]
        )
        scheduler = eval(self.config["scheduler"]["name"])(
            optimizer, **self.config["scheduler"]["params"]
        )
        return [optimizer], [scheduler]
