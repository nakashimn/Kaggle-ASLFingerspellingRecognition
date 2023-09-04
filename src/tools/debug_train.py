import pathlib
import sys
import traceback

import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import onnx
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from components.datamodule import DataModule, TfEncoderForASLDataset
from components.preprocessor import DataPreprocessor
from components.validations import LevenshteinDistance
from components.models import TfEncoderforASLModel
from config.asl_trial_v3 import config

###
# sample
###
# prepare dataloader
data_preprocessor = DataPreprocessor(config)
df_train = data_preprocessor.train_dataset()
datamodule = DataModule(
    df_train=df_train,
    df_val=None,
    df_pred=None,
    Data=TfEncoderForASLDataset,
    config=config["datamodule"],
    transforms=None
)
dataloader = datamodule.train_dataloader()

levenshtein_distance = LevenshteinDistance(config["metrics"]["levenshtein_distance"])

# prepare model
model = TransformerforASLModel(config["model"]).cuda()


model = TransformerforASLModel.load_from_checkpoint(
    "/workspace/tmp/best_loss_0.ckpt",
    config=config["model"],
)

criteria = nn.CrossEntropyLoss(ignore_index=62)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for batch in dataloader:
    break

input_embeds = batch[0].cuda()
attention_mask = batch[1].cuda()
labels = batch[2].cuda()

logits = model(
    input_embeds=input_embeds,
    attention_mask=attention_mask,
    labels=labels,
)

preds = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
targets = labels[:, 1:].contiguous().view(-1)
loss = criteria(preds, targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()

########################################################################################
# predict sample
########################################################################################
idx = 1
sample_inputs = input_embeds[[idx]]
sample_mask = attention_mask[[idx]]
sample_labels = labels[[idx]]

# target
target = levenshtein_distance._convert_ids_to_chars(labels[idx].detach().cpu().numpy())

# predict
logits = model(input_embeds=sample_inputs, attention_mask=sample_mask, labels=sample_labels)
predict = levenshtein_distance._convert_ids_to_chars(
    levenshtein_distance._trim_ids(logits[0].detach().argmax(dim=1).cpu().numpy()))

print(f"target: {target}")
print(f"predict: {predict}")


########################################################################################
# TfEncoderForASL
########################################################################################
model = TfEncoderforASLModel(config=config["model"]).cuda()
model = TfEncoderforASLModel.load_from_checkpoint(
    "/workspace/kaggle/input/asl-trial-v3/best_loss_0.ckpt",
    config=config["model"],
)
ctc_loss = nn.CTCLoss(blank=59, zero_infinity=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

for i, data in enumerate(dataloader):
    input_embeds = data[0].cuda()
    attention_mask = data[1].cuda()
    labels = data[2].cuda()
    # input_lengths: torch.Tensor = (~attention_mask.detach()).sum(dim=1)
    input_lengths: torch.Tensor = torch.ones(input_embeds.size(0), dtype=int) * input_embeds.size(1)
    target_lengths: torch.Tensor = (labels.detach() != 59).clone().detach().sum(dim=1)
    break

pbar = tqdm(range(10000))
for _ in pbar:
    logits = model(input_embeds=input_embeds, attention_mask=attention_mask)
    loss: torch.Tensor = ctc_loss(
        logits.log_softmax(dim=2).transpose(0, 1),
        labels.detach().cpu(), input_lengths.cpu(), target_lengths.cpu(),
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pbar.set_description_str(f"loss: {loss.detach().cpu():.03f}")
    # check
    result_ids = logits.detach().argmax(dim=2).cpu()
    pred = levenshtein_distance._convert_ids_to_chars(result_ids[0].detach().cpu().numpy())
    label = levenshtein_distance._convert_ids_to_chars(labels[0].detach().cpu().numpy())
    pbar.set_postfix_str(f"{pred[:10]} / {label}")
    if (loss.isinf() or loss.isnan()):
        break
    if i == 20:
        break



result_ids = logits.detach().argmax(dim=2).cpu()
valids: torch.Tensor = (result_ids.diff(dim=1) != 0)
masks: torch.Tensor = torch.concat(
    [torch.ones([logits.size(0), 1], dtype=bool), valids], dim=1)
result_ids = torch.where(masks, result_ids, 59).numpy()
# calc LevenshteinDistance
ld_score: float = levenshtein_distance.calc(
    result_ids,
    labels.detach().cpu().numpy()
).mean()

result_ids = logits.detach().argmax(dim=2)
valids = (result_ids.diff(dim=1) != 0)
masks = torch.concat([torch.ones([logits.size(0), 1], dtype=bool).cuda(), valids], dim=1)

result_ids = torch.where(masks, result_ids, 59)
preds = torch.nn.functional.one_hot(result_ids, num_classes=60)
