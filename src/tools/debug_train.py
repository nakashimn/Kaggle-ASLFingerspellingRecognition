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

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from components.datamodule import DataModule, T5ForASLDataset
from components.preprocessor import DataPreprocessor
from components.validations import LevenshteinDistance
from components.models import TransformerforASLModel
from config.asl_trial_v2 import config

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
    Data=T5ForASLDataset,
    config=config["datamodule"],
    transforms=None
)
dataloader = datamodule.train_dataloader()

levenshtein_distance = LevenshteinDistance(config["metrics"]["levenshtein_distance"])

# prepare model
model = TransformerforASLModel(config["model"]).cuda()


model = TransformerforASLModel.load_from_checkpoint(
    "/workspace/kaggle/input/asl-trial-v2/best_loss_0.ckpt",
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

result_ids = preds.detach().argmax(dim=1).cpu().numpy()
levenshtein_distance._convert_ids_to_chars(result_ids)
levenshtein_distance._convert_ids_to_chars(labels[0].cpu().detach().numpy())
levenshtein_distance._convert_ids_to_chars(targets.detach().cpu().numpy())

idx = 1
sample_inputs = input_embeds[[idx]]
sample_mask = attention_mask[[idx]]
sample_labels = labels[[idx]]
levenshtein_distance._convert_ids_to_chars(labels[idx].detach().cpu().numpy())
preds = model(input_embeds=sample_inputs, attention_mask=sample_mask, labels=sample_labels)

levenshtein_distance._convert_ids_to_chars(preds[0].detach().argmax(dim=1).cpu().numpy())
