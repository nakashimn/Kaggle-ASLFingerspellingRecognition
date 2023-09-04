import pathlib
import sys
import traceback

import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import torch
import onnx
import pandas as pd
from torch import nn

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from components.models import TransformerforASLModel, TfEncoderforASLModel, MultiHeadAttention, ScaledDotProductAttention
from config.asl_trial_v3 import config

########################################################################################
# TransformerForASLModel
########################################################################################
# prepare input
input_embeds = torch.rand([1, 10, 184]).cuda() # [Batch x length x dim]
attention_mask = torch.randint(0, 2, [1, 10], dtype=bool).cuda()
labels = torch.randint(0, 60, [1, 10]).cuda()

# training
try:
    model = TransformerforASLModel(config["model"]).cuda()
    logits = model(input_embeds=input_embeds, attention_mask=attention_mask, labels=labels)
    result_ids = logits.detach().argmax(dim=2)

except:
    print(traceback.format_exc())

# inference
try:
    model = TransformerforASLModel(config["model"]).cuda()
    logits = model(input_embeds=input_embeds)
except:
    print(traceback.format_exc())

input_embeds.std(dim=2)

mulhead_attn = MultiHeadAttention(dim_inout=184, n_head=1).cuda()
for param in mulhead_attn.state_dict():
    print(param)

sdp_attn = ScaledDotProductAttention(3)
sdp_attn(input_embeds, input_embeds, input_embeds)

q = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
k = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
v = torch.tensor([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]])
sdp_attn(q, k, v)

scalar: float = np.sqrt(3)
attn: torch.Tensor = torch.matmul(q, torch.transpose(k, 1, 2)) / scalar

mask = torch.tensor([[
    [False,  True,  True,  True],
    [False, False,  True,  True],
    [False, False, False,  True],
]])
attn = attn.data.masked_fill_(mask, -torch.finfo(torch.float).max)
attn = nn.functional.softmax(attn, dim=2)
torch.matmul(attn, v)

output_embeds = mulhead_attn(input_embeds, input_embeds, input_embeds)

plt.imshow(mulhead_attn.W_k.grad[0].detach().cpu().numpy())

########################################################################################
# TfEncoderForASLModel
########################################################################################
# prepare input
input_embeds = torch.rand([1, 11, 184]).cuda() # [Batch x length x dim]
attention_mask = torch.randint(0, 2, [1, 11], dtype=bool).cuda()
labels = torch.randint(0, 60, [1, 10]).cuda()

# training
try:
    model = TfEncoderforASLModel(config["model"]).cuda()
    logits = model(input_embeds=input_embeds, attention_mask=attention_mask)
    result_ids = logits.detach().argmax(dim=2)

    valids = (result_ids.diff(dim=1) != 0)

    masks = torch.concat([torch.ones([logits.size(0), 1], dtype=bool), valids], dim=1)

    result_ids = torch.where(masks, result_ids, 59)
    preds = torch.nn.functional.one_hot(result_ids, num_classes=60)

except:
    print(traceback.format_exc())

# inference
try:
    model = TransformerforASLModel(config["model"]).cuda()
    logits = model(input_embeds=input_embeds)
except:
    print(traceback.format_exc())
