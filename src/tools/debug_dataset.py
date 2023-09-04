import os
import pathlib
import random
import sys
import traceback

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from components.datamodule import DataModule, ImgDataset, TransformerForASLDataset
from components.preprocessor import DataPreprocessor
from config.sample import config

###
# sample
###
# prepare input
data_preprocessor = DataPreprocessor(config)
df_train = data_preprocessor.train_dataset()
df_test = data_preprocessor.test_dataset()
df_pred = data_preprocessor.pred_dataset()

# DataSet
try:
    dataset = TransformerForASLDataset(
        df_train,
        config["datamodule"]["dataset"]
    )
    for i in tqdm(range(dataset.__len__())):
        batch = dataset.__getitem__(i)
        break
    batch[0].shape
    batch[1].shape
    batch[2].shape

except:
    print(traceback.format_exc())

# DataLoader
try:
    dataloader = DataLoader(
        dataset,
        num_workers=0,
        batch_size=16,
        shuffle=True,
        drop_last=False
    )
    for data in tqdm(dataloader):
        print(data)
        break
    data[0].shape
    data[1].shape
    data[2].shape
except:
    print(traceback.format_exc())

# DataModule
try:
    datamodule = DataModule(
        df_train=df_train,
        df_val=None,
        df_pred=None,
        Dataset=ImgDataset,
        config=config,
        transforms=None
    )
except:
    print(traceback.format_exc())


x = torch.randn(3,4)
x = x[None]


features = pd.read_parquet(
    "/workspace/kaggle/input/asl-fingerspelling/train_landmarks/5414471.parquet",
    columns=config["datamodule"]["dataset"]["select_col"]).loc[1848125865]

features = features.fillna(0.0)
feat = features.values
feat.shape
feat_ = np.pad(feat, [[0, 0], [0, 0]], "constant", constant_values=0.0)
feat__ = cv2.resize(feat_, [184, 128])

plt.imshow(feat_)
plt.imshow(feat__)
