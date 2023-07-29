import argparse
import datetime
import glob
import json
import os
import pathlib
import sys
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Config
from transformers import DistilBertModel, DistilBertConfig
from transformers import PreTrainedTokenizer
from tokenizers import Tokenizer


sys.path.append(str(pathlib.Path(__file__).parents[1].absolute()))
from components.utility import print_info
# from config.sample import config
from config.asl_trial_v2 import config

def draw_hist(
        axis,
        vals,
        bins=None,
        xlim=None,
        ylim=None,
        xlabel=None,
        ylabel=None,
        color=None,
        alpha=None,
        ec="gray"
):
    axis.hist(vals, bins=bins, color=color, alpha=alpha, ec=ec)
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_axisbelow(True)
    axis.grid(axis="y")
    return axis

# tokenize
def phrase_to_ids(
        phrase: str,
        bos_id: int,
        eos_id: int,
        unk_id: int,
        pad_id: int,
        length: int | None = None
) -> list[list[int]]:
    pad_length = length - len(phrase) if length is not None else 0
    ids = [bos_id] + [send_character_map.get(s, unk_id) for s in phrase] \
        + [eos_id] + [pad_id] * pad_length
    return ids

if __name__ == "__main__":

    # path
    dirpath_parquet = config["path"]["traindata"]
    filepaths_parquet = glob.glob(f"{dirpath_parquet}/*.parquet")
    filepath_train = config["path"]["trainmeta"]
    filepath_char_ids = config["path"]["vocab_file"]

    # const
    select_cols = config["datamodule"]["dataset"]["select_col"]

    # read
    df_train = pd.read_csv(filepath_train)
    # df_train.loc[:499].to_csv("/workspace/data/sample_train.csv", index=False)
    file_ids: NDArray = df_train["file_id"].unique()

    # standardization factor
    columns: NDArray = pd.read_parquet(
        f"{dirpath_parquet}/{file_ids[0]}.parquet").columns.to_numpy()[1:]
    norm_factor: dict[str, float] = {
        "mean": [],
        "std": [],
    }
    for col in tqdm(columns):
        df_feats: pd.DataFrame = pd.concat(
            [pd.read_parquet(f"{dirpath_parquet}/{file_id}.parquet",
                            columns=[col])
            for file_id in df_train["file_id"].unique()]
        )
        norm_factor["mean"].append(df_feats.mean()[0])
        norm_factor["std"].append(df_feats.std()[0])
    df_norm_factor: pd.DataFrame = pd.DataFrame(norm_factor, index=columns).T
    df_norm_factor.to_csv("norm_factor.csv")


    # charmap to tokenize
    with open (filepath_char_ids, "r") as f:
        character_map = json.load(f)
    special_tokens = {59: "<s>", 60: "</s>", 61: "<unk>", 62: "<pad>"}
    send_character_map = {i:j for i,j in character_map.items()}
    rev_character_map = {j:i for i,j in character_map.items()}

    data = [pd.read_parquet(f)[["frame"]] for f in filepaths_parquet]
    df_data = pd.concat(data)

    # token length
    indices = df_data.index.unique().to_arrow().to_pylist()
    df_sequence_id: pd.DataFrame = pd.DataFrame(
        df_data.index, columns=["sequence_id"]).reset_index(drop=True)
    feature_length = df_sequence_id["sequence_id"].value_counts()
    plt.hist(feature_length.to_numpy(), bins=100)

    print_info(
        {
            "stats of feature length": {
                "mean": feature_length.mean(),
                "median": feature_length.median(),
                "max": feature_length.max(),
                "99%": feature_length.quantile(0.99),
                "95%": feature_length.quantile(0.95),
                "90%": feature_length.quantile(0.90),
                "10%": feature_length.quantile(0.10),
                "5%": feature_length.quantile(0.05),
                "1%": feature_length.quantile(0.01),
                "min": feature_length.min(),
            }
        }
    )

    phrase_length = df_train["phrase"].str.len()

    print_info(
        {
            "stats of phrase length": {
                "mean": phrase_length.mean(),
                "median": phrase_length.median(),
                "max": phrase_length.max(),
                "99%": phrase_length.quantile(0.99),
                "95%": phrase_length.quantile(0.95),
                "90%": phrase_length.quantile(0.90),
                "10%": phrase_length.quantile(0.10),
                "5%": phrase_length.quantile(0.05),
                "1%": phrase_length.quantile(0.01),
                "min": phrase_length.min(),
            }
        }
    )

    df_parquet = pd.read_parquet(filepaths_parquet[0])

    faces = [col for col in df_parquet.columns if "face" in col]
    poses = [col for col in df_parquet.columns if "pose" in col]
    left_hands = [col for col in df_parquet.columns if "left_hand" in col]
    right_hands = [col for col in df_parquet.columns if "right_hand" in col]

    print_info(
        {
            "landmarks": {
                "ALL": len(df_parquet.columns)-1,
                "face": len(faces),
                "pose": len(poses),
                "hand(left)": len(left_hands),
                "hand(right)": len(right_hands),
            }
        }
    )
