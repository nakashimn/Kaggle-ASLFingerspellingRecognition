import argparse
import datetime
import glob
import json
import os
import pathlib
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).parents[1].absolute()))
from config.asl_trial_v3 import config


if __name__ == "__main__":
    # path
    filepath_train: str = "/workspace/kaggle/input/asl-fingerspelling/train.csv"
    filepath_supplemental: str = "/workspace/kaggle/input/asl-fingerspelling/supplemental_metadata.csv"
    dirpath_train: str = "/workspace/kaggle/input/asl-fingerspelling/train_landmarks/"
    dirpath_supplemental: str = "/workspace/kaggle/input/asl-fingerspelling/supplemental_landmarks/"
    dirpath_output: str = "/workspace/kaggle/input/asl-fingerspelling/"

    # data
    df_train: pd.DataFrame = pd.read_csv(filepath_train)
    df_supplemental: pd.DataFrame = pd.read_csv(filepath_supplemental)

    df_train_all: pd.DataFrame = pd.concat([df_train, df_supplemental], ignore_index=True)

    # df_train_all.to_csv(f"{dirpath_output}/train.csv", index=False)

    # screaning
    invalid_idx = []
    for idx in tqdm(df_train_all.index):
        path = df_train_all.loc[idx, "path"]
        sequence_id = df_train_all.loc[idx, "sequence_id"]
        file_id = df_train_all.loc[idx, "file_id"]

        # check if parquet exists.
        if not os.path.exists(f"{dirpath_output}/{path}"):
            print(f"{sequence_id} doesn't exist!")
            invalid_idx.append(idx)
            continue

        # check table consistance.
        if not (pathlib.Path(df_train_all.loc[idx, "path"]).stem == str(df_train_all.loc[idx, "file_id"])):
            print(f"missmatch! path: {path} / file_id: {file_id}")
            invalid_idx.append(idx)
            continue

        # check sequence_id reliability.
        df_parquet = pd.read_parquet(f"{dirpath_output}/{path}", columns=[])
        if sequence_id not in df_parquet.index:
            print(f"{sequence_id} is not in {file_id}.parquet")
            invalid_idx.append(idx)

    df_train_all_fixed: pd.DataFrame = df_train_all.drop(
        invalid_idx).reset_index(drop=True)
    df_train_all_fixed.to_csv(f"{dirpath_output}/train.csv", index=False)

    ############################################################################
    # drop short data
    ############################################################################
    filepath_train: str = "/workspace/kaggle/input/asl-fingerspelling-all/train.csv"
    dirpath_output: str = "/workspace/kaggle/input/asl-fingerspelling-all/"
    df_train: pd.DataFrame = pd.read_csv(filepath_train)

    invalid_idx = []
    for idx in tqdm(df_train.index):
        path = df_train.loc[idx, "path"]
        sequence_id = df_train.loc[idx, "sequence_id"]
        file_id = df_train.loc[idx, "file_id"]
        phrase = df_train.loc[idx, "phrase"]


        # check sequence_id reliability.
        df_parquet = pd.read_parquet(f"{dirpath_output}/{path}", columns=[])

        if len(df_parquet.loc[sequence_id]) <= int(len(phrase)):
            print(f"{sequence_id} is too short.")
            invalid_idx.append(idx)

    df_train_fixed: pd.DataFrame = df_train.drop(
        invalid_idx).reset_index(drop=True)
    df_train_fixed.to_csv(f"{dirpath_output}/train_th_1.0.csv", index=False)


    ############################################################################
    # split withna / withoutna
    ############################################################################
    filepath_train: str = "/workspace/kaggle/input/asl-fingerspelling-all/train_th_1.0.csv"
    dirpath_output: str = "/workspace/kaggle/input/asl-fingerspelling-all/"
    df_train: pd.DataFrame = pd.read_csv(filepath_train)
    select_cols: list[str] = config["datamodule"]["dataset"]["select_col"]
    face_cols = [col for col in select_cols if "face" in col]

    noface_idx = []
    pbar = tqdm(df_train.index)
    for idx in pbar:
        path = df_train.loc[idx, "path"]
        sequence_id = df_train.loc[idx, "sequence_id"]
        file_id = df_train.loc[idx, "file_id"]
        phrase = df_train.loc[idx, "phrase"]

        # check sequence_id reliability.
        df_parquet = pd.read_parquet(f"{dirpath_output}/{path}", columns=face_cols)

        # check no_face
        if df_parquet.loc[sequence_id, face_cols].isna().any().any():
            noface_idx.append(idx)
        pbar.set_postfix_str(f"noface: {len(noface_idx)}")


    df_train_withface: pd.DataFrame = df_train.drop(
        noface_idx).reset_index(drop=True)
    df_train_withface.to_csv(f"{dirpath_output}/train_withface_th_1.0.csv", index=False)

    df_train_noface: pd.DataFrame = df_train.loc[noface_idx].reset_index(drop=True)
    df_train_noface.to_csv(f"{dirpath_output}/train_noface_th_1.0.csv", index=False)
