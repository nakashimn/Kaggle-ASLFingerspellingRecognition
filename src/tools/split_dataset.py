import pathlib
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).parents[1].absolute()))
from config.asl_trial_v3 import config

if __name__ == "__main__":
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
