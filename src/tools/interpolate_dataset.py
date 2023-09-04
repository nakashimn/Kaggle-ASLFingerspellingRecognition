import glob
import pathlib
import sys
import traceback

import pandas as pd
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).parents[1].absolute()))
from config.asl_trial_v3 import config

if __name__ == "__main__":
    dirpath_org: str = "/workspace/kaggle/input/asl-fingerspelling-all/"
    filepaths_parquet: list[str] = glob.glob(f"{dirpath_org}/**/*.parquet", recursive=True)
    dirpath_output: str = "/workspace/kaggle/input/asl-fingerspelling-all-interp/"
    select_cols: list[str] = config["datamodule"]["dataset"]["select_col"]

    pbar = tqdm(filepaths_parquet)
    for filepath_parquet in pbar:
        df_parquet = pd.read_parquet(filepaths_parquet, columns=select_cols)
        seq_ids = df_parquet.index.unique()
        for seq_id in tqdm(seq_ids, leave=True):
            df_parquet.loc[seq_id] = df_parquet.loc[seq_id].interpolate(limit_direction="both")
        filepath_output: str = f"{dirpath_output}/" + "/".join(filepath_parquet.split("/")[-2:])
        df_parquet.to_parquet(filepath_output)
