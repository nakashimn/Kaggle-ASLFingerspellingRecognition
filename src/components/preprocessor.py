import glob
import os
import traceback
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


################################################################################
# AbstractClass
################################################################################
class Preprocessor(ABC):
    def train_dataset(self, *args: Any) -> pd.DataFrame:
        raise NotImplementedError()

    def test_dataset(self, *args: Any) -> pd.DataFrame:
        raise NotImplementedError()

    def pred_dataset(self, *args: Any) -> pd.DataFrame:
        raise NotImplementedError()


################################################################################
# Data Preprocessor
################################################################################
class DataPreprocessor:
    def __init__(self, config: dict[str:Any]) -> None:
        self.config: dict[str, Any] = config

    def train_dataset(self) -> pd.DataFrame:
        df_meta: pd.DataFrame = pd.read_csv(self.config["path"]["trainmeta"])
        df_meta = self._cleansing(df_meta)
        df_meta["label_id"] = self._label_to_id(df_meta)
        return df_meta

    def _label_to_id(self, df: pd.DataFrame) -> list[str | int | float]:
        return [self.config["labels"].index(l) for l in df["label"]]

    def test_dataset(self) -> pd.DataFrame:
        df_meta: pd.DataFrame = pd.read_csv(self.config["path"]["testmeta"])
        df_meta = self._cleansing(df_meta)
        return df_meta

    def pred_dataset(self) -> pd.DataFrame:
        filepaths: list[str] = glob.glob(
            f"{self.config['path']['preddata']}/**/*.npz", recursive=True
        )
        df: pd.DataFrame = pd.DataFrame(filepaths, columns=["filepath"])
        return df

    @staticmethod
    def _cleansing(df) -> pd.DataFrame:
        return df
