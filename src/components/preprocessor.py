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
        return df_meta

    def test_dataset(self) -> pd.DataFrame:
        df_meta: pd.DataFrame = pd.read_csv(self.config["path"]["testmeta"])
        df_meta = self._cleansing(df_meta)
        return df_meta

    def pred_dataset(self) -> pd.DataFrame:
        df_meta: pd.DataFrame = pd.read_csv(self.config["path"]["predmeta"])
        df_meta = self._cleansing(df_meta)
        return df_meta

    @staticmethod
    def _cleansing(df) -> pd.DataFrame:
        return df
