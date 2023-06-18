import traceback
from typing import Any, TypeAlias

import albumentations as A
import cv2
import librosa
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pytorch_lightning import LightningDataModule
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as Tv

################################################################################
# TypeAlias
################################################################################
Transforms: TypeAlias = Tv.Compose | torch.nn.Sequential | torch.nn.Module


################################################################################
# For EfficientNetBaseModel(Image)
################################################################################
class ImgDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
        transform: Transforms | None = None,
    ) -> None:
        self.config: dict[str, Any] = config
        self.filepaths: NDArray = self._read_filepaths(df)
        self.labels: torch.Tensor | None = None
        if self.config["label"] in df.keys():
            self.labels = self._read_labels(df[self.config["label"]])
        self.pre_transform: Transforms = A.Compose(
            [A.Normalize(config["mean"], config["std"])]
        )
        self.to_tensor: Transforms = Tv.ToTensor()
        self.transform: Transforms | None = transform

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor]:
        img = self._read_img(self.filepaths[idx])
        img = self.pre_transform(image=img)["image"]
        img = self.to_tensor(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.labels is not None:
            labels: torch.Tensor = self.labels[idx]
            return img, labels
        return img

    @staticmethod
    def _read_filepaths(df: pd.DataFrame) -> NDArray:
        values: NDArray = df["filepath"].values
        return values

    @staticmethod
    def _read_img(filepath: str) -> NDArray:
        img: NDArray = cv2.imread(filepath)
        return img

    def _read_labels(self, df: pd.DataFrame) -> torch.Tensor:
        labels: torch.Tensor = torch.tensor(
            df.apply(self._to_onehot), dtype=torch.float32
        )
        return labels

    def _to_onehot(self, series: pd.Series) -> list[int]:
        return [1 if l in series else 0 for l in self.config["labels"]]


################################################################################
# For EfficientNetBaseModel(Audio)
################################################################################
class AudioDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config: dict[str, Any],
        transform: Transforms | None = None,
    ) -> None:
        self.config: dict[str, Any] = config
        self.filepaths: NDArray = self._read_filepaths(df)
        self.labels: torch.Tensor | None = None
        if self.config["label"] in df.keys():
            self.labels = self._read_labels(df[self.config["label"]])
        self.pre_transform: Transforms = A.Compose(
            [A.Normalize(config["mean"], config["std"])]
        )
        self.to_tensor: torch.Tensor = Tv.ToTensor()
        self.transform: Transforms = transform

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor]:
        melspec: NDArray | torch.Tensor = self._read_melspec(self.filepaths[idx])
        melspec = self._normalize(melspec)
        melspec = self.pre_transform(image=melspec)["image"]
        melspec = self.to_tensor(melspec)
        if self.transform is not None:
            melspec = self.transform(melspec)
        if self.labels is not None:
            labels: torch.Tensor = self.labels[idx]
            return melspec, labels
        return melspec

    @staticmethod
    def _read_filepaths(df: pd.DataFrame) -> NDArray:
        values: NDArray = df["filepath"].values
        return values

    @staticmethod
    def _read_melspec(filepath: str) -> NDArray:
        melspec: NDArray = np.load(filepath)["arr_0"]
        melspec = np.expand_dims(melspec, axis=-1)
        return melspec

    @staticmethod
    def _normalize(melspec: NDArray, eps: float = 1e-6) -> NDArray:
        melspec: NDArray = (melspec - melspec.mean()) / (melspec.std() + eps)
        if (melspec.max() - melspec.min()) < eps:
            return np.zeros_like(melspec, dtype=np.uint8)
        melspec = (
            255 * ((melspec - melspec.min()) / (melspec.max() - melspec.min()))
        ).astype(np.uint8)
        return melspec

    def _read_labels(self, df: pd.DataFrame) -> torch.Tensor:
        labels: torch.Tensor = torch.tensor(
            df.apply(self._to_onehot), dtype=torch.float32
        )
        return labels

    def _to_onehot(self, series: pd.Series) -> list[int]:
        return [1 if l in series else 0 for l in self.config["labels"]]


################################################################################
# DataModule
################################################################################
class DataModule(LightningDataModule):
    def __init__(
        self,
        df_train: pd.DataFrame | None,
        df_val: pd.DataFrame | None,
        df_pred: pd.DataFrame | None,
        Data: Dataset,
        config: dict[str, Any],
        transforms: Transforms | None,
    ) -> None:
        super().__init__()

        # const
        self.config: dict[str, Any] = config
        self.df_train: pd.DataFrame | None = df_train
        self.df_val: pd.DataFrame | None = df_val
        self.df_pred: pd.DataFrame | None = df_pred
        self.transforms: Transforms | None = self._read_transforms(transforms)

        # class
        self.Data = Data

    @staticmethod
    def _read_transforms(
        transforms: Transforms | None
    ) -> dict[str, Transforms | None]:
        if transforms is not None:
            return transforms
        return {"train": None, "valid": None, "pred": None}

    def train_dataloader(self) -> DataLoader | None:
        if (self.df_train is None) or (len(self.df_train) == 0):
            return None
        dataset: Dataset = self.Data(
            self.df_train, self.config["dataset"], self.transforms["train"]
        )
        dataloader: DataLoader = DataLoader(
            dataset,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            **self.config["dataloader"]
        )
        return dataloader

    def val_dataloader(self) -> DataLoader | None:
        if (self.df_val is None) or (len(self.df_val) == 0):
            return None
        dataset: Dataset = self.Data(
            self.df_val, self.config["dataset"], self.transforms["valid"]
        )
        dataloader: DataLoader = DataLoader(
            dataset,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            **self.config["dataloader"]
        )
        return dataloader

    def predict_dataloader(self) -> DataLoader | None:
        if (self.df_pred is None) or (len(self.df_pred) == 0):
            return None
        dataset: Dataset = self.Data(
            self.df_pred, self.config["dataset"], self.transforms["pred"]
        )
        dataloader: DataLoader = DataLoader(
            dataset,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            **self.config["dataloader"]
        )
        return dataloader
