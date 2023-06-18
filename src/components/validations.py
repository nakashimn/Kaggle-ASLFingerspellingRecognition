import traceback
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score


class MinLoss:
    def __init__(self) -> None:
        self.value: float = np.nan

    def update(self, min_loss: float | None) -> None:
        self.value = np.nanmin([self.value, min_loss])


class ValidResult:
    def __init__(self) -> None:
        self.values: NDArray | int | float | None = None

    def append(self, values: NDArray | int | float) -> NDArray | int | float | None:
        if self.values is None:
            self.values = values
            return self.values
        self.values = np.concatenate([self.values, values])
        return self.values


class ConfusionMatrix:
    def __init__(self, probs: NDArray, labels: NDArray, config: dict[str, Any]) -> None:
        # const
        self.config: dict[str, Any] = config
        self.probs: NDArray = probs
        self.labels: NDArray = labels

        # variables
        self.fig: Figure = plt.figure(figsize=[36, 36], tight_layout=True)

    def draw(self) -> Figure:
        idx_probs: int = np.argmax(self.probs, axis=1)
        idx_labels: int = np.argmax(self.labels, axis=1)

        df_confmat: pd.DataFrame = pd.DataFrame(
            confusion_matrix(idx_probs, idx_labels),
            index=self.config["label"],
            columns=self.config["label"],
        )
        axis: Axes = self.fig.add_subplot(1, 1, 1)
        sns.heatmap(df_confmat, ax=axis, cmap="bwr", square=True, annot=True)
        axis.set_xlabel("label")
        axis.set_ylabel("pred")
        return self.fig


class F1Score:
    def __init__(self, probs: NDArray, labels: NDArray, config: dict[str, Any]) -> None:
        # const
        self.config: dict[str, Any] = config
        self.probs: NDArray = probs
        self.labels: NDArray = labels

        # variables
        self.f1_scores: dict[str, float | None] = {"macro": None, "micro": None}

    def calc(self) -> dict[str, float | None]:
        idx_probs: int = np.argmax(self.probs, axis=1)
        idx_labels: int = np.argmax(self.labels, axis=1)
        self.f1_scores = {
            "macro": f1_score(idx_probs, idx_labels, average="macro"),
            "micro": f1_score(idx_probs, idx_labels, average="micro"),
        }
        return self.f1_scores


class LogLoss:
    def __init__(self, probs: NDArray, labels: NDArray, config: dict[str, Any]) -> None:
        # const
        self.probs: NDArray = probs
        self.labels: NDArray = labels
        self.config: dict[str, Any] = config
        self.prob_min: float = 10 ** (-15)
        self.prob_max: float = 1 - 10 ** (-15)

        # variables
        self.logloss: float = np.nan

    def calc(self) -> float:
        norm_probs: NDArray = self.probs / np.sum(self.probs, axis=1)[:, None]
        log_probs: float = np.log(np.clip(norm_probs, self.prob_min, self.prob_max))
        self.logloss = -np.mean(np.sum(self.labels * log_probs, axis=1))
        return self.logloss


class CMeanAveragePrecision:
    def __init__(self, probs: NDArray, labels: NDArray, config: dict[str, Any]) -> None:
        # const
        self.config: dict[str, Any] = config
        self.probs: NDArray = probs
        self.labels: NDArray = labels

        self.padded_probs: NDArray = self._padding(self.probs, config["padding_num"])
        self.padded_labels: NDArray = self._padding(self.labels, config["padding_num"])

    @staticmethod
    def _padding(values: NDArray, padding_num: int) -> NDArray:
        padded_values: NDArray = np.concatenate(
            [values, np.ones([padding_num, values.shape[1]], dtype=values.dtype)]
        )
        return padded_values

    def calc(self) -> float:
        cmap: float = average_precision_score(
            self.padded_labels, self.padded_probs, average="macro"
        )
        return cmap
