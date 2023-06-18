import traceback
from abc import ABC, abstractmethod
from typing import Any, TypeAlias

import librosa
import numpy as np
import torch
from numpy.typing import NDArray
from torch.distributions.distribution import Distribution
from torchaudio import transforms as Ta
from torchvision import transforms as Tv

################################################################################
# TypeAlias
################################################################################
Transforms: TypeAlias = Tv.Compose | torch.nn.Sequential | torch.nn.Module


################################################################################
# AbstractClass
################################################################################
class Augmentation(ABC):
    def __call__(self, *args: Any) -> Any:
        return self.run(*args)

    def run(self, *args: Any) -> Any:
        raise NotImplementedError()


################################################################################
# AudioAugmentation
################################################################################
class SoundAugmentation(Augmentation):
    def __init__(
        self,
        sampling_rate: int = 32000,
        n_fft: int = 2048,
        ratio_harmonic: float = 0.0,
        ratio_pitch_shift: float = 0.0,
        ratio_percussive: float = 0.0,
        ratio_time_stretch: float = 0.0,
        range_harmonic_margin: list[int] = [1, 3],
        range_n_step_pitch_shift: list[float] = [-0.5, 0.5],
        range_percussive_margin: list[int] = [1, 3],
        range_rate_time_stretch: list[float] = [0.9, 1.1],
    ) -> None:
        # const
        self.sampling_rate: int = sampling_rate
        self.n_fft: int = n_fft
        self.enable: dict[str, bool] = {
            "pitch_shift": (ratio_pitch_shift > 0.0),
            "time_stretch": (ratio_time_stretch > 0.0),
            "harmonic": (ratio_harmonic > 0.0),
            "percussive": (ratio_percussive > 0.0),
        }
        self.config: dict[str, dict[str, Any]] = {
            "harmonic": {
                "ratio": ratio_harmonic,
                "range_margin": range_harmonic_margin,
            },
            "pitch_shift": {
                "ratio": ratio_pitch_shift,
                "range_n_step": range_n_step_pitch_shift,
            },
            "percussive": {
                "ratio": ratio_percussive,
                "range_margin": range_percussive_margin,
            },
            "time_stretch": {
                "ratio": ratio_time_stretch,
                "range_rate": range_rate_time_stretch,
            },
        }

    def run(self, snd: NDArray) -> NDArray:
        # check length
        if len(snd) < self.n_fft:
            return snd

        # harmonic
        if self._is_applied(self.config["harmonic"]["ratio"]):
            harmonic_margin: float = np.random.uniform(
                *self.config["harmonic"]["range_margin"]
            )
            snd = librosa.effects.harmonic(snd, margin=harmonic_margin)
        # pitch shift
        if self._is_applied(self.config["pitch_shift"]["ratio"]):
            pitch_shift_n_step: float = np.random.uniform(
                *self.config["pitch_shift"]["range_n_step"]
            )
            snd = librosa.effects.pitch_shift(
                snd, n_steps=pitch_shift_n_step, sr=self.sampling_rate
            )
        # percussive
        if self._is_applied(self.config["percussive"]["ratio"]):
            percussive_mergin: float = np.random.uniform(
                *self.config["percussive"]["range_margin"]
            )
            snd = librosa.effects.percussive(snd, margin=percussive_mergin)
        # time stretch
        if self._is_applied(self.config["time_stretch"]["ratio"]):
            time_stretch_rate: float = np.random.uniform(
                *self.config["time_stretch"]["range_rate"]
            )
            snd = librosa.effects.time_stretch(snd, rate=time_stretch_rate)
        return snd

    @staticmethod
    def _is_applied(ratio: float) -> bool:
        return np.random.choice([True, False], p=[ratio, 1 - ratio])


class Fadein(Augmentation):
    def __init__(self, ratio: float = 0.5) -> None:
        self.ratio: float = ratio

    def run(self, x: NDArray) -> NDArray:
        fade_gain: float = np.random.uniform(1.0 / (x.shape[2] * self.ratio), 1.0)
        weights: torch.Tensor = torch.clip(
            fade_gain * torch.arange(1, x.shape[2] + 1), 0.0, 1.0
        ).to(torch.float32)
        return x * weights


class Fadeout(Augmentation):
    def __init__(self, ratio: float = 0.5) -> None:
        self.ratio: float = ratio

    def run(self, x: NDArray) -> torch.Tensor:
        fade_gain: float = np.random.uniform(1.0 / (x.shape[2] * self.ratio), 1.0)
        weights: torch.Tensor = torch.flip(
            torch.clip(fade_gain * torch.arange(1, x.shape[2] + 1), 0.0, 1.0), dims=[0]
        ).to(torch.float32)
        return x * weights


class Mixup(Augmentation):
    def __init__(self, alpha: float = 0.01, device: str = "cuda") -> None:
        self.alpha: float = alpha
        self.rand_generator: Distribution = torch.distributions.beta.Beta(alpha, alpha)

    def run(
        self, img: torch.Tensor, label: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lam: torch.Tensor = self.rand_generator.sample()
        img_mixup: torch.Tensor = lam * img + (1 - lam) * img.roll(shifts=1, dims=0)
        label_mixup: torch.Tensor = lam * label + (1 - lam) * label.roll(
            shifts=1, dims=0
        )
        return img_mixup, label_mixup


class LabelSmoothing(Augmentation):
    def __init__(
        self, eps: float = 0.01, n_class: int = 3, device: str = "cuda"
    ) -> None:
        eyes: torch.Tensor = torch.eye(n_class)
        self.softlabels: torch.Tensor = (
            torch.where(eyes <= 0, eps / (n_class - 1), 1 - eps)
            .to(torch.float32)
            .to(device)
        )

    def run(self, label: torch.Tensor) -> torch.Tensor:
        return torch.matmul(label.to(torch.float32), self.softlabels)


class SpecAugmentation(Augmentation):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.spec_transform: Transforms | None = self.create_spec_transform()

    def create_spec_transform(self) -> Transforms | None:
        augmentations: list[Transforms] = []
        if ("pitch_shift" in self.config.keys()) and (
            self.config["pitch_shift"] is not None
        ):
            pitchshift: Transforms = Tv.RandomApply(
                [
                    Tv.RandomAffine(
                        degrees=0, translate=(0, self.config["pitch_shift"]["max"])
                    )
                ],
                p=self.config["pitch_shift"]["probability"],
            )
            augmentations.append(pitchshift)
        if ("time_shift" in self.config.keys()) and (
            self.config["time_shift"] is not None
        ):
            timeshift: Transforms = Tv.RandomApply(
                [
                    Tv.RandomAffine(
                        degrees=0, translate=(self.config["time_shift"]["max"], 0)
                    )
                ],
                p=self.config["time_shift"]["probability"],
            )
            augmentations.append(timeshift)
        if ("freq_mask" in self.config.keys()) and (
            self.config["freq_mask"] is not None
        ):
            freqmask: Transforms = Tv.RandomApply(
                [Ta.FrequencyMasking(self.config["freq_mask"]["max"])],
                p=self.config["freq_mask"]["probability"],
            )
            augmentations.append(freqmask)
        if ("time_mask" in self.config.keys()) and (
            self.config["time_mask"] is not None
        ):
            timemask: Transforms = Tv.RandomApply(
                [Ta.TimeMasking(self.config["time_mask"]["max"])],
                p=self.config["time_mask"]["probability"],
            )
            augmentations.append(timemask)
        if ("fadein" in self.config.keys()) and (self.config["fadein"] is not None):
            fadein: Transforms = Tv.RandomApply(
                [Fadein(self.config["fadein"]["max"])],
                p=self.config["fadein"]["probability"],
            )
            augmentations.append(fadein)
        if ("fadeout" in self.config.keys()) and (self.config["fadeout"] is not None):
            fadeout: Transforms = Tv.RandomApply(
                [Fadeout(self.config["fadeout"]["max"])],
                p=self.config["fadeout"]["probability"],
            )
            augmentations.append(fadeout)
        if len(augmentations) == 0:
            return None
        return Tv.Compose(augmentations)

    def run(
        self, img: torch.Tensor, label: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.spec_transform is None:
            if label is None:
                return img
            return img, label
        else:
            if label is None:
                return self.spec_transform(img)
            return self.spec_transform(img), label
