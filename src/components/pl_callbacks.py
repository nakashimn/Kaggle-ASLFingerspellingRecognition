import os
import subprocess
import traceback
from typing import Any

from pytorch_lightning import LightningModule, Trainer, callbacks


class ModelUploader(callbacks.Callback):
    def __init__(
        self, model_dir: str, every_n_epochs: int = 5, message: str = ""
    ) -> None:
        self.model_dir: str = model_dir
        self.every_n_epochs: int = every_n_epochs
        self.message: str = message
        self.should_upload: bool = False
        super().__init__()

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]
    ) -> None:
        if self.should_upload:
            self._upload_model(f"{self.message}[epoch:{trainer.current_epoch}]")
            self.should_upload = False
        return super().on_save_checkpoint(trainer, pl_module, checkpoint)

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if self.every_n_epochs is None:
            return super().on_train_epoch_end(trainer, pl_module)
        if self.every_n_epochs <= 0:
            return super().on_train_epoch_end(trainer, pl_module)
        if trainer.current_epoch == 0:
            return super().on_train_epoch_end(trainer, pl_module)
        if trainer.current_epoch % self.every_n_epochs == 0:
            self.should_upload = True
        return super().on_train_epoch_end(trainer, pl_module)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.every_n_epochs is not None:
            self._upload_model(f"{self.message}[epoch:{trainer.current_epoch}]")
        return super().on_train_end(trainer, pl_module)

    def _upload_model(self, message: str) -> None:
        try:
            subprocess.run(
                ["kaggle", "datasets", "version", "-m", message], cwd=self.model_dir
            )
        except:
            print(traceback.format_exc())
            raise
