import copy
import datetime
import gc
import os
import pathlib
import sys
import traceback
from typing import Any

import pandas as pd
import pytorch_lightning as pl
import sklearn.model_selection
import torch
from numpy.typing import NDArray
from pytorch_lightning import LightningDataModule, LightningModule, callbacks
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import Dataset

sys.path.append(str(pathlib.Path(__file__).resolve().parents[0]))
from augmentations import Augmentation
from pl_callbacks import ModelUploader
from utility import print_info
from validations import MinLoss, ValidResult


class Trainer:
    def __init__(
        self,
        Model: LightningModule,
        DataModule: LightningDataModule,
        Data: Dataset,
        Aug: Augmentation,
        df_train: pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        # const
        self.config = config
        self.df_train = df_train
        self.timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

        # Class
        self.Model = Model
        self.DataModule = DataModule
        self.Data = Data
        self.Aug = Aug

        # variable
        self.min_loss = MinLoss()

    def run(self) -> None:
        try:
            kfold = sklearn.model_selection.StratifiedKFold(
                n_splits=self.config["n_splits"],
                shuffle=True,
                random_state=self.config["random_seed"],
            )
            for fold, (idx_train, idx_val) in enumerate(
                kfold.split(self.df_train, self.df_train["participant_id"])
            ):
                if fold not in self.config["train_folds"]:
                    continue
                self._run_unit(fold, idx_train, idx_val)

            # train with all data
            if not self.config["train_with_alldata"]:
                return
            self._run_unit()
            return
        except:
            print(traceback.format_exc())
            raise

    def _run_unit(
        self,
        fold: int | None = None,
        idx_train: list[int] | None = None,
        idx_val: list[int] | None = None,
    ) -> None:
        # create logger
        mlflow_logger = self._create_mlflow_logger(fold)

        try:
            # create datamodule
            transforms = self._create_transforms()
            datamodule = self._create_datamodule(
                idx_train, idx_val, transforms=transforms
            )

            # train crossvalid models
            min_loss = self._train(datamodule, fold=fold, logger=mlflow_logger)
            self.min_loss.update(min_loss)

            # valid
            if datamodule.val_dataloader() is not None:
                self._valid(datamodule, fold=fold, logger=mlflow_logger)

            # log
            mlflow_logger.finalize("FINISHED")

        except KeyboardInterrupt:
            print(traceback.format_exc())
            mlflow_logger.finalize("KILLED")
            mlflow_logger.experiment.delete_run(mlflow_logger.run_id)
            raise

        except:
            print(traceback.format_exc())
            mlflow_logger.finalize("FAILED")
            mlflow_logger.experiment.delete_run(mlflow_logger.run_id)
            raise

    def _create_mlflow_logger(self, fold: int | None = None) -> MLFlowLogger:
        # create Logger instance
        experiment_name = f"{self.config['experiment_name']} [{self.timestamp}]"
        run_name = "All" if (fold is None) else f"fold{fold}"
        mlflow_logger = MLFlowLogger(experiment_name=experiment_name, run_name=run_name)
        # save hyper_params
        mlflow_logger.log_hyperparams(self.config)
        # debug message
        print_info(
            {
                "MLflow": {
                    "experiment_name": experiment_name,
                    "run_name": run_name,
                    "run_id": mlflow_logger.run_id,
                }
            }
        )
        return mlflow_logger

    def _create_transforms(self) -> dict[str, Augmentation | None]:
        transforms = {
            "train": self.Aug(self.config["augmentation"]) \
                     if self.Aug is not None else None,
            "valid": None,
            "pred": None,
        }
        return transforms

    def _create_datamodule(
        self,
        idx_train: list[int] | None = None,
        idx_val: list[int] | None = None,
        transforms: dict[str, Augmentation | None] | None = None,
    ) -> LightningDataModule:
        # fold dataset
        if idx_train is None:
            df_train_fold = self.df_train
        else:
            df_train_fold = self.df_train.loc[idx_train].reset_index(drop=True)
        if idx_val is None:
            df_val_fold = None
        else:
            df_val_fold = self.df_train.loc[idx_val].reset_index(drop=True)

        # create datamodule
        datamodule = self.DataModule(
            df_train=df_train_fold,
            df_val=df_val_fold,
            df_pred=None,
            Data=self.Data,
            config=self.config["datamodule"],
            transforms=transforms,
        )
        return datamodule

    def _create_model(self, filepath_checkpoint: str | None = None) -> LightningModule:
        if filepath_checkpoint is None:
            return self.Model(self.config["model"])
        if not self.config["init_with_checkpoint"]:
            return self.Model(self.config["model"])
        if not os.path.exists(filepath_checkpoint):
            return self.Model(self.config["model"])
        model = self.Model.load_from_checkpoint(
            filepath_checkpoint, config=self.config["model"]
        )
        return model

    def _define_earlystopping_monitor(self, fold: int | None = None) -> tuple[str, str]:
        mode: str = "all" if (fold is None) else "cross_valid"
        metrics: str = self.config["monitor"][mode]["metrics"]
        monitor_mode: str = self.config["monitor"][mode]["mode"]
        return (metrics, monitor_mode)

    @staticmethod
    def _define_limit_val_batches(fold: int | None = None) -> float:
        return 0.0 if (fold is None) else 1.0

    def _define_checkpoint_name(self, fold: int | None = None) -> str:
        checkpoint_name = f"{self.config['modelname']}"
        if fold is None:
            return checkpoint_name
        checkpoint_name += f"_{fold}"
        return checkpoint_name

    def _define_checkpoint_path(self, checkpoint_name: str) -> dict[str, str | None]:
        filepath_ckpt_load = f"{self.config['path']['ckpt_dir']}/{checkpoint_name}.ckpt"
        filepath_ckpt_save = (
            f"{self.config['path']['model_dir']}/{checkpoint_name}.ckpt"
        )
        filepath_ckpt = {"init": None, "resume": None, "save": filepath_ckpt_save}
        if not os.path.exists(filepath_ckpt_load):
            return filepath_ckpt
        if self.config["resume"]:
            filepath_ckpt["resume"] = filepath_ckpt_load
        if self.config["init_with_checkpoint"]:
            filepath_ckpt["init"] = filepath_ckpt_load
        return filepath_ckpt

    def _define_callbacks(
        self, callback_config: dict[str, Any]
    ) -> list[callbacks.Callback]:
        callback_list: list[callbacks.Callback] = []
        # define learning rate monitor
        lr_monitor = callbacks.LearningRateMonitor()
        callback_list.append(lr_monitor)
        # define earlystopping
        if "earlystopping" in self.config.keys():
            earlystopping = callbacks.EarlyStopping(
                **callback_config["earlystopping"], **self.config["earlystopping"]
            )
            callback_list.append(earlystopping)
        # define check point
        if "checkpoint" in self.config.keys():
            loss_checkpoint = callbacks.ModelCheckpoint(
                **callback_config["checkpoint"], **self.config["checkpoint"]
            )
            callback_list.append(loss_checkpoint)
        # define model uploader
        model_uploader = ModelUploader(
            model_dir=self.config["path"]["model_dir"],
            every_n_epochs=self.config["upload_every_n_epochs"],
            message=self.config["experiment_name"],
        )
        callback_list.append(model_uploader)
        return callback_list

    def _train(
        self,
        datamodule: LightningDataModule,
        fold: int | None = None,
        min_delta: float = 0.0,
        min_loss: float | None = None,
        logger: MLFlowLogger | None = None,
    ) -> float | None:
        # switch mode
        metrics, monitor_mode = self._define_earlystopping_monitor(fold)
        limit_val_batches = self._define_limit_val_batches(fold)

        # define saved checkpoint name
        checkpoint_name = self._define_checkpoint_name(fold)

        # define loading checkpoint
        filepath_checkpoint = self._define_checkpoint_path(checkpoint_name)

        # define pytorch_lightning callbacks
        callback_config = {
            "earlystopping": {
                "monitor": metrics,
                "mode": monitor_mode,
                "min_delta": min_delta,
                "stopping_threshold": min_loss,
            },
            "checkpoint": {
                "filename": checkpoint_name,
                "monitor": metrics,
                "mode": monitor_mode,
            },
        }
        callback_list = self._define_callbacks(callback_config)

        # create model
        model = self._create_model(filepath_checkpoint["init"])

        # define trainer
        trainer = pl.Trainer(
            logger=logger,
            callbacks=callback_list,
            fast_dev_run=False,
            num_sanity_val_steps=0,
            limit_val_batches=limit_val_batches,
            **self.config["trainer"],
        )

        # train
        trainer.fit(
            model, datamodule=datamodule, ckpt_path=filepath_checkpoint["resume"]
        )
        min_loss = copy.deepcopy(model.min_loss)

        # logging
        if logger is not None:
            logger.experiment.log_artifact(logger.run_id, filepath_checkpoint["save"])

        # clean memory
        del model
        gc.collect()

        return min_loss

    def _valid(
        self,
        datamodule: LightningDataModule,
        fold: int | None,
        logger: MLFlowLogger | None = None,
    ) -> None:
        # load model
        model = self.Model(self.config["model"])
        model.eval()

        # define trainer
        trainer = pl.Trainer(
            logger=logger,
            **self.config["trainer"],
        )

        # validation
        filepath_checkpoint = (
            f"{self.config['path']['model_dir']}/{self.config['modelname']}_{fold}.ckpt"
        )
        trainer.validate(model, datamodule=datamodule, ckpt_path=filepath_checkpoint)

        # clean memory
        del model
        gc.collect()
