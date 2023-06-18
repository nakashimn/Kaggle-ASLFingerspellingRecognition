import argparse
import copy
import datetime
import gc
import glob
import importlib
import os
import pathlib
import random
import shutil
import subprocess
import traceback

import numpy as np
import optuna
import pandas as pd
import pytorch_lightning as pl
import sklearn.model_selection
import torch
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from components.preprocessor import DataPreprocessor
from components.validations import (ConfusionMatrix, F1Score, LogLoss, MinLoss,
                                    ValidResult)


class Optimizer:
    def __init__(self, Model, DataModule, Dataset, Augmentation, df_train, config):
        # const
        self.config = config
        self.df_train = df_train
        self.timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")

        # Class
        self.Model = Model
        self.DataModule = DataModule
        self.Dataset = Dataset
        self.Augmentation = Augmentation

    def run(self, trial):
        # train
        self._update_config(trial)
        mlflow_logger = self._create_mlflow_logger()

        try:
            # create datamodule
            transforms = self._create_transforms()
            datamodule = self._create_datamodule(transforms=transforms)

            # train with all data
            collapse_level = self._train(datamodule, logger=mlflow_logger)

            # finalize logger
            mlflow_logger.finalize("success")

        except:
            mlflow_logger.finalize("failed")
            mlflow_logger.experiment.delete_run(mlflow_logger.run_id)
        return collapse_level

    def _update_config(self, trial):
        self.config["augmentation"]["freq_mask"] = {
            "probability": trial.suggest_float("freq_mask_prob", 0.0, 1.0),
            "max": trial.suggest_int("freq_mask_max", 0, 200),
        }
        self.config["augmentation"]["time_mask"] = {
            "probability": trial.suggest_float("time_mask_prob", 0.0, 1.0),
            "max": trial.suggest_int("time_mask_max", 0, 300),
        }
        self.config["augmentation"]["fadein"] = {
            "probability": trial.suggest_float("fadein_prob", 0.0, 1.0),
            "max": trial.suggest_float("fadein_max", 0, 1.0),
        }
        self.config["augmentation"]["fadein"] = {
            "probability": trial.suggest_float("fadeout_prob", 0.0, 1.0),
            "max": trial.suggest_float("fadeout_max", 0, 1.0),
        }
        self.config["model"]["out_dim"] = trial.suggest_int("out_dim", 128, 2048)
        self.config["model"]["projection_hidden_dim"] = trial.suggest_int(
            "projection_hidden_dim", 128, 2048
        )

    def _create_mlflow_logger(self, fold=None):
        # create Logger instance
        experiment_name = f"{self.config['experiment_name']}[{self.timestamp}]"
        run_name = "All" if (fold is None) else fold
        mlflow_logger = MLFlowLogger(experiment_name=experiment_name, run_name=run_name)
        # save hyper_params
        mlflow_logger.log_hyperparams(self.config)
        mlflow_logger.experiment.log_artifact(
            mlflow_logger.run_id, f"./config/{args.config}.py"
        )
        # debug message
        print("================================================")
        print("MLflow:")
        print(f"  experiment_name : {experiment_name}")
        print(f"  run_name        : {run_name}")
        print(f"  run_id          : {mlflow_logger.run_id}")
        print("================================================")
        return mlflow_logger

    def _create_transforms(self):
        transforms = {
            "train": self.Augmentation(config["augmentation"]),
            "valid": None,
            "pred": None,
        }
        return transforms

    def _create_datamodule(self, idx_train=None, idx_val=None, transforms=None):
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
            Dataset=self.Dataset,
            config=self.config["datamodule"],
            transforms=transforms,
        )
        return datamodule

    def _create_model(self, filepath_checkpoint=None):
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

    def _define_monitor_value(self, fold=None):
        return "train_loss" if (fold is None) else "val_loss"

    def _define_checkpoint_name(self, fold=None):
        checkpoint_name = f"{self.config['modelname']}"
        if fold is None:
            return checkpoint_name
        checkpoint_name += f"_{fold}"
        return checkpoint_name

    def _define_checkpoint_path(self, checkpoint_name):
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

    def _define_callbacks(self, callback_config):
        # define earlystopping
        earlystopping = EarlyStopping(
            **callback_config["earlystopping"], **self.config["earlystopping"]
        )
        # define learning rate monitor
        lr_monitor = callbacks.LearningRateMonitor()
        # define check point
        loss_checkpoint = callbacks.ModelCheckpoint(
            **callback_config["checkpoint"], **self.config["checkpoint"]
        )
        # define model uploader
        model_uploader = ModelUploader(
            model_dir=self.config["path"]["model_dir"],
            every_n_epochs=self.config["upload_every_n_epochs"],
            message=self.config["experiment_name"],
        )

        callback_list = [earlystopping, lr_monitor, loss_checkpoint, model_uploader]
        return callback_list

    def _train(self, datamodule, fold=None, min_delta=0.0, min_loss=None, logger=None):
        # switch mode
        monitor = self._define_monitor_value(fold)

        # define saved checkpoint name
        checkpoint_name = self._define_checkpoint_name(fold)

        # define loading checkpoint
        filepath_checkpoint = self._define_checkpoint_path(checkpoint_name)

        # define pytorch_lightning callbacks
        callback_config = {
            "earlystopping": {
                "monitor": monitor,
                "min_delta": min_delta,
                "stopping_threshold": min_loss,
            },
            "checkpoint": {"filename": checkpoint_name, "monitor": monitor},
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
            **self.config["trainer"],
        )

        # train
        trainer.fit(
            model, datamodule=datamodule, ckpt_path=filepath_checkpoint["resume"]
        )
        collapse_level = copy.deepcopy(model.collapse_level)

        # logging
        if logger is not None:
            logger.experiment.log_artifact(logger.run_id, filepath_checkpoint["save"])

        # clean memory
        del model
        gc.collect()

        return collapse_level

    def _valid(self, datamodule, fold, logger=None):
        # load model
        model = self.Model(self.config["model"])
        model.eval()

        # define trainer
        trainer = pl.Trainer(logger=logger, **self.config["trainer"])

        # validation
        filepath_checkpoint = (
            f"{self.config['path']['model_dir']}/{self.config['modelname']}_{fold}.ckpt"
        )
        trainer.validate(model, datamodule=datamodule, ckpt_path=filepath_checkpoint)

        # get result
        val_probs = copy.deepcopy(model.val_probs)
        val_labels = copy.deepcopy(model.val_labels)

        # clean memory
        del model
        gc.collect()

        return val_probs, val_labels


def update_config(config, filepath_config):
    # copy ConfigFile from temporal_dir to model_dir
    dirpath_model = pathlib.Path(config["path"]["model_dir"])
    filename_config = pathlib.Path(filepath_config).name
    shutil.copy2(filepath_config, str(dirpath_model / filename_config))


def remove_exist_models(config):
    filepaths_ckpt = glob.glob(f"{config['path']['model_dir']}/*.ckpt")
    for fp in filepaths_ckpt:
        os.remove(fp)


def fix_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_classes(config):
    # import Classes dynamically
    Model = getattr(
        importlib.import_module(f"components.models"), config["model"]["ClassName"]
    )
    Dataset = getattr(
        importlib.import_module(f"components.datamodule"),
        config["datamodule"]["dataset"]["ClassName"],
    )
    DataModule = getattr(
        importlib.import_module(f"components.datamodule"),
        config["datamodule"]["ClassName"],
    )
    Augmentation = getattr(
        importlib.import_module(f"components.augmentation"),
        config["augmentation"]["ClassName"],
    )
    # debug message
    print("================================================")
    print(f"Components:")
    print(f"  Model        : {Model.__name__}")
    print(f"  Dataset      : {Dataset.__name__}")
    print(f"  DataModule   : {DataModule.__name__}")
    print(f"  Augmentation : {Augmentation.__name__}")
    print("================================================")
    return Model, Dataset, DataModule, Augmentation


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="stem of config filepath.", type=str, required=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    # args
    args = get_args()
    config = importlib.import_module(f"config.{args.config}").config

    # import Classes
    Model, Dataset, DataModule, Augmentation = import_classes(config)

    # update config
    update_config(config, f"./config/{args.config}.py")

    # create model dir
    os.makedirs(config["path"]["model_dir"], exist_ok=True)

    # torch setting
    torch.set_float32_matmul_precision("medium")

    try:
        # Preprocess
        remove_exist_models(config)
        fix_seed(config["random_seed"])
        data_preprocessor = DataPreprocessor(config)
        df_train = data_preprocessor.train_dataset_for_pretrain()

        # Training
        optimizer = Optimizer(
            Model, DataModule, Dataset, Augmentation, df_train, config
        )

        # Search HyperParams
        study = optuna.create_study(direction="minimize")
        study.optimize(optimizer.run, n_trials=100, timeout=3600 * 9)

    finally:
        print(f"Number of finished trials: {len(study.trials)}")
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
