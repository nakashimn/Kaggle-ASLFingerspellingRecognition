import argparse
import glob
import importlib
import os
import pathlib
import shutil
import traceback
from argparse import Namespace
from typing import Any

import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import Dataset

from components.augmentations import Augmentation
from components.preprocessor import DataPreprocessor
from components.trainer import Trainer
from components.utility import fix_seed, print_info


def update_config(config: dict[str, Any], filepath_config: str) -> None:
    # copy ConfigFile from temporal_dir to model_dir
    dirpath_model = pathlib.Path(config["path"]["model_dir"])
    filename_config = pathlib.Path(filepath_config).name
    shutil.copy2(filepath_config, str(dirpath_model / filename_config))


def remove_exist_models(config: dict[str, Any]) -> None:
    filepaths_ckpt = glob.glob(f"{config['path']['model_dir']}/*.ckpt")
    for fp in filepaths_ckpt:
        os.remove(fp)


def import_classes(
    config: dict[str, Any]
) -> tuple[LightningModule, Dataset, LightningDataModule, Augmentation]:
    # import Classes dynamically
    Model = getattr(
        importlib.import_module(f"components.models"), config["model"]["ClassName"]
    )
    Data = getattr(
        importlib.import_module(f"components.datamodule"),
        config["datamodule"]["dataset"]["ClassName"],
    )
    DataModule = getattr(
        importlib.import_module(f"components.datamodule"),
        config["datamodule"]["ClassName"],
    )
    Aug = getattr(
        importlib.import_module(f"components.augmentations"),
        config["augmentation"]["ClassName"],
    )
    # debug message
    print_info(
        {
            "Components": {
                "Model": Model.__name__,
                "Dataset": Data.__name__,
                "DataModule": DataModule.__name__,
                "Augmentation": Aug.__name__,
            }
        }
    )
    return Model, Data, DataModule, Aug


def get_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="stem of config filepath.", type=str, required=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    # args
    args = get_args()
    config = importlib.import_module(f"config.{args.config}").config
    from config.sample import config

    # import Classes
    Model, Data, DataModule, Aug = import_classes(config)

    # update config
    update_config(config, f"./config/{args.config}.py")

    # create model dir
    os.makedirs(config["path"]["model_dir"], exist_ok=True)

    # torch setting
    torch.set_float32_matmul_precision("medium")

    # Preprocess
    remove_exist_models(config)
    fix_seed(config["random_seed"])
    data_preprocessor = DataPreprocessor(config)
    df_train = data_preprocessor.train_dataset()

    # Training
    trainer = Trainer(Model, DataModule, Data, Aug, df_train, config)
    trainer.run()
