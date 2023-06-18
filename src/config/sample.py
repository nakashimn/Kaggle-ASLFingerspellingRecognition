config = {
    "random_seed": 57,
    "pred_device": "cuda",
    "label": "label",
    "labels": ["A", "B", "C"],
    "experiment_name": "sample",
    "path": {
        "traindata": "/workspace/data/sample/img/",
        "trainmeta": "/workspace/data/sample/labels.csv",
        "testdata": "/workspace/data/sample/img/",
        "testmeta": "/workspace/data/sample/labels.csv",
        "preddata": "/workspace/data/sample/img/",
        "temporal_dir": "../tmp/artifacts/",
        "model_dir": "/kaggle/input/model/",
        "ckpt_dir": "/workspace/tmp/checkpoint/"
    },
    "modelname": "best_loss",
    "init_with_checkpoint": False,
    "resume": False,
    "upload_every_n_epochs": 5,
    "pred_ensemble": False,
    "train_with_alldata": True
}
config["augmentation"] = {
    "ClassName": "SpecAugmentation"
}
config["model"] = {
    "ClassName": "EfficientNetModel",
    "base_model_name": "tf_efficientnet_b0_ns",
    "num_class": len(config["labels"]),
    "fc_mid_dim": 1024,
    "gradient_checkpointing": True,
    "mixup": {
        "alpha": 0.2
    },
    "label_smoothing": {
        "eps": 0.01
    },
    "loss": {
        "name": "nn.CrossEntropyLoss",
        "params": {
            "weight": None
        }
    },
    "optimizer":{
        "name": "optim.RAdam",
        "params":{
            "lr": 1e-3
        },
    },
    "scheduler":{
        "name": "optim.lr_scheduler.CosineAnnealingWarmRestarts",
        "params":{
            "T_0": 40,
            "eta_min": 0,
        }
    }
}
config["earlystopping"] = {
    "patience": 3
}
config["checkpoint"] = {
    "dirpath": config["path"]["model_dir"],
    "save_top_k": 1,
    "mode": "min",
    "save_last": False,
    "save_weights_only": False
}
config["trainer"] = {
    "accelerator": "gpu",
    "devices": 1,
    "max_epochs": 10,
    "accumulate_grad_batches": 1,
    "deterministic": False,
    "precision": 32
}
config["datamodule"] = {
    "ClassName": "DataModule",
    "dataset":{
        "ClassName": "ImgDataset",
        "base_model_name": config["model"]["base_model_name"],
        "num_class": config["model"]["num_class"],
        "label": config["label"],
        "labels": config["labels"],
        "path": config["path"],
        "mean": 0.485,
        "std": 0.229
    },
    "dataloader": {
        "batch_size": 2,
        "num_workers": 8
    }
}
config["Metrics"] = {
}
