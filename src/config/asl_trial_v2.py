config = {
    "random_seed": 57,
    "pred_device": "cuda",
    "n_splits": 5,
    "label": "phrase",
    "experiment_name": "asl-trial-v2",
    "path": {
        "dataset": "/kaggle/input/asl-fingerspelling/",
        "traindata": "/kaggle/input/asl-fingerspelling/train_landmarks/",
        "trainmeta": "/kaggle/input/asl-fingerspelling/train.csv",
        "testdata": "/kaggle/input/asl-fingerspelling/train_landmarks/",
        "testmeta": "/kaggle/input/asl-fingerspelling/train.csv",
        "preddata": "/kaggle/input/asl-fingerspelling/train_landmarks/",
        "predmeta": "/kaggle/input/asl-fingerspelling/train.csv",
        "vocab_file": "/kaggle/input/asl-fingerspelling/character_to_prediction_index.json",
        "norm_factor": "/workspace/data/norm_factor_trial_v2.csv",
        "temporal_dir": "../tmp/artifacts/",
        "model_dir": "/kaggle/input/asl-trial-v2/",
        "ckpt_dir": "/workspace/tmp/checkpoint/"
    },
    "vocab_size": 63,
    "special_token_ids": {
        "bos_token_id": 59,
        "eos_token_id": 60,
        "unk_token_id": 61,
        "pad_token_id": 62,
    },
    "modelname": "best_loss",
    "monitor": {
        "cross_valid": {
            "metrics": "levenshtein_distance",
            "mode": "max",
        },
        "all": {
            "metrics": "train_loss",
            "mode": "min",
        },
    },
    "init_with_checkpoint": False,
    "resume": False,
    "upload_every_n_epochs": None,
    "pred_ensemble": False,
    "train_with_alldata": True,
}
config["augmentation"] = {
    "ClassName": None,
}
config["metrics"] = {
    "levenshtein_distance": {
        "charmap_path": config["path"]["vocab_file"],
        "normalize": True,
    }
}
config["model"] = {
    "ClassName": "TransformerforASLModel",
    "gradient_checkpointing": True,
    "fillna_val": 0.0,
    "norm_factor_path": config["path"]["norm_factor"],
    "Transformer": {
        "dim_model": 184,
        "dim_output": 63,
        "dim_ff": 1024,
        "vocab_size": config["vocab_size"],
        "bos_idx": config["special_token_ids"]["bos_token_id"],
        "pad_idx": config["special_token_ids"]["pad_token_id"],
        "max_len_encoder": 1000,
        "max_len_decoder": 50,
        "n_layer": 4,
        "n_head": 8,
        "dropout_rate": 0.0,
        "layer_norm_eps": 1e-7,
        "prenorm": False,
        "device": config["pred_device"],
    },
    "metrics": config["metrics"],
    "optimizer":{
        "name": "optim.RAdam",
        "params":{
            "lr": 1e-3
        },
    },
    "loss": {
        "name": "nn.CrossEntropyLoss",
        "params": {
            "weight": None,
            "ignore_index": config["special_token_ids"]["pad_token_id"],
            "label_smoothing": 0.0,
        }
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
    "save_last": False,
    "save_weights_only": False
}
config["trainer"] = {
    "accelerator": "gpu",
    "devices": 1,
    "max_epochs": 20,
    "accumulate_grad_batches": 1,
    "deterministic": True,
    "precision": 32
}
config["datamodule"] = {
    "ClassName": "DataModule",
    "dataset":{
        "ClassName": "T5ForASLDataset",
        "label": config["label"],
        "feat_max_length": 400,
        "select_col": [f"x_right_hand_{i}" for i in range(21)] \
                      + [f"y_right_hand_{i}" for i in range(21)] \
                      + [f"x_left_hand_{i}" for i in range(21)] \
                      + [f"y_left_hand_{i}" for i in range(21)] \
                      + [f"x_face_{i}" for i in [
                            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                            95, 88, 178, 87, 14, 317, 402, 318, 324, 308
                        ]] \
                      + [f"y_face_{i}" for i in [
                            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                            95, 88, 178, 87, 14, 317, 402, 318, 324, 308
                        ]] \
                      + [f"x_pose_{i}" for i in [
                          13, 15, 17, 19, 21, 14, 16, 18, 20, 22
                        ]] \
                      + [f"y_pose_{i}" for i in [
                          13, 15, 17, 19, 21, 14, 16, 18, 20, 22
                        ]],
        "phrase_max_length": 35,
        "path": config["path"],
        "special_token_ids": config["special_token_ids"],
    },
    "dataloader": {
        "batch_size": 128,
        "num_workers": 4
    }
}
