config = {
    "random_seed": 57,
    "pred_device": "cuda",
    "n_splits": 5,
    "label": "phrase",
    "experiment_name": "asl-trial-v1",
    "path": {
        "dataset": "/kaggle/input/asl-fingerspelling/",
        "traindata": "/kaggle/input/asl-fingerspelling/train_landmarks/",
        "trainmeta": "/kaggle/input/asl-fingerspelling/train.csv",
        "testdata": "/kaggle/input/asl-fingerspelling/train_landmarks/",
        "testmeta": "/kaggle/input/asl-fingerspelling/train.csv",
        "preddata": "/kaggle/input/asl-fingerspelling/train_landmarks/",
        "predmeta": "/kaggle/input/asl-fingerspelling/train.csv",
        "vocab_file": "/kaggle/input/asl-fingerspelling/character_to_prediction_index.json",
        "temporal_dir": "../tmp/artifacts/",
        "model_dir": "/kaggle/input/asl-trial-v1/",
        "ckpt_dir": "/workspace/tmp/checkpoint/"
    },
    "vocab_size": 62,
    "special_token_ids": {
        "bos_token_id": 59,
        "eos_token_id": 60,
        "unk_token_id": 61,
        "pad_token_id": 59,
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
    "ClassName": "T5forASLModel",
    "gradient_checkpointing": True,
    "fillna_val": 0.0,
    "dim_input": 184,
    "dim_output": 59,
    "max_length": 50,
    "special_token_ids": config["special_token_ids"],
    "T5_config": {
        "bos_token_id": config["special_token_ids"]["bos_token_id"],
        "eos_token_id": config["special_token_ids"]["eos_token_id"],
        "pad_token_id": config["special_token_ids"]["pad_token_id"],
        "decoder_start_token_id": config["special_token_ids"]["bos_token_id"],
        "vocab_size": config["vocab_size"],
        "d_model": 256,
        "d_ff": 1024,
        "d_kv": 64,
        "dropout_rate": 0.1,
        "relative_attention_max_distance": 128,
        "relative_attention_num_buckets": 32,
        "num_heads": 8,
        "num_layers": 6,
        "num_decoder_layers": 6,
        "use_cache": False,
    },
    "metrics": config["metrics"],
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
