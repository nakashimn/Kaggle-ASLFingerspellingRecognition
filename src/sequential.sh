#! /bin/bash

# preprocess split dataset
python ./tools/split_dataset.py

# all w/ CTCLoss
python main.py -c asl_trial_v3

# dropna w/ CTCLoss
python main.py -c asl_trial_v4

# includena w/ CTCLoss
python main.py -c asl_trial_v5


# preprocess split dataset
python ./tools/interpolate_dataset.py

# all deeper w/ CTCLoss
python main.py -c asl_trial_v6
