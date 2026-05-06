import argparse
import itertools
import os
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

import experiments
from src.utils import (
    GridSearch,
    load_config,
    wrap_config_dict,
    seed_everything,
)


### Entrypoint for the experiments
if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-c", type=str, help="Path to the configuration file.")
    parser.add_argument("--config_idx", type=int, default=None, help="Index of the config file.")
    parser.add_argument("--root_dir", type=str, default=".", help="Home directory for saving files.")
    parser.add_argument("--num_resub", type=int, default=0, help="Counter for the number of resubmissions")
    args = parser.parse_args()
    root_dir = Path(args.root_dir)

    # Load config
    config_dict = load_config(args.config_file)
    if isinstance(config_dict["seed"], int):
        seed_everything(config_dict["seed"])
        config_dict["seed"] = list(map(int, np.random.randint(0, 10**6, size=config_dict["n_seeds"])))

    if config_dict["device"] == "mig":
        device_id = int(args.config_idx) % 10
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("ALL_MIG_IDS").split(',')[device_id]
        config_dict["device"] = "cuda"

    # Generate all the combinations of hyperparameters
    grid_search = GridSearch(config_dict, return_dir_name=True)
    torch.set_num_threads(1)

    if args.config_idx is None:
        for c, dir_name in grid_search:
            # Save and process config
            config = wrap_config_dict(c)

            # Set seed for everything
            seed_everything(config.seed)

            manager = getattr(experiments, config.experiment_type)(config, root_dir, dir_name)
            manager.run_experiment()
    else:
        cur_config_dict, dir_name = grid_search.get_by_idx(args.config_idx)
        cur_config_dict["config_idx"] = args.config_idx
        config = wrap_config_dict(cur_config_dict)

        manager = getattr(experiments, config.experiment_type)(config, root_dir, dir_name, args.num_resub)
        manager.run_experiment()
