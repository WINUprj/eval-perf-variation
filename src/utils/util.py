import os
from pathlib import Path
import random
import time

import numpy as np
import toml
import torch

### Seeding ###
def seed_everything(seed: int) -> None:
    """
    Set seed of the entire environment.

    Args:
        seed (int): Random seed.
    """
    # Standard Python3 library-related
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # NumPy-related
    np.random.seed(seed)

    # Torch-related
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

### File handling ###
def get_project_root_dir() -> Path:
    """
    Return the root directory of the project (git root directory).

    Returns:
        Path: Project's root directory.
    """
    return Path(__file__).parent.parent.parent


def mkdir(directory: Path) -> None:
    """
    Create the given directory.

    Args:
        directory (Path): Directory path to make.
    """
    if not directory.is_dir():
        directory.mkdir(parents=True)

### Config handling ###
class ConfigWrapper(dict):
    """
    Wrapper for the config dictionary.
    """
    def __getattribute__(self, key: str):
        return self[key]


def wrap_config_dict(config_dict: dict) -> ConfigWrapper:
    """
    Wrap config dictionary with the ConfigWrapper.

    Args:
        config_dict (dict): Dictionary containing the configuration.

    Returns:
        ConfigWrapper: Config dict wrapped by ConfigWrapper.
    """
    for k in config_dict:
        if isinstance(config_dict[k], dict):
            wrap_config_dict(config_dict[k])
            config_dict[k] = ConfigWrapper(config_dict[k])

    return ConfigWrapper(config_dict)


def load_config(config_path: Path) -> dict:
    """
    Load configuration dictionary from the toml file.

    Args:
        config_path (Path): Path of toml file containing the configuration.

    Returns:
        dict: Dictionary containing the configurtation.
    """
    with open(config_path, 'r') as f:
        config = toml.load(f)
    return config

### Time management ###
def duration(st_time: float) -> tuple[int, int, float]:
    """
    Reformat the floating seconds into hours, minutes, and seconds.

    Args:
        st_time (float): Time when program started running.

    Returns:
        tuple[int, int, float]: Elapsed time in hours, minutes, and seconds.
    """
    total_secs = time.time() - st_time
    hours = total_secs // 3600
    rem = total_secs % 3600
    mins = rem // 60
    secs = int(rem - (mins * 60))

    return hours, mins, secs

### Statistics ###
def negative_frequency(arr: np.ndarray) -> float:
    """
    Compute a ratio of negative values in the given array.

    Args:
        arr (np.ndarray): Array of numerals.

    Returns:
        float: Ratio of negative values in the given array.
    """
    n_data = arr.shape[0]
    return (arr < 0).sum() / n_data


def frequency(arr):
    """
    Compute a ratio of non-zero values in the given array.

    Args:
        arr (np.ndarray): Array of numerals.

    Returns:
        float: Ratio of non-zero values in the given array.
    """
    n_data = arr.shape[0]
    return (arr > 0).sum() / n_data
