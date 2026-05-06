import datetime
from pathlib import Path

import numpy as np
import torch
import toml

from src.utils.util import mkdir

### Logger/Tracker
class Logger:
    """
    Log the infos which are necessary for analyzing the experiments.
    """
    def __init__(self, fname: str, session_dir: Path) -> None:
        """
        Args:
            fname (str): Name of logger file.
            session_dir (Path): Directory to save the log file.
        """
        self.log_path = session_dir / f"{fname}.log"
        self.log_dict = {}

    def write(self, msg: dict) -> None:
        """
        Write the log information to log_dict.

        Args:
            msg (dict): Dictionary containing the dictionary.
        """
        for k, v in msg.items():
            self.log_dict[k] = v

    def save(self) -> None:
        """
        Save the logged information to log file.
        """
        with open(self.log_path, 'w') as f:
            toml.dump(self.log_dict, f)


class StatTracker:
    """
    Helper class to track the metrics during the learning.
    """
    def __init__(self, n_slots: int, output_dir: Path) -> None:
        """
        Args:
            n_slots (int): Max number of elements to store.
            output_dir (Path): Directory to save the npy files containing the statistics.
        """
        self.n_slots = n_slots
        self.output_dir = output_dir
        self.results = {}
        self.indices = {}

    def add(self, value_dict: dict) -> None:
        """
        Add given statistics to saving array.

        Args:
            value_dict (dict): Dictionary containing the statistics.
        """
        for k in value_dict.keys():
            # Ignore the none values
            if value_dict[k] is None:
                continue

            # Check the type of values to record
            if isinstance(value_dict[k], list):
                if isinstance(value_dict[k][0], torch.Tensor):
                    value_dict[k] = torch.stack(value_dict[k], dim=0).numpy()
                    if value_dict[k].ndim >= 3:
                        value_dict[k] = np.squeeze(value_dict[k], axis=1)
                else:
                    value_dict[k] = np.array(value_dict[k])
            if isinstance(value_dict[k], torch.Tensor):
                value_dict[k] = value_dict[k].numpy()

            if not isinstance(value_dict[k], (np.ndarray)):
                raise ValueError("Values to record needs to be either list, np array, or torch tensor.")

            # Allocate the memory if the given value was not given in the past
            if not k in self.results:
                self.results[k] = np.zeros([self.n_slots] + list(value_dict[k].shape[1:] if value_dict[k].ndim >= 2 else value_dict[k].shape))
                self.indices[k] = 0

            # Register the value
            n_entries = value_dict[k].shape[0]
            end_idx = self.indices[k] + n_entries
            # Handle cases when the buffer capacity is not enough
            if end_idx <= self.n_slots:
                self.results[k][self.indices[k]:end_idx] = value_dict[k]
                self.indices[k] = end_idx
                # Consider the case when the slots are full (auto save)
                if self.indices[k] == self.n_slots:
                    self.save([k])
                    self.reset([k])
            else:
                # Save as much as possible
                rem = end_idx - self.n_slots
                self.results[k][self.indices[k]:self.n_slots] = value_dict[k][:n_entries-rem]
                self.indices[k] = self.n_slots

                # Store reseults in files
                self.save([k])
                self.reset([k])

                # Save remainders
                self.results[k][:rem] = value_dict[k][n_entries-rem:]
                self.indices[k] += rem

    def reset(self, keys: list) -> None:
        """
        Reset the saving array for the given keys.

        Args:
            keys (list): Dictionary containing the statistics.
        """
        for key in keys:
            self.results[key][:] = 0
            self.indices[key] = 0

    def save(self, keys: list) -> None:
        """
        Save the array for the given keys to npy file.

        Args:
            keys (list): Dictionary containing the statistics.
        """
        for key in keys:
            value = self.results[key][:self.indices[key]]
            fname = self.output_dir / f"{key}.npy"
            np.save(fname, value)

    def save_non_empty(self) -> None:
        """
        Save non-empty arrays to npy file.
        """
        save_keys = []
        for k, v in self.indices.items():
            if v != 0:
                save_keys.append(k)

        self.save(save_keys)
