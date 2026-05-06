import copy
import itertools


class GridSearch:
    """
    Generate copies of configs to conduct the grid search.
    """
    def __init__(self, config_dict: dict, return_dir_name=True) -> None:
        """
        Args:
            config_dict (dict): Dictionary containing configuration.
            return_dir_name (bool): Specify whether to get the directory name based on the configuration.
        """
        self.config_dict = config_dict
        self.return_dir_name = return_dir_name

        # Get all the combinations of lists
        dicts = [(self.config_dict, '')]
        self.search_keys = []
        self.search_values = []
        self.idx = 0
        while len(dicts) > 0:
            cur_dict, prefix = dicts.pop()
            for k, v in cur_dict.items():
                if isinstance(v, dict):
                    if prefix == '':
                        new_prefix = k
                    else:
                        new_prefix = prefix + f".{k}"

                    dicts.append((v, new_prefix))
                elif isinstance(v, list):
                    if prefix == '':
                        key = k
                    else:
                        key = prefix + f".{k}"

                    self.search_keys.append(key)
                    self.search_values.append(v)

                    if key == "task.name":
                        task_idx = len(self.search_keys) - 1

        if len(self.search_keys) > 0:
            self.search_values = sorted(list(itertools.product(*self.search_values)), key=lambda x: x[task_idx])

    def __len__(self):
        return len(self.search_values)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            comb = self.search_values[self.idx]
            cur_config_dict = self.prep_config_dict(comb)

            dir_name = None
            if self.return_dir_name:
                dir_name = self.prep_dir_name(comb)

            self.idx += 1
            return cur_config_dict, dir_name
        except:
            raise StopIteration()

    def get_by_idx(self, i: int) -> tuple[dict, str | None]:
        """
        Get configuration for single run based on the index.

        Args:
            i (int): Index of configuration.

        Returns:
            dict: Config dict for single run.
            str: Directory name based on the configuration.
        """
        comb = self.search_values[i]
        d = self.prep_config_dict(comb)

        dir_name = None
        if self.return_dir_name:
            dir_name = self.prep_dir_name(comb)

        return d, dir_name

    def prep_config_dict(self, comb: list) -> dict:
        """
        Get configuration dictionary for single run.

        Args:
            comb (list): Combination of configurations.

        Returns:
            dict: Config dict for single run.
        """
        cur_config_dict = copy.deepcopy(self.config_dict)
        for key, value in zip(self.search_keys, comb):
            temp = cur_config_dict
            split_keys = key.split('.')
            for k in split_keys[:-1]:
                temp = temp[k]
            temp[split_keys[-1]] = value

        return cur_config_dict

    def prep_dir_name(self, comb: list) -> str:
        """
        Create a directory name given the combination of configurations.

        Args:
            comb (list): Combination of configurations.

        Returns:
            str: Directory name based on the configuration.
        """
        dir_name = ''
        for i, (key, value) in enumerate(zip(self.search_keys, comb)):
            if key == "seed" or "layer_sizes" in key:
                continue
            dir_name += f"{key}_{value['name'] if isinstance(value, dict) else value}{'_' if i < len(self.search_keys)-1 else ''}"

        return dir_name
