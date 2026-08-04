from src.utils import mkdir

import torch


class Experiment:
    def __init__(self, config, root_dir, session_name, num_resub=0):
        self.config = config
        if "ALE" in session_name:
            session_name = session_name.replace("ALE/", "")
        self.session_name = session_name

        self.root_dir = root_dir
        self.exp_dir = self.root_dir / config.experiment_name
        self.save_dir = self.exp_dir / "results" / self.session_name / f"{str(config.seed)}"

        print(f"Root directory is: {self.root_dir.as_posix()}")

        mkdir(self.save_dir)

        self.device = torch.device(self.config.device)
        self.num_resub = num_resub

    def run_experiment(self):
        raise NotImplementedError()
