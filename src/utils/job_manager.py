import os
import time

from src.utils.util import ConfigWrapper


class JobManager:
    """
    Manager for the slurm jobs. Heavily inspired by https://github.com/qlan3/Explorer.
    """
    def __init__(self, config: ConfigWrapper, slurm_config: dict) -> None:
        """
        Args:
            config (ConfigWrapper): Config related to the cluster.
            slurm_config (dict): Config related to compute resource per job.
        """
        self.jobs = config.job_list
        self.user = config.user
        self.cluster_capacity = config.cluster_capacity
        self.interval = config.interval
        self.sh_file = config.sh_file
        self.mx_jobs = config.mx_jobs
        self.slurm_config = slurm_config

    def submit_multiple(self) -> None:
        """
        Submit jobs where each execute multiple runs.
        """
        while True:
            squeue_cmd = f"squeue -u {self.user} -r"
            cmd_out = os.popen(squeue_cmd).read()
            cmd_out_lines = cmd_out.split('\n')
            cur_jobs = 0
            for line in cmd_out_lines:
                cur_jobs += (self.user in line)

            if cur_jobs < self.cluster_capacity and len(self.jobs) > 0:
                rem_jobs = min(self.cluster_capacity - cur_jobs, len(self.jobs))
                while rem_jobs > 0:
                    if rem_jobs >= self.mx_jobs:
                        self.submit_multiple_jobs(self.mx_jobs)
                        rem_jobs -= self.mx_jobs
                    else:
                        self.submit_multiple_jobs(rem_jobs)
                        rem_jobs = 0

                if len(self.jobs) == 0:
                    print("Submitted all the jobs.")
                    exit()

            time.sleep(self.interval)

    def submit_multiple_jobs(self, rem_jobs: int) -> None:
        """
        Submit single job with multiple runs.

        Args:
            rem_jobs (int): Remaining numbers of jobs.
        """
        # Create input file for the job
        job_lines = '\n'.join(list(map(str, self.jobs[:rem_jobs])))
        with open(f"jobs_{self.slurm_config['job-name']}_{self.jobs[0]}.txt", 'w') as f:
            f.write(job_lines)
        time.sleep(2)

        # Submit job with slurm sbatch
        original_export_command = self.slurm_config["export"]
        self.slurm_config["export"] += f",config_idx={self.jobs[0]},num_resub=0"
        slurm_options = ' '.join([f"--{k}={v}" for k, v in self.slurm_config.items()])
        sbatch_cmd = f"sbatch {slurm_options} {self.sh_file}"
        print(sbatch_cmd)
        cmd_out = os.popen(sbatch_cmd).read()
        print(cmd_out)
        print(f"Submitted jobs {self.jobs[0]} to {self.jobs[rem_jobs-1]}")
        self.slurm_config["export"] = original_export_command
        self.jobs = self.jobs[rem_jobs:]

    def submit(self) -> None:
        """
        Submit jobs where each execute single run.
        """
        while True:
            squeue_cmd = f"squeue -u {self.user} -r"
            cmd_out = os.popen(squeue_cmd).read()
            cmd_out_lines = cmd_out.split('\n')
            cur_jobs = 0
            for line in cmd_out_lines:
                cur_jobs += (self.user in line)

            if cur_jobs < self.cluster_capacity and len(self.jobs) > 0:
                rem_jobs = min(self.cluster_capacity - cur_jobs, len(self.jobs))
                if rem_jobs > 0:
                    self.submit_jobs(rem_jobs)

                if len(self.jobs) == 0:
                    print("Submitted all the jobs.")
                    exit()

            time.sleep(self.interval)

    def submit_jobs(self, rem_jobs: int) -> None:
        """
        Submit single job with single run.

        Args:
            rem_jobs (int): Remaining numbers of jobs.
        """
        flg = True
        for i in range(1, rem_jobs):
            if self.jobs[i] != self.jobs[i-1] + 1:
                flg = False
                break

        if flg:
            job_indices = f"{self.jobs[0]}-{self.jobs[rem_jobs-1]}"
        else:
            job_indices = ','.join(self.jobs[:rem_jobs])

        slurm_options = ' '.join([f"--{k}={v}" for k, v in self.slurm_config.items()])
        sbatch_cmd = f"sbatch --array={job_indices} {slurm_options} {self.sh_file}"
        cmd_out = os.popen(sbatch_cmd).read()
        print(cmd_out)
        print(f"Submitted jobs from {self.jobs[0]} to {self.jobs[rem_jobs-1]}")
        self.jobs = self.jobs[rem_jobs:]
