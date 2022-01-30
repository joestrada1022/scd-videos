from pathlib import Path
from pprint import pprint


def test_failed_jobs():
    files = Path(r'/home/p288722/git_code/scd_videos_first_revision/_miscellaneous/slurm_jobsripts').glob('slurm-2222*')
    job_status = {}
    for file in files:
        with open(file) as f:
            lines = f.readlines()
        for line in lines:
            if 'Job ID' in line:
                job_id = int(line.strip().split('_')[-1])
            if 'State' in line:
                job_status[job_id] = line.strip().split(':')[-1].strip()

    job_status = {x: job_status[x] for x in sorted(job_status)}
    pprint(job_status)


def print_val_loss():
    slurm_output = Path(r'/scratch/p288722/runtime_data/scd_videos_first_revision/05_contrastive_loss/'
                        r'slurm-22244039-rn-50-ft.out')
    with open(slurm_output) as f:
        for line in f:
            if 'val_loss' in line:
                print(line)


if __name__ == '__main__':
    print_val_loss()
