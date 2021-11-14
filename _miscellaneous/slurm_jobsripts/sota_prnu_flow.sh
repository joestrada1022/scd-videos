#!/bin/bash
#SBATCH --job-name=prnu-50-flat-frames-resize
#SBATCH --time=6:00:00
#SBATCH --mem=120g
#SBATCH --partition=lab
# --gres=gpu:v100:1
#SBATCH --reservation=infsys
#SBATCH --cpus-per-task=39
#SBATCH --dependency=afterok:22048647

# create the following directory manually
#SBATCH --chdir=/scratch/p288722/runtime_data/scd_videos_first_revision/sota_prnu_old_baseline_split
#SBATCH --output=slurm-%j-%x.out
#SBATCH --error=slurm-%j-%x.out

module load cuDNN/7.6.4.38-gcccuda-2019b
source /data/p288722/python_venv/scd_videos/bin/activate

python3 /home/p288722/git_code/scd_videos_first_revision/_sota/prnu-python/run_prnu_flow.py
