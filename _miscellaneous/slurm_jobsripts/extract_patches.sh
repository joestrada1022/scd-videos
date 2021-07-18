#!/bin/bash

#SBATCH --job-name=patches
#SBATCH --time=12:00:00
#SBATCH --mem=2g
#--partition=cpu
#--reservation=infsys
# --gres=gpu:v100:1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-7

ssh pg-lab01
module load cuDNN/7.6.4.38-gcccuda-2019b
source /data/p288722/python_venv/scd_videos/bin/activate

python3 /home/p288722/git_code/scd-videos/_miscellaneous/patches/extract_patches.py --device_id=${SLURM_ARRAY_TASK_ID} --ppf=50

python3 /home/p288722/git_code/scd-videos/_miscellaneous/patches/extract_patches.py --ppf=50
