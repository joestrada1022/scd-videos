#!/bin/bash

#SBATCH --job-name=stats
#SBATCH --time=04:00:00
#SBATCH --mem=2GB

module load Python/3.8.6-GCCcore-10.2.0
pip install pillow

python "/home/p288722/git_code/scd_videos_tf/miscelaneous/compare_train_test_split.py"
