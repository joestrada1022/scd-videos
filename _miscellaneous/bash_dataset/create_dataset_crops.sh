#!/bin/bash

#SBATCH --job-name=cr_10
#SBATCH --time=6:00:00
#SBATCH --mem=1g
#SBATCH --ntasks=1
# --reservation=infsys
# --gres=gpu:v100:1
#SBATCH --array=0-27

module load cuDNN/7.6.4.38-gcccuda-2019b

source /data/p288722/python_venv/scd_videos/bin/activate

python3 /home/p288722/git_code/scd_videos_tf/dataset/frames/3.create_dataset_crops.py --device_id=${SLURM_ARRAY_TASK_ID} --count=10