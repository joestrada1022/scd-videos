#!/bin/bash

#SBATCH --job-name=I_frames
#SBATCH --time=12:00:00
#SBATCH --mem=1GB
# --array=27

module load cuDNN/7.6.4.38-gcccuda-2019b
source /data/p288722/python_venv/scd_videos/bin/activate

#python3 /home/p288722/git_code/scd-videos/dataset/frames/1.frame_extractor.py --input_dir="/data/p288722/VISION/dataset" --output_dir="/scratch/p288722/datasets/vision/all_frames" --device_id=${SLURM_ARRAY_TASK_ID}
python3 /home/p288722/git_code/scd-videos/_miscellaneous/I_frames/extract_I_frames.py
