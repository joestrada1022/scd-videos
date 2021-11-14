#!/bin/bash

#SBATCH --job-name=bm3d
#SBATCH --time=16:00:00
#SBATCH --mem=4GB
#SBATCH --array=0-200

module purge
module load OpenBLAS/0.3.15-GCC-10.3.0
source /data/p288722/python_venv/scd_videos/bin/activate


#python3 /home/p288722/git_code/scd_videos_first_revision/dataset/frames/1_frame_extractor.py --input_dir="/data/p288722/datasets/VISION/dataset" --output_dir="/scratch/p288722/datasets/vision/all_frames" --device_id=${SLURM_ARRAY_TASK_ID}
python3 /home/p288722/git_code/scd_videos_first_revision/_sota/bm3d/create_noise_dataset.py
