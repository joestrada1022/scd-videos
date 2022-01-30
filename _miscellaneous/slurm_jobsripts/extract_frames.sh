#!/bin/bash

#SBATCH --job-name=I-frames
#SBATCH --time=04:00:00
#SBATCH --mem=4GB
# --partition=short
#SBATCH --array=0-12,16-27

module purge
#module load MATLAB/2020b
module load OpenBLAS/0.3.15-GCC-10.3.0

#export PYTHONUNBUFFERED=1
#export PATH=${PATH}:/home/p288722/build/lib/matlab

source /data/p288722/python_venv/scd_videos_first_revision/bin/activate

#python3 /home/p288722/git_code/scd_videos_first_revision/dataset/frames/1_frame_extractor.py --input_dir="/data/p288722/datasets/VISION/dataset" --output_dir="/scratch/p288722/datasets/vision/all_frames" --device_id=${SLURM_ARRAY_TASK_ID}
python3 /home/p288722/git_code/scd_videos_first_revision/dataset/frames/extract_I_frames.py --input_dir="/data/p288722/datasets/VISION/dataset" --output_dir="/scratch/p288722/datasets/vision/all_I_frames" --device_id="${SLURM_ARRAY_TASK_ID}"
#/data/p288722/python_venv/scd_videos/bin/python -u /home/p288722/git_code/scd_videos_first_revision/_miscellaneous/iqm/compute_iqms.py --metric_type="brisque"

#python3 /home/p288722/git_code/scd_videos_first_revision/_sota/bm3d/create_noise_dataset.py

#export PYTHONPATH=${PYTHONPATH}:/home/p288722/git_code/scd_videos_first_revision
#python3 /home/p288722/git_code/scd_videos_first_revision/dataset/frames/utils_N_frames_to_all_frames.py
#python3 /home/p288722/git_code/scd_videos_first_revision/dataset/frames/extract_I_frames.py


