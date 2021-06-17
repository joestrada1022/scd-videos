#!/bin/bash

#SBATCH --job-name=bal_ds
#SBATCH --time=4:00:00
#SBATCH --mem=1GB
#SBATCH --array=0-27

module load cuDNN/7.6.4.38-gcccuda-2019b
source /data/p288722/python_venv/scd_videos/bin/activate

# echo running_the_flow
#python3 /home/p288722/git_code/scd_videos_tf/dataset/frames/frame_extractor/frame_extractor.py --input_dir="/data/p288722/VISION/dataset" --output_dir="/scratch/p288722/datasets/vision/all_frames" --device_id=${SLURM_ARRAY_TASK_ID}
python3 /home/p288722/git_code/scd_videos_tf/dataset/frames/from_derrick.py --device_id=${SLURM_ARRAY_TASK_ID}  --num_frames=200
# echo completed_the_job