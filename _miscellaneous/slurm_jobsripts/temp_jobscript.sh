#!/bin/bash
#SBATCH --job-name=temp-evaluations
#SBATCH --time=2:00:00
#SBATCH --mem=64000
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:v100:1
# --reservation=infsys
#SBATCH --cpus-per-task=12
# --dependency=afterok:22194316

# create the following directory manually
#SBATCH --chdir=/scratch/p288722/runtime_data/scd_videos_first_revision/04_glcm_training
#SBATCH --output=slurm-%j-%x.out
#SBATCH --error=slurm-%j-%x.out

#module load cuDNN/8.0.4.30-CUDA-11.1.1
#module load TensorFlow/2.5.0-fosscuda-2020b

module purge
module load OpenBLAS/0.3.15-GCC-10.3.0
module load CUDAcore/11.1.1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/p288722/softwares/cuda/lib64
source /data/p288722/python_venv/scd_videos/bin/activate

python3 /home/p288722/git_code/scd_videos_first_revision/run_evaluate.py --homo_or_not=None \
--eval_set=val \
--dataset=/scratch/p288722/datasets/vision/homo_frames_split/bal_50_frames_mean_glcm \
--batch_size=64 \
--height=480 \
--width=800 \
--gpu_id=0 \
--suffix=50_frames_val \
--input_dir=/scratch/p288722/runtime_data/scd_videos_first_revision/04_glcm_training/50_frames/mobile_net/models/MobileNet_mean


python3 /home/p288722/git_code/scd_videos_first_revision/run_evaluate.py --homo_or_not=None \
--eval_set=val \
--dataset=/scratch/p288722/datasets/vision/homo_frames_split/bal_50_frames_max_glcm \
--batch_size=64 \
--height=480 \
--width=800 \
--gpu_id=0 \
--suffix=50_frames_val \
--input_dir=/scratch/p288722/runtime_data/scd_videos_first_revision/04_glcm_training/50_frames/mobile_net/models/MobileNet_max

sbatch /home/p288722/git_code/scd_videos_first_revision/_miscellaneous/slurm_jobsripts/temp_jobscript.sh