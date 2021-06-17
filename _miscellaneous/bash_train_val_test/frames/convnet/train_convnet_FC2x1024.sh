#!/bin/bash

#SBATCH --job-name=n_net
#SBATCH --time=2:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpushort
#--reservation=infsys
#SBATCH --gres=gpu:v100:1

module load cuDNN/7.6.4.38-gcccuda-2019b

source /data/p288722/python_venv/scd_videos/bin/activate

# baseline
python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/28_dev_homogeneous_crop" --epochs=45 --batch_size=128 --constrained=0 --model_name="ConvNet_native" --gpu_id=0 --category="native" --model_path="/scratch/p288722/runtime_data/scd_videos_tf/extension/models/ConvNet_native/fm-e00027.h5"

# center crop
#python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/28_dev_center_crop" --epochs=45 --batch_size=32 --constrained=0 --model_name="ConvNet_center_crop" --gpu_id=0
# To continue training from a previous model add parameter --model_path with the value pointing to the particular *.h5 model

## homo crop
#python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/28_dev_homogeneous_crop" --epochs=45 --batch_size=32 --constrained=0 --model_name="ConvNet_homo_crop" --gpu_id=0