#!/bin/bash

#SBATCH --job-name=cnet
#SBATCH --time=2:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpushort
#--reservation=infsys
#SBATCH --gres=gpu:v100:1

module load cuDNN/7.6.4.38-gcccuda-2019b

source /data/p288722/python_venv/scd_videos/bin/activate

# baseline
python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --epochs=45 --batch_size=128 --constrained=1 --model_name="ConstNet_native" --category="native" --model_path="/scratch/p288722/runtime_data/scd_videos_tf/extension/models/ConstNet_native/fm-e00019.h5"
# To continue training from a previous model add parameter --model_path with the value pointing to the particular *.h5 model

# center crop
# python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/28_dev_center_crop" --epochs=45 --batch_size=32 --constrained=1 --model_name="ConstNet_center_crop" --gpu_id=0

# homogeneous crop
# python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/28_dev_homogeneous_crop" --epochs=45 --batch_size=32 --constrained=1 --model_name="ConstNet_homo_crop" --gpu_id=0
