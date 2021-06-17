#!/bin/bash

#SBATCH --job-name=n_net
#SBATCH --time=2:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:v100:1

module load cuDNN/7.6.4.38-gcccuda-2019b

source /data/p288722/python_venv/scd_videos/bin/activate

# baseline
# python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=64 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/baseline/models/ConvNet_FC2x1024_28D"

# center crop
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/28_dev_center_crop" --batch_size=128 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/480x800_crops/models/ConvNet_center_crop"

# homogeneous crop
python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/28_dev_homogeneous_crop" --batch_size=128 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/480x800_crops/models/ConvNet_homo_crop"
#sbatch /home/p288722/git_code/scd_videos_tf/constrained_net/bash/frames/convnet/predict_convnet_FC2x1024.sh