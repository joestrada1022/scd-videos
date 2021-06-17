#!/bin/bash

#SBATCH --job-name=p_n_cnet
#SBATCH --time=2:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:v100:1

module load cuDNN/7.6.4.38-gcccuda-2019b

source /data/p288722/python_venv/scd_videos/bin/activate

# baseline
python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=1 --category="native" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/extension/models/ConstNet_native"
python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=1 --category="whatsapp" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/extension/models/ConstNet_native"
python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=1 --category="youtube" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/extension/models/ConstNet_native"


# center crop
# timeout 119m python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/28_dev_center_crop" --batch_size=128 --constrained=1 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/480x800_crops/models/ConstNet_center_crop"

# homogeneous crop
#timeout 119m python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/28_dev_homogeneous_crop" --batch_size=128 --constrained=1 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/480x800_crops/models/ConstNet_homo_crop"
#sbatch /home/p288722/git_code/scd_videos_tf/constrained_net/bash/frames/constrainednet/predict_constrainednet_FC2x1024.sh