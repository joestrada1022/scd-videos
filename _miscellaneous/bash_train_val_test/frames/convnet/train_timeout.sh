#!/bin/bash

#SBATCH --job-name=hc_unnet
#SBATCH --time=2:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpushort
# --reservation=infsys
# --gres=gpu:v100:1

module load cuDNN/7.6.4.38-gcccuda-2019b

source /data/p288722/python_venv/scd_videos/bin/activate

# center crop
# python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/28_dev_center_crop" --epochs=30 --batch_size=128 --constrained=0 --model_name="ConvNet_center_crop" --center_crop=1 --gpu_id=0 --model_path="/scratch/p288722/runtime_data/scd_videos_tf/480x800_crops/models/ConvNet_center_crop/fm-e00004.h5"
# To continue training from a previous model add parameter --model_path with the value pointing to the particular *.h5 model

# homo crop
timeout 118m python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/28_dev_homogeneous_crop" --epochs=30 --batch_size=128 --constrained=0 --model_name="ConvNet_homo_crop" --gpu_id=0
python3 /home/p288722/git_code/scd_videos_tf/constrained_net/bash/frames/convnet/update_bash.py --models_path="/scratch/p288722/runtime_data/scd_videos_tf/480x800_crops/models/ConvNet_homo_crop" --bash_path="/home/p288722/git_code/scd_videos_tf/constrained_net/bash/frames/convnet/train_timeout.sh"
sbatch /home/p288722/git_code/scd_videos_tf/constrained_net/bash/frames/convnet/train_timeout.sh