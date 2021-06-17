#!/bin/bash

#SBATCH --job-name=up_pr
#SBATCH --time=2:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpushort
# --reservation=infsys
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12

#ssh pg-lab01
module load cuDNN/7.6.4.38-gcccuda-2019b
source /data/p288722/python_venv/scd_videos/bin/activate

#python3 /home/p288722/git_code/scd_videos_tf/predict_misl_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new_pred/misl_net/models/ConstNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_misl_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new_pred/misl_net/models/ConvNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new_pred/mobile_net_baseline/models/ConstNet"
python3 /home/p288722/git_code/scd_videos_tf/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new_pred/mobile_net_baseline/models/ConvNet"
python3 /home/p288722/git_code/scd_videos_tf/predict_efficient_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new_pred/eff_net_baseline/models/ConstNet"
python3 /home/p288722/git_code/scd_videos_tf/predict_efficient_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new_pred/eff_net_baseline/models/ConvNet"
