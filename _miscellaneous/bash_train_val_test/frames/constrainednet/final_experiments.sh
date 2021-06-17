#!/bin/bash

#SBATCH --job-name=time
#SBATCH --time=2:00:00
#SBATCH --mem=32000
#SBATCH --partition=lab
#SBATCH --reservation=infsys
# --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
# --array=0-9

ssh pg-lab01
module load cuDNN/7.6.4.38-gcccuda-2019b
source /data/p288722/python_venv/scd_videos/bin/activate
#python3 /home/p288722/git_code/scd_videos_tf/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_50_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="50_frames_dev" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/50_frames_pred_bal_val/mobile_net_1/models/ConvNet"

#python3 /home/p288722/git_code/scd_videos_tf/train_misl_net.py --dataset="/scratch/p288722/datasets/vision/bal_50_frames" --epochs=20 --batch_size=64 --constrained=1 --model_name="ConstNet" --global_results_dir="/scratch/p288722/runtime_data/scd_videos_tf/50_frames/misl_net" --gpu_id=0
#python3 /home/p288722/git_code/scd_videos_tf/validate_misl_net.py  --dataset="/scratch/p288722/datasets/vision/bal_50_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="50_frames_val" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/50_frames/misl_net/models/ConstNet"
#python3 /home/p288722/git_code/scd_videos_tf/miscelaneous/plots/validation_plots.py
#python3 /home/p288722/git_code/scd_videos_tf/predict_misl_net.py  --dataset="/scratch/p288722/datasets/vision/bal_50_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="50_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/50_frames_pred/misl_net/models/ConstNet"

#python3 /home/p288722/git_code/scd_videos_tf/miscelaneous/plots/validation_plots.py
#python3 /home/p288722/git_code/scd_videos_tf/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/50_frames_pred/mobile_net_lr_0.1_3/models/ConstNet"

#python3 /home/p288722/git_code/scd_videos_tf/train_mobile_net.py --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --epochs=20 --batch_size=64 --constrained=0 --model_name="ConvNet" --global_results_dir="/scratch/p288722/runtime_data/scd_videos_tf/200_frames/mobile_net_3" --gpu_id=0
#python3 /home/p288722/git_code/scd_videos_tf/validate_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="200_frames_val" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/200_frames/mobile_net_3/models/ConvNet"
#python3 /home/p288722/git_code/scd_videos_tf/miscelaneous/plots/validation_plots.py
#python3 /home/p288722/git_code/scd_videos_tf/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new_pred/mobile_net_3/models/ConvNet"

#python3 /home/p288722/git_code/scd_videos_tf/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_all_frames_fragments/${SLURM_ARRAY_TASK_ID}" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="all_frames_${SLURM_ARRAY_TASK_ID}" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/200_frames_pred/mobile_net_2/models/ConvNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_all_frames_fragments/7" --batch_size=128 --constrained=0 --gpu_id=1 --suffix="all_frames_7" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/200_frames_pred/mobile_net_2/models/ConvNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_all_frames_fragments/8" --batch_size=128 --constrained=0 --gpu_id=1 --suffix="all_frames_8" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/200_frames_pred/mobile_net_2/models/ConvNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_all_frames_fragments/9" --batch_size=128 --constrained=0 --gpu_id=1 --suffix="all_frames_9" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/200_frames_pred/mobile_net_2/models/ConvNet"

python3 /home/p288722/git_code/scd_videos_tf/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_50_frames" --batch_size=1 --constrained=0 --gpu_id=0 --suffix="000000001_frame_time_warmup" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/50_frames_pred/mobile_net_2/models/ConvNet"
python3 /home/p288722/git_code/scd_videos_tf/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_50_frames" --batch_size=1 --constrained=0 --gpu_id=0 --suffix="000000001_frame_time" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/50_frames_pred/mobile_net_2/models/ConvNet"
python3 /home/p288722/git_code/scd_videos_tf/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_50_frames" --batch_size=1 --constrained=1 --gpu_id=0 --suffix="000000001_frame_time" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/50_frames_pred/mobile_net_2/models/ConstNet"
python3 /home/p288722/git_code/scd_videos_tf/predict_efficient_net.py  --dataset="/scratch/p288722/datasets/vision/bal_50_frames" --batch_size=1 --constrained=1 --gpu_id=0 --suffix="000000001_frame_time" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/50_frames_pred/efficient_net/models/ConstNet"

python3 /home/p288722/git_code/scd_videos_tf/predict_misl_net.py  --dataset="/scratch/p288722/datasets/vision/bal_50_frames" --batch_size=1 --constrained=0 --gpu_id=1 --suffix="000000001_frame_time_warmup" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/50_frames_pred/misl_net_2/models/ConvNet"
python3 /home/p288722/git_code/scd_videos_tf/predict_misl_net.py  --dataset="/scratch/p288722/datasets/vision/bal_50_frames" --batch_size=1 --constrained=0 --gpu_id=1 --suffix="000000001_frame_time" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/50_frames_pred/misl_net_2/models/ConvNet"
python3 /home/p288722/git_code/scd_videos_tf/predict_misl_net.py  --dataset="/scratch/p288722/datasets/vision/bal_50_frames" --batch_size=1 --constrained=1 --gpu_id=1 --suffix="000000001_frame_time" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/50_frames_pred/misl_net_2/models/ConstNet"
python3 /home/p288722/git_code/scd_videos_tf/predict_efficient_net.py  --dataset="/scratch/p288722/datasets/vision/bal_50_frames" --batch_size=1 --constrained=0 --gpu_id=1 --suffix="000000001_frame_time" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/50_frames_pred/efficient_net/models/ConvNet"


python3 /home/p288722/git_code/scd_videos_tf/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_50_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="mixed_50_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/50_frames_pred/mobile_net_1/models/ConvNet"
