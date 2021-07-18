#!/bin/bash

#SBATCH --job-name=i8d_cd
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --partition=lab
#SBATCH --reservation=infsys
# --gres=gpu:v100:1
#SBATCH --cpus-per-task=12

#ssh pg-lab01
module load cuDNN/7.6.4.38-gcccuda-2019b
source /data/p288722/python_venv/scd_videos/bin/activate

#python3 /home/p288722/git_code/scd-videos/train_mobile_net.py --dataset="/scratch/p288722/datasets/vision/8_devices/bal_50_frames" --epochs=20 --batch_size=64 --constrained=1 --height=480 --width=800 --gpu_id=0 --model_name="h0_lab_ConstNet_derrick" --const_type="derrick" --global_results_dir="/scratch/p288722/runtime_data/scd-videos/no_frame_selection/50_frames_8d_64/mobile_net"
#python3 /home/p288722/git_code/scd-videos/validate_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/8_devices/bal_50_frames" --batch_size=64 --constrained=1 --height=480 --width=800 --gpu_id=0 --suffix="50_frames_val" --input_dir="/scratch/p288722/runtime_data/scd-videos/no_frame_selection/50_frames_8d_64/mobile_net/models/h0_lab_ConstNet_derrick"
#python3 /home/p288722/git_code/scd-videos/_miscellaneous/plots/validation_plots.py
#python3 /home/p288722/git_code/scd-videos/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/8_devices/bal_50_frames" --batch_size=64 --constrained=1 --height=480 --width=800 --gpu_id=0 --suffix="50_frames" --input_dir="/scratch/p288722/runtime_data/scd-videos/no_frame_selection/50_frames_8d_64_pred/mobile_net/models/h0_lab_ConstNet_derrick"

#python3 /home/p288722/git_code/scd-videos/train_mobile_net.py --dataset="/scratch/p288722/datasets/vision/8_devices/bal_50_frames" --epochs=20 --batch_size=64 --constrained=0 --height=480 --width=800 --gpu_id=0 --model_name="h0_lab_ConvNet" --global_results_dir="/scratch/p288722/runtime_data/scd-videos/no_frame_selection/50_frames_8d_64/mobile_net"
#python3 /home/p288722/git_code/scd-videos/validate_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/8_devices/bal_50_frames" --batch_size=64 --constrained=0 --height=480 --width=800 --gpu_id=0 --suffix="50_frames_val" --input_dir="/scratch/p288722/runtime_data/scd-videos/no_frame_selection/50_frames_8d_64/mobile_net/models/h0_lab_ConvNet"
#python3 /home/p288722/git_code/scd-videos/_miscellaneous/plots/validation_plots.py
#python3 /home/p288722/git_code/scd-videos/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/8_devices/bal_50_frames" --batch_size=64 --constrained=0 --height=480 --width=800 --gpu_id=0 --suffix="50_frames" --input_dir="/scratch/p288722/runtime_data/scd-videos/no_frame_selection/50_frames_8d_64_pred/mobile_net/models/h0_lab_ConvNet"

#python3 /home/p288722/git_code/scd-videos/train_mobile_net.py --dataset="/scratch/p288722/datasets/vision/8_devices/bal_all_I_frames" --epochs=20 --batch_size=64 --constrained=1 --height=480 --width=800 --gpu_id=0 --model_name="h0_lab_ConstNet_derrick" --const_type="derrick" --global_results_dir="/scratch/p288722/runtime_data/scd-videos/i_frames/all_frames_8d_64/mobile_net"
#python3 /home/p288722/git_code/scd-videos/validate_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/8_devices/bal_all_I_frames" --batch_size=64 --constrained=1 --height=480 --width=800 --gpu_id=0 --suffix="all_frames_val" --input_dir="/scratch/p288722/runtime_data/scd-videos/i_frames/all_frames_8d_64/mobile_net/models/h0_lab_ConstNet_derrick"
#python3 /home/p288722/git_code/scd-videos/_miscellaneous/plots/validation_plots.py
#python3 /home/p288722/git_code/scd-videos/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/8_devices/bal_all_I_frames" --batch_size=64 --constrained=1 --height=480 --width=800 --gpu_id=0 --suffix="all_frames" --input_dir="/scratch/p288722/runtime_data/scd-videos/i_frames/all_frames_8d_64_pred/mobile_net/models/h0_lab_ConstNet_derrick"

#python3 /home/p288722/git_code/scd-videos/train_mobile_net.py --dataset="/scratch/p288722/datasets/vision/8_devices/bal_all_I_frames" --epochs=20 --batch_size=64 --constrained=0 --height=480 --width=800 --gpu_id=0 --model_name="h0_lab_ConvNet" --global_results_dir="/scratch/p288722/runtime_data/scd-videos/i_frames/all_frames_8d_64/mobile_net"
#python3 /home/p288722/git_code/scd-videos/validate_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/8_devices/bal_all_I_frames" --batch_size=64 --constrained=0 --height=480 --width=800 --gpu_id=0 --suffix="all_frames_val" --input_dir="/scratch/p288722/runtime_data/scd-videos/i_frames/all_frames_8d_64/mobile_net/models/h0_lab_ConvNet"
#python3 /home/p288722/git_code/scd-videos/_miscellaneous/plots/validation_plots.py
#python3 /home/p288722/git_code/scd-videos/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/8_devices/bal_all_I_frames" --batch_size=64 --constrained=0 --height=480 --width=800 --gpu_id=0 --suffix="all_frames" --input_dir="/scratch/p288722/runtime_data/scd-videos/i_frames/all_frames_8d_64_pred/mobile_net/models/h0_lab_ConvNet"
