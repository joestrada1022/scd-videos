#!/bin/bash

#SBATCH --job-name=256
#SBATCH --time=2:00:00
#SBATCH --mem=3200
#SBATCH --partition=gpu
#--reservation=infsys
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12
# --array=0-9

ssh pg-lab01
module load cuDNN/7.6.4.38-gcccuda-2019b
source /data/p288722/python_venv/scd_videos/bin/activate


#python3 /home/p288722/git_code/scd-videos/_miscellaneous/exp_pos_const_layer/plot_correlations.py

#python3 /home/p288722/git_code/scd-videos/train_mobile_net.py --dataset="/scratch/p288722/datasets/vision/2_devices/bal_50_frames" --epochs=20 --batch_size=2 --constrained=1  --gpu_id=0 --model_name="ConstNet_guru" --const_type='guru' --global_results_dir="/scratch/p288722/runtime_data/scd-videos/dev_const_layer/50_frames_2d/mobile_net"
#python3 /home/p288722/git_code/scd-videos/validate_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/2_devices/bal_50_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="50_frames_val" --input_dir="/scratch/p288722/runtime_data/scd-videos/dev_const_layer/50_frames_2d/mobile_net/models/ConstNet_guru"
#python3 /home/p288722/git_code/scd-videos/_miscellaneous/plots/validation_plots.py
#python3 /home/p288722/git_code/scd-videos/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/2_devices/bal_50_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="50_frames" --input_dir="/scratch/p288722/runtime_data/scd-videos/dev_const_layer/50_frames_2d_pred/mobile_net/models/ConstNet_guru"
#
#python3 /home/p288722/git_code/scd-videos/train_mobile_net.py --dataset="/scratch/p288722/datasets/vision/2_devices/bal_50_frames" --epochs=20 --batch_size=64 --constrained=1  --gpu_id=0 --model_name="ConstNet_bayar" --global_results_dir="/scratch/p288722/runtime_data/scd-videos/dev_const_layer/50_frames_2d/mobile_net"
#python3 /home/p288722/git_code/scd-videos/validate_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/2_devices/bal_50_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="50_frames_val" --input_dir="/scratch/p288722/runtime_data/scd-videos/dev_const_layer/50_frames_2d/mobile_net/models/ConstNet_bayar"
#python3 /home/p288722/git_code/scd-videos/_miscellaneous/plots/validation_plots.py
#python3 /home/p288722/git_code/scd-videos/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/2_devices/bal_50_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="50_frames" --input_dir="/scratch/p288722/runtime_data/scd-videos/dev_const_layer/50_frames_2d_pred/mobile_net/models/ConstNet_bayar"
#
#python3 /home/p288722/git_code/scd-videos/train_mobile_net.py --dataset="/scratch/p288722/datasets/vision/2_devices/bal_50_frames" --epochs=20 --batch_size=64 --constrained=1  --gpu_id=0 --model_name="ConstNet_bug" --global_results_dir="/scratch/p288722/runtime_data/scd-videos/dev_const_layer/50_frames_2d/mobile_net"
#python3 /home/p288722/git_code/scd-videos/validate_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/2_devices/bal_50_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="50_frames_val" --input_dir="/scratch/p288722/runtime_data/scd-videos/dev_const_layer/50_frames_2d/mobile_net/models/ConstNet_bug"
#python3 /home/p288722/git_code/scd-videos/_miscellaneous/plots/validation_plots.py
#python3 /home/p288722/git_code/scd-videos/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/2_devices/bal_50_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="50_frames" --input_dir="/scratch/p288722/runtime_data/scd-videos/dev_const_layer/50_frames_2d_pred/mobile_net/models/ConstNet_bug"
#
#python3 /home/p288722/git_code/scd-videos/train_mobile_net.py --dataset="/scratch/p288722/datasets/vision/2_devices/bal_50_frames" --epochs=20 --batch_size=64 --constrained=0  --gpu_id=0 --model_name="ConvNet" --global_results_dir="/scratch/p288722/runtime_data/scd-videos/dev_const_layer/50_frames_2d/mobile_net"
#python3 /home/p288722/git_code/scd-videos/validate_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/2_devices/bal_50_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="50_frames_val" --input_dir="/scratch/p288722/runtime_data/scd-videos/dev_const_layer/50_frames_2d/mobile_net/models/ConvNet"
#python3 /home/p288722/git_code/scd-videos/_miscellaneous/plots/validation_plots.py
#python3 /home/p288722/git_code/scd-videos/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/2_devices/bal_50_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="50_frames" --input_dir="/scratch/p288722/runtime_data/scd-videos/dev_const_layer/50_frames_2d_pred/mobile_net/models/ConvNet"


python3 /home/p288722/git_code/scd-videos/train_mobile_net.py --dataset="/scratch/p288722/datasets/vision/8_devices/bal_50_frames_1ppf" --epochs=20 --batch_size=64 --constrained=1  --height=128 --width=128 --gpu_id=1 --model_name="ConstNet_guru_1ppf" --const_type="guru" --global_results_dir="/scratch/p288722/runtime_data/scd-videos/dev_combine_layer_patches/50_frames_8d_64/mobile_net"
python3 /home/p288722/git_code/scd-videos/validate_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/8_devices/bal_50_frames_1ppf" --batch_size=64 --constrained=1 --height=128 --width=128 --gpu_id=1 --suffix="50_frames_val" --input_dir="/scratch/p288722/runtime_data/scd-videos/dev_combine_layer_patches/50_frames_8d_64/mobile_net/models/ConstNet_guru_1ppf"
python3 /home/p288722/git_code/scd-videos/_miscellaneous/plots/validation_plots.py
python3 /home/p288722/git_code/scd-videos/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/8_devices/bal_50_frames_1ppf" --batch_size=64 --constrained=1 --height=128 --width=128 --gpu_id=1 --suffix="50_frames" --input_dir="/scratch/p288722/runtime_data/scd-videos/dev_combine_layer_patches/50_frames_8d_64_pred/mobile_net/models/ConstNet_guru_1ppf"

python3 /home/p288722/git_code/scd-videos/train_mobile_net.py --dataset="/scratch/p288722/datasets/vision/8_devices/bal_50_frames_1ppf" --epochs=20 --batch_size=64 --constrained=0  --height=128 --width=128 --gpu_id=1 --model_name="ConvNet_1ppf" --const_type="guru" --global_results_dir="/scratch/p288722/runtime_data/scd-videos/dev_combine_layer_patches/50_frames_8d_64/mobile_net"
python3 /home/p288722/git_code/scd-videos/validate_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/8_devices/bal_50_frames_1ppf" --batch_size=64 --constrained=0 --height=128 --width=128 --gpu_id=1 --suffix="50_frames_val" --input_dir="/scratch/p288722/runtime_data/scd-videos/dev_combine_layer_patches/50_frames_8d_64/mobile_net/models/ConvNet_1ppf"
python3 /home/p288722/git_code/scd-videos/_miscellaneous/plots/validation_plots.py
python3 /home/p288722/git_code/scd-videos/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/8_devices/bal_50_frames_1ppf" --batch_size=64 --constrained=0 --height=128 --width=128 --gpu_id=1 --suffix="50_frames" --input_dir="/scratch/p288722/runtime_data/scd-videos/dev_combine_layer_patches/50_frames_8d_64_pred/mobile_net/models/ConvNet_1ppf"


python3 /home/p288722/git_code/scd-videos/train_mobile_net.py --dataset="/scratch/p288722/datasets/vision/8_devices/bal_50_frames_50ppf" --epochs=20 --batch_size=64 --constrained=1  --height=128 --width=128 --gpu_id=0 --model_name="ConstNet_guru_50ppf" --const_type="guru" --global_results_dir="/scratch/p288722/runtime_data/scd-videos/dev_combine_layer_patches/50_frames_8d_64/mobile_net"
python3 /home/p288722/git_code/scd-videos/validate_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/8_devices/bal_50_frames_50ppf" --batch_size=64 --constrained=1 --height=128 --width=128 --gpu_id=0 --suffix="50_frames_val" --input_dir="/scratch/p288722/runtime_data/scd-videos/dev_combine_layer_patches/50_frames_8d_64/mobile_net/models/ConstNet_guru_50ppf"
python3 /home/p288722/git_code/scd-videos/_miscellaneous/plots/validation_plots.py
python3 /home/p288722/git_code/scd-videos/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/8_devices/bal_50_frames_50ppf" --batch_size=64 --constrained=1 --height=128 --width=128 --gpu_id=0 --suffix="50_frames" --input_dir="/scratch/p288722/runtime_data/scd-videos/dev_combine_layer_patches/50_frames_8d_64_pred/mobile_net/models/ConstNet_guru_50ppf"

python3 /home/p288722/git_code/scd-videos/train_mobile_net.py --dataset="/scratch/p288722/datasets/vision/8_devices/bal_50_frames_50ppf" --epochs=20 --batch_size=64 --constrained=0  --height=128 --width=128 --gpu_id=1 --model_name="ConvNet_50ppf" --const_type="guru" --global_results_dir="/scratch/p288722/runtime_data/scd-videos/dev_combine_layer_patches/50_frames_8d_64/mobile_net"
python3 /home/p288722/git_code/scd-videos/validate_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/8_devices/bal_50_frames_50ppf" --batch_size=64 --constrained=0 --height=128 --width=128 --gpu_id=1 --suffix="50_frames_val" --input_dir="/scratch/p288722/runtime_data/scd-videos/dev_combine_layer_patches/50_frames_8d_64/mobile_net/models/ConvNet_50ppf"
python3 /home/p288722/git_code/scd-videos/_miscellaneous/plots/validation_plots.py
python3 /home/p288722/git_code/scd-videos/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/8_devices/bal_50_frames_50ppf" --batch_size=64 --constrained=0 --height=128 --width=128 --gpu_id=1 --suffix="50_frames" --input_dir="/scratch/p288722/runtime_data/scd-videos/dev_combine_layer_patches/50_frames_8d_64_pred/mobile_net/models/ConvNet_50ppf"


python3 /home/p288722/git_code/scd-videos/train_mobile_net.py --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --epochs=20 --batch_size=64 --constrained=0  --height=480 --width=800 --gpu_id=0 --model_name="ConvNet" --const_type="guru" --global_results_dir="/scratch/p288722/runtime_data/scd-videos/dev_combine_layer/200_frames_28d_64/mobile_net"
python3 /home/p288722/git_code/scd-videos/validate_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=64 --constrained=0 --height=480 --width=800 --gpu_id=0 --suffix="200_frames_val" --input_dir="/scratch/p288722/runtime_data/scd-videos/dev_combine_layer/200_frames_28d_64/mobile_net/models/ConvNet"
python3 /home/p288722/git_code/scd-videos/_miscellaneous/plots/validation_plots.py
python3 /home/p288722/git_code/scd-videos/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=64 --constrained=0 --height=480 --width=800 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd-videos/dev_combine_layer/200_frames_28d_64_pred/mobile_net/models/ConvNet"

