#!/bin/bash

#SBATCH --job-name=mob
#SBATCH --time=2:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpushort
# --reservation=infsys
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=12

#ssh pg-lab01
module load cuDNN/7.6.4.38-gcccuda-2019b
source /data/p288722/python_venv/scd_videos/bin/activate

## native
#python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --epochs=45 --batch_size=128 --constrained=1 --model_name="ConstNet_native" --category="native" --global_results_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=1 --category="native" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConstNet_native"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=1 --category="whatsapp" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConstNet_native"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=1 --category="youtube" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConstNet_native"

## whatsapp
#python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --epochs=45 --batch_size=128 --constrained=1 --model_name="ConstNet_whatsapp" --category="whatsapp" --global_results_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=1 --category="native" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConstNet_whatsapp"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=1 --category="whatsapp" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConstNet_whatsapp"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=1 --category="youtube" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConstNet_whatsapp"

## youtube
#python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --epochs=45 --batch_size=128 --constrained=1 --model_name="ConstNet_youtube" --category="youtube" --global_results_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=1 --category="native" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConstNet_youtube"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=1 --category="whatsapp" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConstNet_youtube"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=1 --category="youtube" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConstNet_youtube"

## center crop
#python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/center_crop_128x128" --epochs=45 --batch_size=128 --constrained=1 --height=128 --width=128 --model_name="ConstNet_cc" --global_results_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/center_crop_128x128" --batch_size=128 --constrained=1 --height=128 --width=128 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128/models/ConstNet_cc"

## random crop
#python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/rand_crop_128x128_5" --epochs=45 --batch_size=128 --constrained=1 --height=128 --width=128 --model_name="ConstNet_5_rc" --global_results_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/rand_crop_128x128_5" --batch_size=128 --constrained=1 --height=128 --width=128 --gpu_id=0 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128/models/ConstNet_5_rc"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/rand_crop_128x128_10" --batch_size=128 --constrained=1 --height=128 --width=128 --gpu_id=0 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128/models/ConstNet_5_rc"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/rand_crop_128x128_20" --batch_size=128 --constrained=1 --height=128 --width=128 --gpu_id=0 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128/models/ConstNet_5_rc"

## homogeneous crop
#python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/homo_crop_128x128_5" --epochs=45 --batch_size=128 --constrained=1 --height=128 --width=128 --model_name="ConstNet_5_hc" --global_results_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/homo_crop_128x128_5" --batch_size=128 --constrained=1 --height=128 --width=128 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128/models/ConstNet_5_hc"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/homo_crop_128x128_10" --batch_size=128 --constrained=1 --height=128 --width=128 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128/models/ConstNet_5_hc"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/homo_crop_128x128_20" --batch_size=128 --constrained=1 --height=128 --width=128 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128/models/ConstNet_5_hc"


#python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --epochs=32 --batch_size=32 --constrained=1 --model_name="ConstNet" --global_results_dir="/scratch/p288722/runtime_data/scd_videos_tf/new/misl_net" --gpu_id=1

#ssh pg-lab01
#module load cuDNN/7.6.4.38-gcccuda-2019b
#source /data/p288722/python_venv/scd_videos/bin/activate
#python3 /home/p288722/git_code/scd_videos_tf/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new_predictions/mobile_net/models/ConstNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new_predictions/mobile_net/models/ConvNet"

#python3 /home/p288722/git_code/scd_videos_tf/train_efficient_net.py --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --epochs=32 --batch_size=32 --constrained=0 --model_name="ConvNet" --global_results_dir="/scratch/p288722/runtime_data/scd_videos_tf/new/eff_net_baseline" --gpu_id=1 --model_path="/scratch/p288722/runtime_data/scd_videos_tf/new/eff_net_baseline/models/ConvNet/fm-e00011.h5"
#python3 /home/p288722/git_code/scd_videos_tf/train_efficient_net.py --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --epochs=32 --batch_size=32 --constrained=1 --model_name="ConstNet" --global_results_dir="/scratch/p288722/runtime_data/scd_videos_tf/new/eff_net_baseline" --gpu_id=0
#python3 /home/p288722/git_code/scd_videos_tf/train_mobile_net.py --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --epochs=32 --batch_size=64 --constrained=0 --model_name="ConvNet" --global_results_dir="/scratch/p288722/runtime_data/scd_videos_tf/new/mobile_net_baseline" --gpu_id=0 --model_path="/scratch/p288722/runtime_data/scd_videos_tf/new/mobile_net_baseline/models/ConvNet/fm-e00026.h5"
#python3 /home/p288722/git_code/scd_videos_tf/validate_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="200_frames_val" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new/mobile_net_baseline/models/ConvNet"
#python3 /home/p288722/git_code/scd_videos_tf/validate_misl_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=1 --gpu_id=1 --suffix="200_frames_val" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new/misl_net/models/ConstNet"
#python3 /home/p288722/git_code/scd_videos_tf/validate_misl_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=0 --gpu_id=1 --suffix="200_frames_val" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new/misl_net/models/ConvNet"
#python3 /home/p288722/git_code/scd_videos_tf/validate_efficient_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="200_frames_val" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new/eff_net_baseline/models/ConstNet"

#python3 /home/p288722/git_code/scd_videos_tf/predict_misl_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new_predictions/misl_net/models/ConstNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_constrained_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new_predictions/misl_net/models/ConvNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_constrained_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new_predictions/misl_net_baseline/models/ConstNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_constrained_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new_predictions/misl_net_baseline/models/ConvNet"

#python3 /home/p288722/git_code/scd_videos_tf/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new_predictions/mobile_net/models/ConstNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new_predictions/mobile_net/models/ConvNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new_predictions/mobile_net_baseline/models/ConstNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new_predictions/mobile_net_baseline/models/ConvNet"


python3 /home/p288722/git_code/scd_videos_tf/validate_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="200_frames_val" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new/mobile_net_baseline1/models/ConvNet"
python3 /home/p288722/git_code/scd_videos_tf/validate_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="200_frames_val" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new/mobile_net_baseline1/models/ConstNet"
python3 /home/p288722/git_code/scd_videos_tf/validate_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=0 --gpu_id=1 --suffix="200_frames_val" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new/mobile_net_baseline2/models/ConvNet"
python3 /home/p288722/git_code/scd_videos_tf/validate_mobile_net.py  --dataset="/scratch/p288722/datasets/vision/bal_200_frames" --batch_size=128 --constrained=1 --gpu_id=1 --suffix="200_frames_val" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/new/mobile_net_baseline2/models/ConstNet"
