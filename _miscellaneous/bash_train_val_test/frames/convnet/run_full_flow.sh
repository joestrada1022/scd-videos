#!/bin/bash

#SBATCH --job-name=conv
#SBATCH --time=12:00:00
#SBATCH --mem=32000
#SBATCH --partition=gpu
# --reservation=infsys
#SBATCH --gres=gpu:v100:1
# --array=0-9

module load cuDNN/7.6.4.38-gcccuda-2019b
source /data/p288722/python_venv/scd_videos/bin/activate

#echo "start copying data"
#cp -r "/scratch/p288722/datasets/VISION/bal_28_devices_all_frames" "/nvme/p288722/"
#echo "finish copying data"

## native
#python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --epochs=45 --batch_size=128 --constrained=0 --model_name="ConvNet_native" --gpu_id=0 --category="native" --global_results_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=0 --category="native" --gpu_id=0 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConvNet_native"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=0 --category="whatsapp" --gpu_id=0 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConvNet_native"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=0 --category="youtube" --gpu_id=0 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConvNet_native"

## whatsapp
#python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --epochs=45 --batch_size=128 --constrained=0 --model_name="ConvNet_whatsapp" --gpu_id=0 --category="whatsapp" --global_results_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=0 --category="native" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConvNet_whatsapp"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=0 --category="whatsapp" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConvNet_whatsapp"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=0 --category="youtube" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConvNet_whatsapp"

## youtube
#python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --epochs=45 --batch_size=128 --constrained=0 --model_name="ConvNet_youtube" --gpu_id=0 --category="youtube" --global_results_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=0 --category="native" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConvNet_youtube"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=0 --category="whatsapp" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConvNet_youtube"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=0 --category="youtube" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConvNet_youtube"

## center crop
#python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/center_crop_128x128" --epochs=45 --batch_size=128 --constrained=0 --height=128 --width=128 --model_name="ConvNet_cc" --global_results_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/center_crop_128x128" --batch_size=128 --constrained=0 --height=128 --width=128 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128/models/ConvNet_cc"

## random crop
#python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/rand_crop_128x128_5" --epochs=45 --batch_size=128 --constrained=0 --height=128 --width=128 --model_name="ConvNet_5_rc" --global_results_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/rand_crop_128x128_5" --batch_size=128 --constrained=0 --height=128 --width=128 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128/models/ConvNet_5_rc"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/rand_crop_128x128_10" --batch_size=128 --constrained=0 --height=128 --width=128 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128/models/ConvNet_5_rc"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/rand_crop_128x128_20" --batch_size=128 --constrained=0 --height=128 --width=128 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128/models/ConvNet_5_rc"

## homogeneous crop
#python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/homo_crop_128x128_5" --epochs=45 --batch_size=128 --constrained=0 --height=128 --width=128 --model_name="ConvNet_5_hc" --global_results_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/homo_crop_128x128_5" --batch_size=128 --constrained=0 --height=128 --width=128 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128/models/ConvNet_5_hc"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/homo_crop_128x128_10" --batch_size=128 --constrained=0 --height=128 --width=128 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128/models/ConvNet_5_hc"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/homo_crop_128x128_20" --batch_size=128 --constrained=0 --height=128 --width=128 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_crops_128/models/ConvNet_5_hc"


# JOBS FOR THE LAB PARTITION

# native
#python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/nvme/p288722/bal_28_devices_derrick" --epochs=1 --batch_size=128 --constrained=0 --model_name="ConvNet_native_bs_128" --gpu_id=0 --category="native"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/nvme/p288722/bal_28_devices_derrick" --batch_size=128 --constrained=0 --category="native" --gpu_id=0 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConvNet_native"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/nvme/p288722/bal_28_devices_derrick" --batch_size=128 --constrained=0 --category="whatsapp" --gpu_id=0 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConvNet_native"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/nvme/p288722/bal_28_devices_derrick" --batch_size=128 --constrained=0 --category="youtube" --gpu_id=0 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConvNet_native"

### whatsapp
#python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --epochs=45 --batch_size=128 --constrained=0 --model_name="ConvNet_whatsapp" --gpu_id=1 --category="whatsapp"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=0 --category="native" --gpu_id=1 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConvNet_whatsapp"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=0 --category="whatsapp" --gpu_id=1 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConvNet_whatsapp"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=128 --constrained=0 --category="youtube" --gpu_id=1 --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/ext_bs_128/models/ConvNet_whatsapp"


#Un-constrained
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28D_all_frames/${SLURM_ARRAY_TASK_ID}" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="all_frames${SLURM_ARRAY_TASK_ID}" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConvNet"

#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_all_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="all_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConvNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_all_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="800_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConvNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_all_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="400_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConvNet"

#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/nvme/p288722/bal_28_devices_all_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConvNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/nvme/p288722/bal_28_devices_all_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="100_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConvNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/nvme/p288722/bal_28_devices_all_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="50_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConvNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/nvme/p288722/bal_28_devices_all_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="20_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConvNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/nvme/p288722/bal_28_devices_all_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="10_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConvNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/nvme/p288722/bal_28_devices_all_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="5_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConvNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/nvme/p288722/bal_28_devices_all_frames" --batch_size=128 --constrained=0 --gpu_id=0 --suffix="1_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConvNet"

#Constrained
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28D_all_frames/${SLURM_ARRAY_TASK_ID}" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="all_frames${SLURM_ARRAY_TASK_ID}" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConstNet"

#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_all_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="all_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConstNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_all_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="800_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConstNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_all_frames" --batch_size=128 --constrained=1 --gpu_id=0 --suffix="400_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConstNet"

#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/nvme/p288722/bal_28_devices_all_frames" --batch_size=128 --constrained=1 --gpu_id=1 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConstNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/nvme/p288722/bal_28_devices_all_frames" --batch_size=128 --constrained=1 --gpu_id=1 --suffix="100_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConstNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/nvme/p288722/bal_28_devices_all_frames" --batch_size=128 --constrained=1 --gpu_id=1 --suffix="50_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConstNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/nvme/p288722/bal_28_devices_all_frames" --batch_size=128 --constrained=1 --gpu_id=1 --suffix="20_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConstNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/nvme/p288722/bal_28_devices_all_frames" --batch_size=128 --constrained=1 --gpu_id=1 --suffix="10_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConstNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/nvme/p288722/bal_28_devices_all_frames" --batch_size=128 --constrained=1 --gpu_id=1 --suffix="5_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConstNet"
#python3 /home/p288722/git_code/scd_videos_tf/predict_flow.py --dataset="/nvme/p288722/bal_28_devices_all_frames" --batch_size=128 --constrained=1 --gpu_id=1 --suffix="1_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/icpram/baseline_ConstNet"

# MobileNet v3
#python3 /home/p288722/git_code/scd_videos_tf/train_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --epochs=20 --batch_size=32 --constrained=0  --gpu_id=1 --model_name="ConvNet" --global_results_dir="/scratch/p288722/runtime_data/scd_videos_tf/mobile_net_v3"
#python3 /home/p288722/git_code/scd_videos_tf/predict_constrained_net.py --dataset="/scratch/p288722/datasets/VISION/bal_28_devices_derrick" --batch_size=64 --constrained=0 --gpu_id=0 --suffix="200_frames" --input_dir="/scratch/p288722/runtime_data/scd_videos_tf/mobile_net_v3/models/ConvNet"
