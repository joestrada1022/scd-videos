import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# def create_I_frame_dataset(all_frames_views_dir, all_I_frames_views_dir, dataset_root_dir):
#     module = 'module load FFmpeg/4.2.2-GCCcore-9.3.0'  # load the ffmpeg module on the Peregrine
#     for split in all_frames_views_dir.glob('*.json'):
#         with open(split) as f:
#             all_frames_dict = json.load(f)
#
#         all_frames_I_dict = {}
#         for device in tqdm(all_frames_dict):
#             all_frames_I_dict[device] = []
#             videos = set([Path(x).parent.name for x in all_frames_dict[device]])
#             for video in videos:
#                 if len(list(dataset_root_dir.glob(rf'*/videos/*/{video}*'))) != 1:
#                     raise ValueError('Debug and check, why this is not equal to 1')
#                 video_path = list(dataset_root_dir.glob(rf'*/videos/*/{video}*'))[0]
#                 cmd = f"{module}\nffprobe {str(video_path)} -show_frames | grep -E 'pict_type'"
#                 frame_types = subprocess.run(cmd, shell=True, capture_output=True).stdout.decode('ascii').split('\n')
#                 frame_ids = set([str(idx + 1).zfill(5) for idx, x in enumerate(frame_types) if x == 'pict_type=I'])
#
#                 all_frame_paths = [Path(x) for x in all_frames_dict[device] if Path(x).parent.name == video]
#                 I_frame_paths = sorted([str(x) for x in all_frame_paths if x.name.split('-')[1][:5] in frame_ids])
#                 all_frames_I_dict[device].extend(I_frame_paths)
#
#         all_I_frames_views_dir.mkdir(parents=True, exist_ok=True)
#         with open(all_I_frames_views_dir.joinpath(split.name), 'w+') as f:
#             json.dump(all_frames_I_dict, f, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract and save I frames from videos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_dir', type=Path, required=True,
                        help='Path to VISION dataset (input directory consisting of folders per device)')
    parser.add_argument('--output_dir', type=Path, required=True, help='Output directory to save the frames')
    parser.add_argument('--device_id', type=int, required=True, help='Specify device ID')
    args = parser.parse_args()
    """
    --input_dir="/data/p288722/VISION/dataset" --output_dir="/scratch/p288722/datasets/vision/all_I_frames" --device_id=0
    """
    assert args.input_dir.exists(), 'Input directory does not exists!'
    assert 0 <= args.device_id <= 27, "Device ID must be in [0, 27]"
    # args.output_dir.mkdir(parents=True, exist_ok=True)

    devices = ['D01_Samsung_GalaxyS3Mini',
               'D02_Apple_iPhone4s',
               'D03_Huawei_P9',
               'D04_LG_D290',
               'D05_Apple_iPhone5c',
               'D06_Apple_iPhone6',
               'D07_Lenovo_P70A',
               'D08_Samsung_GalaxyTab3',
               'D09_Apple_iPhone4',
               'D10_Apple_iPhone4s',
               'D11_Samsung_GalaxyS3',
               'D12_Sony_XperiaZ1Compact',
               'D14_Apple_iPhone5c',
               'D15_Apple_iPhone6',
               'D16_Huawei_P9Lite',
               'D18_Apple_iPhone5c',
               'D19_Apple_iPhone6Plus',
               'D24_Xiaomi_RedmiNote3',
               'D25_OnePlus_A3000',
               'D26_Samsung_GalaxyS3Mini',
               'D27_Samsung_GalaxyS5',
               'D28_Huawei_P8',
               'D29_Apple_iPhone5',
               'D30_Huawei_Honor5c',
               'D31_Samsung_GalaxyS4Mini',
               'D32_OnePlus_A3003',
               'D33_Huawei_Ascend',
               'D34_Apple_iPhone5']

    args.device_name = devices[args.device_id]
    return args


def create_I_frames_dataset(args):
    source_device_dir = args.input_dir.joinpath(args.device_name)
    dest_device_dir = args.output_dir.joinpath(args.device_name)
    # dest_device_dir.mkdir(parents=True, exist_ok=True)

    module = 'module load FFmpeg/4.2.2-GCCcore-9.3.0'  # load the ffmpeg module on the Peregrine
    for video in tqdm(source_device_dir.glob('videos/*/*')):
        print(f'\nStarted processing video - {video}')

        # Identify the I Frames
        cmd = f"{module}\nffprobe {str(video)} -show_frames | grep -E 'pict_type'"
        frame_types = subprocess.run(cmd, shell=True, capture_output=True).stdout.decode('ascii').split('\n')
        frame_ids = set([str(idx + 1).zfill(5) for idx, x in enumerate(frame_types) if x == 'pict_type=I'])

        # Extract and save the I Frames
        dest_frames_dir = dest_device_dir.joinpath(video.stem)
        dest_frames_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video))  # Start capturing the feed
        total_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0

        while cap.isOpened():
            ret, frame = cap.read()  # Extract the frame
            if ret:  # Frame is available
                frame_id = str(int(cap.get(1))).zfill(5)  # Get current frame id
                if frame_id in frame_ids:
                    frame_path = dest_frames_dir.joinpath(f"{video.stem}-{frame_id}.png")
                    if not Path(frame_path).exists():
                        cv2.imwrite(str(frame_path), frame)

            count += 1

            if count >= total_num_frames:
                # Release the feed
                if cap.isOpened():
                    cap.release()
                break

        print('Finished processing the video')


def fix_file_names():
    pass


def run_flow():
    args = parse_args()
    create_I_frames_dataset(args)


def create_N_frames_split_from_all_frames(source_split_dir, dest_split_dir, num_frames=50):
    assert source_split_dir.exists(), 'The source directory does not exists!'
    dest_split_dir.mkdir(parents=True, exist_ok=True)

    for input_split_file in sorted(source_split_dir.glob('*.json')):
        with open(input_split_file, 'r') as f:
            input_data = json.load(f)

        output_data = {}
        # corrected_input_data = {}
        # correct_I_frame_dir = Path(r'/scratch/p288722/datasets/vision/all_I_frames')
        for device_name, frames in tqdm(sorted(input_data.items())):
            # frames = [correct_I_frame_dir.joinpath(Path(x).parent.parent.name).joinpath(Path(x).parent.name).joinpath(
            #     Path(x).name) for x in old_frames]
            # frames = [Path(str(x).replace('.mov-','-')) for x in frames]
            # frames = [Path(str(x).replace('.mov/', '/')) for x in frames]

            # for x in frames:
            #     assert x.exists(), f'Path: {x} - does not exists!'
            # frames = sorted([str(x) for x in frames])

            # corrected_input_data[device_name] = frames
            if frames:
                # segregate frames based on frames-per-video
                frames_per_video = {}
                for frame in frames:
                    video = str(Path(frame).parent)
                    if video in frames_per_video:
                        frames_per_video[video].append(frame)
                    else:
                        frames_per_video[video] = [frame]

                output_data[device_name] = []
                for video in frames_per_video:
                    n = len(frames_per_video[video])
                    uniformly_distributed_indices = np.unique(np.linspace(0, n, num_frames, endpoint=False).astype(int))
                    output_data[device_name].extend([frames_per_video[video][x] for x in uniformly_distributed_indices])
            else:
                output_data[device_name] = frames

            output_data[device_name] = sorted(output_data[device_name])

        output_split_file = dest_split_dir.joinpath(input_split_file.name)
        with open(output_split_file, 'w+') as f:
            json.dump(output_data, f, indent=2)
        # with open(input_split_file, 'w+') as f:
        #     json.dump(corrected_input_data, f, indent=2)


if __name__ == '__main__':
    # run_flow()
    create_N_frames_split_from_all_frames(
        source_split_dir=Path(r'/scratch/p288722/datasets/vision/I_frame_splits/bal_all_frames'),
        dest_split_dir=Path(r'/scratch/p288722/datasets/vision/I_frame_splits/bal_50_frames'),
        num_frames=50
    )

    # create_I_frame_dataset(
    #     all_frames_views_dir=Path(r'/scratch/p288722/datasets/vision/old_baseline_split/bal_all_frames'),
    #     all_I_frames_views_dir=Path(r'/scratch/p288722/datasets/vision/old_baseline_split/bal_all_I_frames'),
    #     dataset_root_dir=Path(r'/data/p288722/datasets/VISION/dataset')
    # )
