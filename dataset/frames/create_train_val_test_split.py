import argparse
import json
import random
from pathlib import Path

import math
from tqdm import tqdm

if __name__ == "__main__":
    """
        This script creates a balanced dataset for 28 devices.
        For each device 6x3 videos are included in train, followed by test and 1x3 in validation set
    """

    parser = argparse.ArgumentParser(
        description='Balance datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_frames', type=int, required=True, help='enter num frames to copy')
    parser.add_argument('--dest_path', type=str, required=True, help='enter the destination path')

    args = parser.parse_args()
    num_frames = args.num_frames
    dest_dir = Path(args.dest_path)

    input_dir = Path(r'/scratch/p288722/datasets/vision/all_frames')
    output_dir = dest_dir.joinpath(f'bal_all_frames') if num_frames == -1 else dest_dir.joinpath(
        f'bal_{num_frames}_frames')
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir.joinpath('train.json')
    val_file = output_dir.joinpath('val.json')
    test_file = output_dir.joinpath('test.json')

    random.seed(108)

    train_dict = {}
    val_dict = {}
    test_dict = {}


    def __check_WA_YT(x):
        if 'flat' in x.name:
            doesnt_exists_count = 2 - int(x.parent.joinpath(x.name.replace('_flat_', '_flatWA_')).exists()) - int(
                x.parent.joinpath(x.name.replace('_flat_', '_flatYT_')).exists())
        if 'indoor' in x.name:
            doesnt_exists_count = 2 - int(x.parent.joinpath(x.name.replace('_indoor_', '_indoorWA_')).exists()) - int(
                x.parent.joinpath(x.name.replace('_indoor_', '_indoorYT_')).exists())
        if 'outdoor' in x.name:
            doesnt_exists_count = 2 - int(x.parent.joinpath(x.name.replace('_outdoor_', '_outdoorWA_')).exists()) - int(
                x.parent.joinpath(x.name.replace('_outdoor_', '_outdoorYT_')).exists())

        return doesnt_exists_count


    def __add_WA_YT_versions(videos):
        extended_list = []
        for v in videos:
            for scenario in ['flat', 'indoor', 'outdoor']:
                if scenario in v.name:
                    break
            if v.parent.joinpath(v.name.replace(f'_{scenario}_', f'_{scenario}WA_')).exists():
                extended_list.append(v.parent.joinpath(v.name.replace(f'_{scenario}_', f'_{scenario}WA_')))
            if v.parent.joinpath(v.name.replace(f'_{scenario}_', f'_{scenario}YT_')).exists():
                extended_list.append(v.parent.joinpath(v.name.replace(f'_{scenario}_', f'_{scenario}YT_')))

        return videos + extended_list


    def __get_frames(videos, num_frames):
        all_frames = []
        for video in videos:
            frames = sorted([str(x) for x in video.glob('*')])
            if num_frames == -1:
                selected_frames = frames
            else:
                selected_frames = []
                for index in range(0, len(frames), int(math.floor(len(frames) / num_frames))):
                    selected_frames.append(frames[index])
                selected_frames = selected_frames[:num_frames]
            all_frames.extend(selected_frames)

        return all_frames


    for device in tqdm(sorted(list(input_dir.glob('*')))):
        val_dict[device.name] = []
        test_dict[device.name] = []

        train_videos, val_videos, test_videos = [], [], []

        all_videos = list(device.glob('*'))
        random.shuffle(all_videos)
        flat = sorted([x for x in all_videos if '_flat_' in x.name], key=__check_WA_YT)
        indoor = sorted([x for x in all_videos if '_indoor_' in x.name], key=__check_WA_YT)
        outdoor = sorted([x for x in all_videos if '_outdoor_' in x.name], key=__check_WA_YT)

        # 6 native videos in train
        train_videos.extend(flat[:2])
        train_videos.extend(indoor[:2])
        train_videos.extend(outdoor[:2])

        # 6 native videos in test
        test_videos.extend(flat[2:4])
        test_videos.extend(indoor[2:4])
        test_videos.extend(outdoor[2:4])

        # 3 native videos in val
        val_videos.extend(flat[4:5])
        val_videos.extend(indoor[4:5])
        val_videos.extend(outdoor[4:5])

        # augment with WA and YT versions
        train_videos = __add_WA_YT_versions(train_videos)
        test_videos = __add_WA_YT_versions(test_videos)
        val_videos = __add_WA_YT_versions(val_videos)

        # update the dictionary by choosing num_frames

        train_dict[device.name] = __get_frames(train_videos, num_frames)
        test_dict[device.name] = __get_frames(test_videos, num_frames)
        val_dict[device.name] = __get_frames(val_videos, num_frames)
        print(f'{device.name} : train - {len(train_videos)}, test - {len(test_videos)}, val - {len(val_videos)}')

    with open(train_file, 'w') as f:
        f.write(json.dumps(train_dict, indent=2))
    with open(test_file, 'w') as f:
        f.write(json.dumps(test_dict, indent=2))
    with open(val_file, 'w') as f:
        f.write(json.dumps(val_dict, indent=2))
