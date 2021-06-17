import json
import os
from pathlib import Path

from PIL import Image


def collect_filenames(devices_dir, destination_dir):
    filenames_dict = {}
    devices_list = sorted(list(devices_dir.glob('*')))
    for device in devices_list:
        print(f'Collecting stats for device - {device.name}')
        filenames_dict[device.name] = {}
        for frame in sorted(list(device.glob('*'))):
            img = Image.open(frame)
            filenames_dict[device.name][frame.name] = {
                'file_size': os.stat(frame).st_size,
                'frame_resolution': img.size
            }

    output_filename = destination_dir.joinpath(devices_dir.name + '.json')
    with open(output_filename, 'w+') as f:
        f.write(json.dumps(filenames_dict))


def generate_files_summary(source_dir, destination_dir):
    train_dir = source_dir.joinpath('train')
    test_dir = source_dir.joinpath('test')
    destination_dir.mkdir(exist_ok=True, parents=True)

    print('collecting stats from train frames')
    collect_filenames(train_dir, destination_dir)
    print('collecting stats from test frames')
    collect_filenames(test_dir, destination_dir)


if __name__ == '__main__':
    # Path to the folder containing train and test folders
    dataset_path = Path(r'/data/p288722/from_f118170/vbdi/datasets/VCDI_DS_28D')
    output_files_dir = Path(r'/scratch/p288722/datasets/VISION/files_for_comparison1')

    generate_files_summary(source_dir=dataset_path,
                           destination_dir=output_files_dir)
