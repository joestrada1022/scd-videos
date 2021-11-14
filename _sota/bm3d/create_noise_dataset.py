import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from bm3d import bm3d
from tqdm import tqdm


def extract_bm3d_noise(source_im_path, dest_im_path):
    """
    Center the mean to (127.5) and return the noise
    :param source_im_path:
    :param dest_im_path:
    :return:
    """
    im = Image.open(source_im_path)
    im = im.resize((800, 480))
    im = np.asarray(im) / 255.0

    # start = time.perf_counter()
    im_denoised = bm3d(im, sigma_psd=0.02)
    # end = time.perf_counter()
    # print(f'Total time for 1 img {end - start} sec')

    noise = im - im_denoised
    noise = (noise - np.mean(noise)) * 255.0 + 127.5
    noise = np.minimum(np.maximum(noise, 0), 255)
    noise = noise.astype(np.uint8)

    im = Image.fromarray(noise)
    im.save(dest_im_path)


def create_noise_dataset(source_splits_dir, dest_dataset_dir):
    """
    :param source_splits_dir:
    :param dest_dataset_dir:
    :return:
    """
    train_file = source_splits_dir.joinpath('train.json')
    val_file = source_splits_dir.joinpath('val.json')
    test_file = source_splits_dir.joinpath('test.json')

    for split_file in [train_file, val_file, test_file]:
        print(f'Processing: {split_file}')
        with open(split_file) as f:
            paths_dict = json.load(f)

        source_images = [Path(y) for x in paths_dict for y in paths_dict[x]]
        # source_dirs = set([x.parent for x in source_images])
        # for x in source_dirs:
        #     dest_dir = dest_dataset_dir.joinpath(x.parent.name, x.name)
        #     dest_dir.mkdir(parents=True, exist_ok=True)

        random.shuffle(source_images)
        for source_file in tqdm(source_images):
            # print(source_file)
            dest_file = dest_dataset_dir.joinpath(source_file.parent.parent.name,
                                                  source_file.parent.name,
                                                  source_file.name)
            if dest_file.exists():
                try:
                    Image.open(dest_file)
                except:
                    extract_bm3d_noise(source_file, dest_file)
            else:
                extract_bm3d_noise(source_file, dest_file)


def create_split_files(source_splits_dir, dest_dataset_dir, dest_splits_dir):
    """

    :param source_splits_dir:
    :param dest_dataset_dir:
    :param dest_splits_dir:
    :return:
    """
    dest_splits_dir.mkdir(exist_ok=True, parents=True)
    train_file = source_splits_dir.joinpath('train.json')
    val_file = source_splits_dir.joinpath('val.json')
    test_file = source_splits_dir.joinpath('test.json')

    for split_file in [train_file, val_file, test_file]:
        dest_paths_dict = {}
        with open(split_file) as f:
            source_paths_dict = json.load(f)
        for device in source_paths_dict:
            dest_paths_dict[device] = []
            for img in source_paths_dict[device]:
                source_img = Path(img)
                dest_img = dest_dataset_dir.joinpath(source_img.parent.parent.name,
                                                     source_img.parent.name,
                                                     source_img.name)
                dest_paths_dict[device].append(str(dest_img))

        dest_json_file = dest_splits_dir.joinpath(split_file.name)
        with open(dest_json_file, 'w+') as f:
            json.dump(dest_paths_dict, f, indent=2)


if __name__ == '__main__':
    create_noise_dataset(
        source_splits_dir=Path(r'/scratch/p288722/datasets/vision/old_baseline_split/bal_50_frames'),
        dest_dataset_dir=Path(r'/scratch/p288722/datasets/vision/bm3d_noise'),
    )

    # create_split_files(
    #     source_splits_dir=Path(r'/scratch/p288722/datasets/vision/old_baseline_split/bal_50_frames'),
    #     dest_dataset_dir=Path(r'/scratch/p288722/datasets/vision/bm3d_noise'),
    #     dest_splits_dir=Path(r'/scratch/p288722/datasets/vision/old_baseline_split/bal_50_frames_bm3d'),
    # )
