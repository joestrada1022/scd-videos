import argparse
import random
from pathlib import Path

import cv2
import lmdb
import numpy as np
from tqdm import tqdm


def generate_center_crop(source_images_dir, dest_images_dir, width=128, height=128):
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    source_img_paths = list(source_images_dir.glob('*'))
    random.shuffle(source_img_paths)

    for source_img_path in tqdm(source_img_paths):
        dest_img_path = dest_images_dir.joinpath(source_img_path.name)
        if not dest_img_path.exists():
            source_img = cv2.imread(str(source_img_path))
            img_height, img_width, _ = source_img.shape
            dest_img = source_img[
                       int(img_height / 2 - height / 2):int(img_height / 2 + height / 2),
                       int(img_width / 2 - width / 2):int(img_width / 2 + width / 2)
                       ]
            cv2.imwrite(str(dest_img_path), dest_img)


def generate_crops_lmdb(source_images_dir, split_type, dest_homo_root, dest_rand_root,
                        num_crops=1, width=128, height=128, stride=16):
    """
    Extract the most homogeneous crop among all the image crops. The crops are sampled by using the specified size.
    :param dest_rand_root:
    :param dest_homo_root:
    :param split_type:
    :param num_crops: number of crops to extract from each video frame based on the specified mode
    :param source_images_dir:
    :param width: of the crop in pixels
    :param height: of the crop in pixels
    :param stride: in number of pixels
    :return: None. Save the crops in the destination directory
    """

    dest_homo_root.mkdir(parents=True, exist_ok=True)
    dest_rand_root.mkdir(parents=True, exist_ok=True)

    source_img_paths = list(source_images_dir.glob('*'))
    total_num_images = len(list(source_images_dir.parent.glob('*/*.jpg')))  # images count from all devices
    random.shuffle(source_img_paths)

    # open lmdb database for writing
    estimated_img_size = 128 * 128 * 3
    estimated_img_name_size = 150
    buffer = 150
    map_size = total_num_images * (estimated_img_size + estimated_img_name_size + buffer) * num_crops
    lmdb_homo_filename = str(dest_homo_root.joinpath(f'{split_type}_lmdb'))
    lmdb_rand_filename = str(dest_rand_root.joinpath(f'{split_type}_lmdb'))

    for source_img_path in tqdm(source_img_paths):
        source_image_id = source_img_path.stem.split('-')[-1]
        source_device_id = source_img_path.stem.split('-')[0]
        dest_image_ids = [source_device_id + '-' + source_image_id + str(x).zfill(3) for x in range(num_crops)]

        with env_h.begin(write=True) as txn_h_check:
            test_id = txn_h_check.get(key=dest_image_ids[0].encode('ascii'), default=None)
        if not test_id:
            source_img = cv2.imread(str(source_img_path))
            img_height, img_width, _ = source_img.shape

            # Search for the most homogeneous crop
            all_crops = []
            for x in range(0, img_width - width + 1, stride):
                for y in range(0, img_height - height + 1, stride):
                    crop_img = source_img[y:y + height, x:x + width]
                    crop_std = np.std(crop_img / 255.0)
                    all_crops.append((crop_std, crop_img))

            # Save Random crops
            with lmdb.open(lmdb_rand_filename, map_size) as env_r:
                with env_r.begin(write=True) as txn_r:
                    random.seed(999)
                    random.shuffle(all_crops)
                    dest_crops = all_crops[:num_crops]
                    for patch_name, patch in zip(dest_image_ids, dest_crops):
                        patch = patch[1].tobytes()
                        patch_name = patch_name.encode('ascii')
                        txn_r.put(key=patch_name, value=patch)
                    txn_r.commit()

            # Save Homogeneous crops
            with lmdb.open(lmdb_homo_filename, map_size) as env_h:
                with env_h.begin(write=True) as txn_h:
                    dest_crops = sorted(all_crops, key=lambda z: z[0])[:num_crops]
                    for patch_name, patch in zip(dest_image_ids, dest_crops):
                        patch = patch[1].tobytes()
                        patch_name = patch_name.encode('ascii')
                        txn_h.put(key=patch_name, value=patch)
                    txn_h.commit()


def generate_crops(source_images_dir, split_type, dest_homo_root, dest_rand_root,
                   num_crops=1, width=128, height=128, stride=16):
    """
    Extract the most homogeneous crop among all the image crops. The crops are sampled by using the specified size.
    :param dest_rand_root:
    :param dest_homo_root:
    :param split_type:
    :param num_crops: number of crops to extract from each video frame based on the specified mode
    :param source_images_dir:
    :param width: of the crop in pixels
    :param height: of the crop in pixels
    :param stride: in number of pixels
    :return: None. Save the crops in the destination directory
    """

    dest_homo_root = dest_homo_root.joinpath(split_type).joinpath(source_images_dir.name)
    dest_rand_root = dest_rand_root.joinpath(split_type).joinpath(source_images_dir.name)
    dest_homo_root.mkdir(parents=True, exist_ok=True)
    dest_rand_root.mkdir(parents=True, exist_ok=True)

    source_img_paths = list(source_images_dir.glob('*'))
    random.shuffle(source_img_paths)

    for source_img_path in tqdm(source_img_paths):
        source_image_id = source_img_path.stem.split('-')[-1]
        source_device_id = source_img_path.stem.split('-')[0]
        dest_image_ids = [source_device_id + '-' + source_image_id + str(x).zfill(3) for x in range(num_crops)]
        dest_homo_image_paths = [dest_homo_root.joinpath(f'{x}.jpg') for x in dest_image_ids]
        dest_rand_image_paths = [dest_rand_root.joinpath(f'{x}.jpg') for x in dest_image_ids]

        if not dest_homo_image_paths[0].exists():
            source_img = cv2.imread(str(source_img_path))
            img_height, img_width, _ = source_img.shape

            # Search for the most homogeneous crop
            all_crops = []
            for x in range(0, img_width - width + 1, stride):
                for y in range(0, img_height - height + 1, stride):
                    crop_img = source_img[y:y + height, x:x + width]
                    crop_std = np.std(crop_img / 255.0)
                    all_crops.append((crop_std, crop_img))

            # Save Random crops
            random.seed(999)
            random.shuffle(all_crops)
            dest_crops = all_crops[:num_crops]
            for patch_path, patch_data in zip(dest_rand_image_paths, dest_crops):
                patch_data = patch_data[1]
                cv2.imwrite(str(patch_path), patch_data)

            # Save Homogeneous crops
            dest_crops = sorted(all_crops, key=lambda z: z[0])[:num_crops]
            for patch_path, patch_data in zip(dest_homo_image_paths, dest_crops):
                patch_data = patch_data[1]
                cv2.imwrite(str(patch_path), patch_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract dataset crops',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--device_id', type=int, required=True, help='enter the task id')
    parser.add_argument('--method', type=str, required=False,
                        help='choose between `center_crop`, `homogeneous_crop`, and `random_crop`')
    parser.add_argument('--count', type=int, required=True, help='enter the number of crops per image')

    args = parser.parse_args()
    device_id = args.device_id
    crops_per_frame = args.count
    method = args.method

    source_root = Path(r'/scratch/p288722/datasets/VISION/bal_28_devices_derrick')

    for split_type in ['train', 'test']:
        print(f'Starting {split_type}')
        source_device_paths = sorted(list(source_root.joinpath(split_type).glob('*')))
        if method == 'center_crop':
            dest_root = Path(rf'/scratch/p288722/datasets/VISION/center_crop_128x128_{crops_per_frame}')
            generate_center_crop(
                source_images_dir=source_device_paths[device_id],
                dest_images_dir=dest_root.joinpath(split_type).joinpath(source_device_paths[device_id].name)
            )
        else:
            dest_homo_root = Path(rf'/scratch/p288722/datasets/VISION/homo_crop_128x128_{crops_per_frame}')
            dest_rand_root = Path(rf'/scratch/p288722/datasets/VISION/rand_crop_128x128_{crops_per_frame}')
            generate_crops(
                source_images_dir=source_device_paths[device_id],
                split_type=split_type,
                dest_homo_root=dest_homo_root,
                dest_rand_root=dest_rand_root,
                num_crops=crops_per_frame
            )
