import json
import os
from collections import namedtuple
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'   # to run only on CPU
StdDev = namedtuple('StdDev', 'whole_frame, resize_frame')


def measure_std_dev(dataset_view, frame_predictions_file_const, frame_predictions_file_conv):
    with open(dataset_view) as f:
        json_data = json.load(f)

    file_paths = {}
    for device in json_data:
        paths = json_data[device]
        for path in paths:
            file_paths[f'b\'{Path(path).name}\''] = path

    with open(frame_predictions_file_const) as f1, open(frame_predictions_file_conv) as f2:
        header = f1.__next__()
        f1_lines = f1.readlines()
        f2_lines = f2.readlines()[1:]

    all_file_names = set([x.split(',')[0] for x in f1_lines] + [x.split(',')[0] for x in f2_lines])
    frame_dict = {}
    for item in tqdm(all_file_names):
        img = tf.io.read_file(file_paths[item])
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.dtypes.float32)

        # Resize to ConvNet input dimensions
        resize_img = tf.image.resize(img, size=tf.constant((480, 800), tf.dtypes.int32))

        whole_frame_std = tf.math.reduce_std(img).numpy()
        resize_frame_std = tf.math.reduce_std(resize_img).numpy()

        frame_dict[item] = StdDev(whole_frame=whole_frame_std, resize_frame=resize_frame_std)

    for index, item in enumerate(f1_lines):
        frame_parts = [x for x in item.strip().split(',')][:4]
        frame_std = frame_dict[frame_parts[0]]
        frame_parts += [str(frame_std.whole_frame), str(frame_std.resize_frame)]
        f1_lines[index] = ','.join(frame_parts) + '\n'

    for index, item in enumerate(f2_lines):
        frame_parts = [x for x in item.strip().split(',')][:4]
        frame_std = frame_dict[frame_parts[0]]
        frame_parts += [str(frame_std.whole_frame), str(frame_std.resize_frame)]
        f2_lines[index] = ','.join(frame_parts) + '\n'

    header = [','.join(header.strip().split(',')[:4] + ['whole_frame_std', 'resize_frame_std']) + '\n']

    with open(frame_predictions_file_const, 'w+') as f1, open(frame_predictions_file_conv, 'w+') as f2:
        f1.writelines(header + f1_lines)
        f2.writelines(header + f2_lines)

    print(' ')


if __name__ == '__main__':
    # 50 frames Validation Set
    # measure_std_dev(
    #     dataset_view=Path(r'/scratch/p288722/datasets/vision/bal_50_frames/val.json'),
    #     frame_predictions_file_const=
    #     Path(r'/scratch/p288722/runtime_data/scd_videos_tf/50_frames/mobile_net_1/models/ConvNet/'
    #          r'predictions_50_frames_val/frames/fm-e00018_F_predictions.csv'),
    #     frame_predictions_file_conv=
    #     Path(r'/scratch/p288722/runtime_data/scd_videos_tf/50_frames/mobile_net_2/models/ConstNet/'
    #          r'predictions_50_frames_val/frames/fm-e00014_F_predictions.csv')
    # )

    measure_std_dev(
        dataset_view=Path(r'/scratch/p288722/datasets/vision/bal_200_frames/test.json'),
        frame_predictions_file_const=
        Path(r'/scratch/p288722/runtime_data/scd_videos_tf/200_frames_pred/mobile_net_2/models/ConvNet/'
             r'predictions_200_frames/frames/fm-e00007_F_predictions.csv'),
        frame_predictions_file_conv=
        Path(r'/scratch/p288722/runtime_data/scd_videos_tf/200_frames_pred/mobile_net_2/models/ConstNet/'
             r'predictions_200_frames/frames/fm-e00007_F_predictions.csv')
    )
