import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
from mat4py import loadmat
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute ImageQualityMetrics from MATLAB',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--frames_dataset', type=Path, default=Path(r'/scratch/p288722/datasets/vision/all_I_frames'))
    parser.add_argument('--dest_dir', type=Path, default=Path(r'/scratch/p288722/datasets/vision/I_frame_metrics'))
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--metric_type', type=str, default=None, choices=['brisque', 'niqe', 'piqe'])
    args = parser.parse_args()
    return args


def compute_iqms(args):
    """
    https://nl.mathworks.com/help/images/image-quality-metrics.html
    :param args:
    :return:
    """

    # import matlab.engine
    # eng = matlab.engine.start_matlab()  # Python to MATLAB connectivity takes about 12 seconds

    iqms_dict = {}
    iqms_info = {'device_name': None, 'scenario': None, 'compression_type': None, 'frame_type': 'I',
                 'brisque': None, 'niqe': None, 'piqe': None}

    # device_dir = sorted(args.frames_dataset.glob('*'))[args.device_id]
    dest_metrics_file = args.dest_dir.joinpath(f'{args.metric_type}_{args.device_id}.json')

    s = loadmat('iqm_scores.mat')['scores']
    scores = {img_name: {'brisque': b, 'piqe': p, 'niqe': n} for img_name, b, p, n in
              zip(s['name'], s['brisque'], s['piqe'], s['niqe'])}

    # count = 0
    for device_dir in tqdm(sorted(args.frames_dataset.glob('*'))):
        for image_path in sorted(device_dir.glob('*/*.png')):
            image_name = image_path.name
            iqms_dict[image_name] = copy.deepcopy(iqms_info)
            iqms_dict[image_name]['device_name'] = device_dir.name

            if 'WA' in image_name:
                iqms_dict[image_name]['compression_type'] = 'WhatsApp'
            elif 'YT' in image_name:
                iqms_dict[image_name]['compression_type'] = 'YouTube'
            else:
                iqms_dict[image_name]['compression_type'] = 'Native'

            if 'flat' in image_name:
                iqms_dict[image_name]['scenario'] = 'Flat'
            elif 'indoor' in image_name:
                iqms_dict[image_name]['scenario'] = 'Indoor'
            elif 'outdoor' in image_name:
                iqms_dict[image_name]['scenario'] = 'Outdoor'

            iqms_dict[image_name]['brisque'] = scores[image_name]['brisque']
            iqms_dict[image_name]['niqe'] = scores[image_name]['niqe']
            iqms_dict[image_name]['piqe'] = scores[image_name]['piqe']

            # img = np.array(Image.open(image_path))
            # matlab_img = matlab.uint8(initializer=img.tolist(), size=img.shape, is_complex=False)
            # if args.metric_type is None:
            #     iqms_dict[image_name]['brisque'] = eng.brisque(matlab_img)
            #     iqms_dict[image_name]['niqe'] = eng.niqe(matlab_img)
            #     iqms_dict[image_name]['piqe'] = eng.piqe(matlab_img)
            # elif args.metric_type == 'brisque':
            #     iqms_dict[image_name]['brisque'] = eng.brisque(matlab_img)
            # elif args.metric_type == 'niqe':
            #     iqms_dict[image_name]['niqe'] = eng.niqe(matlab_img)
            # elif args.metric_type == 'piqe':
            #     iqms_dict[image_name]['piqe'] = eng.piqe(matlab_img)
            #
            # count += 1
            # if count % 10 == 0:
            #     with open(dest_metrics_file, 'w+') as f:
            #         json.dump(iqms_dict, f, indent=2)

    with open(dest_metrics_file, 'w+') as f:
        json.dump(iqms_dict, f, indent=2)
    return iqms_dict


def analyze_data():
    metrics_file = Path(r'D:\Datasets\vision\all_I_frames.json')
    with open(metrics_file, 'r') as f:
        X = json.load(f)

    plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15, 5))

    ax1.hist([X[x]['brisque'] for x in X if X[x]['scenario'] == 'Flat'],
             bins=100, label='Flat', alpha=0.70)
    ax1.hist([X[x]['brisque'] for x in X if X[x]['scenario'] == 'Indoor'],
             bins=100, label='Indoor', alpha=0.70)
    ax1.hist([X[x]['brisque'] for x in X if X[x]['scenario'] == 'Outdoor'],
             bins=100, label='Outdoor', alpha=0.70)
    ax1.set_ylabel('Count')
    ax1.set_xlabel('brisque score')
    # ax1.legend()

    ax2.hist([X[x]['piqe'] for x in X if X[x]['scenario'] == 'Flat'],
             bins=100, label='Flat', alpha=0.70)
    ax2.hist([X[x]['piqe'] for x in X if X[x]['scenario'] == 'Indoor'],
             bins=100, label='Indoor', alpha=0.70)
    ax2.hist([X[x]['piqe'] for x in X if X[x]['scenario'] == 'Outdoor'],
             bins=100, label='Outdoor', alpha=0.70)
    ax2.set_xlabel('piqe score')
    # ax2.legend()

    ax3.hist([X[x]['niqe'] for x in X if X[x]['scenario'] == 'Flat'],
             bins=100, label='Flat', alpha=0.70)
    ax3.hist([X[x]['niqe'] for x in X if X[x]['scenario'] == 'Indoor'],
             bins=100, label='Indoor', alpha=0.70)
    ax3.hist([X[x]['niqe'] for x in X if X[x]['scenario'] == 'Outdoor'],
             bins=100, label='Outdoor', alpha=0.70)
    ax3.set_xlabel('niqe score')
    # ax3.legend()

    fig.suptitle('Distribution of Image Quality Metrics for all I-frames (considering all compression types)', fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15, 5))

    ax1.hist([X[x]['brisque'] for x in X if X[x]['scenario'] == 'Flat' and X[x]['compression_type'] == 'Native'],
             bins=100, label='Flat', alpha=0.70)
    ax1.hist([X[x]['brisque'] for x in X if X[x]['scenario'] == 'Indoor' and X[x]['compression_type'] == 'Native'],
             bins=100, label='Indoor', alpha=0.70)
    ax1.hist([X[x]['brisque'] for x in X if X[x]['scenario'] == 'Outdoor' and X[x]['compression_type'] == 'Native'],
             bins=100, label='Outdoor', alpha=0.70)
    ax1.set_ylabel('Count')
    ax1.set_xlabel('brisque score')

    ax2.hist([X[x]['piqe'] for x in X if X[x]['scenario'] == 'Flat' and X[x]['compression_type'] == 'Native'],
             bins=100, label='Flat', alpha=0.70)
    ax2.hist([X[x]['piqe'] for x in X if X[x]['scenario'] == 'Indoor' and X[x]['compression_type'] == 'Native'],
             bins=100, label='Indoor', alpha=0.70)
    ax2.hist([X[x]['piqe'] for x in X if X[x]['scenario'] == 'Outdoor' and X[x]['compression_type'] == 'Native'],
             bins=100, label='Outdoor', alpha=0.70)
    ax2.set_xlabel('piqe score')

    ax3.hist([X[x]['niqe'] for x in X if X[x]['scenario'] == 'Flat' and X[x]['compression_type'] == 'Native'],
             bins=100, label='Flat', alpha=0.70)
    ax3.hist([X[x]['niqe'] for x in X if X[x]['scenario'] == 'Indoor' and X[x]['compression_type'] == 'Native'],
             bins=100, label='Indoor', alpha=0.70)
    ax3.hist([X[x]['niqe'] for x in X if X[x]['scenario'] == 'Outdoor' and X[x]['compression_type'] == 'Native'],
             bins=100, label='Outdoor', alpha=0.70)
    ax3.set_xlabel('niqe score')

    fig.suptitle('Distribution of Image Quality Metrics for all I-frames (considering Native videos)', fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15, 5))

    ax1.hist([X[x]['brisque'] for x in X if X[x]['scenario'] == 'Flat' and X[x]['compression_type'] == 'YouTube'],
             bins=100, label='Flat', alpha=0.70)
    ax1.hist([X[x]['brisque'] for x in X if X[x]['scenario'] == 'Indoor' and X[x]['compression_type'] == 'YouTube'],
             bins=100, label='Indoor', alpha=0.70)
    ax1.hist([X[x]['brisque'] for x in X if X[x]['scenario'] == 'Outdoor' and X[x]['compression_type'] == 'YouTube'],
             bins=100, label='Outdoor', alpha=0.70)
    ax1.set_ylabel('Count')
    ax1.set_xlabel('brisque score')

    ax2.hist([X[x]['piqe'] for x in X if X[x]['scenario'] == 'Flat' and X[x]['compression_type'] == 'YouTube'],
             bins=100, label='Flat', alpha=0.70)
    ax2.hist([X[x]['piqe'] for x in X if X[x]['scenario'] == 'Indoor' and X[x]['compression_type'] == 'YouTube'],
             bins=100, label='Indoor', alpha=0.70)
    ax2.hist([X[x]['piqe'] for x in X if X[x]['scenario'] == 'Outdoor' and X[x]['compression_type'] == 'YouTube'],
             bins=100, label='Outdoor', alpha=0.70)
    ax2.set_xlabel('piqe score')

    ax3.hist([X[x]['niqe'] for x in X if X[x]['scenario'] == 'Flat' and X[x]['compression_type'] == 'YouTube'],
             bins=100, label='Flat', alpha=0.70)
    ax3.hist([X[x]['niqe'] for x in X if X[x]['scenario'] == 'Indoor' and X[x]['compression_type'] == 'YouTube'],
             bins=100, label='Indoor', alpha=0.70)
    ax3.hist([X[x]['niqe'] for x in X if X[x]['scenario'] == 'Outdoor' and X[x]['compression_type'] == 'YouTube'],
             bins=100, label='Outdoor', alpha=0.70)
    ax3.set_xlabel('niqe score')

    fig.suptitle('Distribution of Image Quality Metrics for all I-frames (considering YouTube videos)', fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15, 5))

    ax1.hist([X[x]['brisque'] for x in X if X[x]['scenario'] == 'Flat' and X[x]['compression_type'] == 'WhatsApp'],
             bins=100, label='Flat', alpha=0.70)
    ax1.hist([X[x]['brisque'] for x in X if X[x]['scenario'] == 'Indoor' and X[x]['compression_type'] == 'WhatsApp'],
             bins=100, label='Indoor', alpha=0.70)
    ax1.hist([X[x]['brisque'] for x in X if X[x]['scenario'] == 'Outdoor' and X[x]['compression_type'] == 'WhatsApp'],
             bins=100, label='Outdoor', alpha=0.70)
    ax1.set_ylabel('Count')
    ax1.set_xlabel('brisque score')

    ax2.hist([X[x]['piqe'] for x in X if X[x]['scenario'] == 'Flat' and X[x]['compression_type'] == 'WhatsApp'],
             bins=100, label='Flat', alpha=0.70)
    ax2.hist([X[x]['piqe'] for x in X if X[x]['scenario'] == 'Indoor' and X[x]['compression_type'] == 'WhatsApp'],
             bins=100, label='Indoor', alpha=0.70)
    ax2.hist([X[x]['piqe'] for x in X if X[x]['scenario'] == 'Outdoor' and X[x]['compression_type'] == 'WhatsApp'],
             bins=100, label='Outdoor', alpha=0.70)
    ax2.set_xlabel('piqe score')

    ax3.hist([X[x]['niqe'] for x in X if X[x]['scenario'] == 'Flat' and X[x]['compression_type'] == 'WhatsApp'],
             bins=100, label='Flat', alpha=0.70)
    ax3.hist([X[x]['niqe'] for x in X if X[x]['scenario'] == 'Indoor' and X[x]['compression_type'] == 'WhatsApp'],
             bins=100, label='Indoor', alpha=0.70)
    ax3.hist([X[x]['niqe'] for x in X if X[x]['scenario'] == 'Outdoor' and X[x]['compression_type'] == 'WhatsApp'],
             bins=100, label='Outdoor', alpha=0.70)
    ax3.set_xlabel('niqe score')

    fig.suptitle('Distribution of Image Quality Metrics for all I-frames (considering WhatsApp videos)', fontsize=18)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(' ')


def run_flow():
    # args = parse_args()
    # iqms_dict = compute_iqms(args)
    analyze_data()


if __name__ == '__main__':
    """
    Make use of the three non-reference based metrics available in MATLAB to compute Image Quality Metrics (IQM)
    """
    run_flow()
