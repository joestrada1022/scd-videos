import argparse
import copy
import json
from pathlib import Path

import numpy as np
from mat4py import loadmat
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute ImageQualityMetrics from MATLAB',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--frames_dataset', type=Path, default=Path(r'/scratch/p288722/datasets/vision/all_I_frames'))
    parser.add_argument('--dest_dir', type=Path, default=Path(r'/scratch/p288722/datasets/vision/I_frame_metrics'))
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--metric_type', type=str, default='all', choices=['brisque', 'niqe', 'piqe'])
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
    iqms_info = {'device_name': '', 'scenario': '', 'compression_type': '', 'frame_type': 'I',
                 'brisque': float('nan'), 'niqe': float('nan'), 'piqe': float('nan'),
                 'mean_homogeneity': float('nan'), 'mean_energy': float('nan'), 'entropy': float('nan')}

    device_dir = sorted(args.frames_dataset.glob('*'))[args.device_id]
    # dest_metrics_file = args.dest_dir.joinpath(f'all_I_frames.json')
    dest_metrics_file = args.dest_dir.joinpath(f'{args.metric_type}_{args.device_id}.json')

    s = loadmat(str(Path(__file__).resolve().parent.joinpath('scores.mat')))['scores']
    scores = {img_name: {'brisque': b, 'piqe': p, 'niqe': n} for img_name, b, p, n in
              zip(s['name'], s['brisque'], s['piqe'], s['niqe'])}

    # count = 0
    # for device_dir in tqdm(sorted(args.frames_dataset.glob('*'))):
    for image_path in tqdm(sorted(device_dir.glob('*/*.png'))):
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

        # compute homogeneity, entropy, and energy scores using skimage library

        img = np.array(Image.open(image_path).convert('L'))
        glcm = graycomatrix(img, [1, 2], [0, np.pi / 2], levels=256, normed=True)
        iqms_dict[image_name]['mean_homogeneity'] = float(np.mean(graycoprops(glcm, prop='homogeneity')))
        iqms_dict[image_name]['mean_energy'] = float(np.mean(graycoprops(glcm, prop='energy')))
        iqms_dict[image_name]['entropy'] = float(shannon_entropy(img, base=2))

    with open(dest_metrics_file, 'w+') as f:
        json.dump(iqms_dict, f, indent=2)
    return iqms_dict


def plot_iqm_scores(X, save_filepath=None, c='all'):
    """
    Plots the IQM scores
    :param X: Input .json data as a dictionary
    :param save_filepath: the destination filename
    :param c: Compression type (all, Native, WhatsApp, YouTube)
    :return: None
    """
    plt.figure()
    fig, ax = plt.subplots(nrows=3, ncols=3, sharey='col', sharex='col', figsize=(15, 5))

    if c == 'all':
        ax[0][0].hist([X[x]['brisque'] for x in X if X[x]['scenario'] == 'Flat'],
                      bins=100, label='Flat', alpha=0.70, color="#ffa500")
        ax[1][0].hist([X[x]['brisque'] for x in X if X[x]['scenario'] == 'Indoor'],
                      bins=100, label='Indoor', alpha=0.70, color="#88a174")
        ax[2][0].hist([X[x]['brisque'] for x in X if X[x]['scenario'] == 'Outdoor'],
                      bins=100, label='Outdoor', alpha=0.70, color="#80576b")

        ax[0][1].hist([X[x]['piqe'] for x in X if X[x]['scenario'] == 'Flat'],
                      bins=100, label='Flat', alpha=0.70, color="#ffa500")
        ax[1][1].hist([X[x]['piqe'] for x in X if X[x]['scenario'] == 'Indoor'],
                      bins=100, label='Indoor', alpha=0.70, color="#88a174")
        ax[2][1].hist([X[x]['piqe'] for x in X if X[x]['scenario'] == 'Outdoor'],
                      bins=100, label='Outdoor', alpha=0.70, color="#80576b")

        ax[0][2].hist([X[x]['niqe'] for x in X if X[x]['scenario'] == 'Flat'],
                      bins=100, label='Flat', alpha=0.70, color="#ffa500")
        ax[1][2].hist([X[x]['niqe'] for x in X if X[x]['scenario'] == 'Indoor'],
                      bins=100, label='Indoor', alpha=0.70, color="#88a174")
        ax[2][2].hist([X[x]['niqe'] for x in X if X[x]['scenario'] == 'Outdoor'],
                      bins=100, label='Outdoor', alpha=0.70, color="#80576b")
    else:
        ax[0][0].hist([X[x]['brisque'] for x in X if X[x]['scenario'] == 'Flat' and X[x]['compression_type'] == c],
                      bins=100, label='Flat', alpha=0.70, color="#ffa500")
        ax[1][0].hist([X[x]['brisque'] for x in X if X[x]['scenario'] == 'Indoor' and X[x]['compression_type'] == c],
                      bins=100, label='Indoor', alpha=0.70, color="#88a174")
        ax[2][0].hist([X[x]['brisque'] for x in X if X[x]['scenario'] == 'Outdoor' and X[x]['compression_type'] == c],
                      bins=100, label='Outdoor', alpha=0.70, color="#80576b")

        ax[0][1].hist([X[x]['piqe'] for x in X if X[x]['scenario'] == 'Flat' and X[x]['compression_type'] == c],
                      bins=100, label='Flat', alpha=0.70, color="#ffa500")
        ax[1][1].hist([X[x]['piqe'] for x in X if X[x]['scenario'] == 'Indoor' and X[x]['compression_type'] == c],
                      bins=100, label='Indoor', alpha=0.70, color="#88a174")
        ax[2][1].hist([X[x]['piqe'] for x in X if X[x]['scenario'] == 'Outdoor' and X[x]['compression_type'] == c],
                      bins=100, label='Outdoor', alpha=0.70, color="#80576b")

        ax[0][2].hist([X[x]['niqe'] for x in X if X[x]['scenario'] == 'Flat' and X[x]['compression_type'] == c],
                      bins=100, label='Flat', alpha=0.70, color="#ffa500")
        ax[1][2].hist([X[x]['niqe'] for x in X if X[x]['scenario'] == 'Indoor' and X[x]['compression_type'] == c],
                      bins=100, label='Indoor', alpha=0.70, color="#88a174")
        ax[2][2].hist([X[x]['niqe'] for x in X if X[x]['scenario'] == 'Outdoor' and X[x]['compression_type'] == c],
                      bins=100, label='Outdoor', alpha=0.70, color="#80576b")

    ax[2][0].set_xlabel('brisque score')
    ax[2][1].set_xlabel('piqe score')
    ax[2][2].set_xlabel('niqe score')
    ax[0][0].set_ylabel('Count')
    ax[1][0].set_ylabel('Count')
    ax[2][0].set_ylabel('Count')
    ax[0][0].legend(loc='upper left')
    ax[1][0].legend(loc='upper left')
    ax[2][0].legend(loc='upper left')

    fig.suptitle(f'Distribution of IQM scores for all I-frames (considering {c} videos)', fontsize=16)
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_filepath)


def plot_iqm_scores_with_predictions(X, save_filepath=None, c='all'):
    """
    Plots the IQM scores
    :param X: Input .json data as a dictionary
    :param save_filepath: the destination filename
    :param c: Compression type (all, Native, WhatsApp, YouTube)
    :return: None
    """
    plt.figure()
    fig, ax = plt.subplots(nrows=3, ncols=3, sharey='col', sharex='col', figsize=(15, 5))

    scenarios = ['Flat', 'Indoor', 'Outdoor']

    if c == 'all':
        for col_idx, score in enumerate(['brisque', 'piqe', 'niqe']):
            for row_idx, (scenario, color) in enumerate(zip(scenarios, ['#88bc70', '#ffe069', '#5d8bba'])):
                x1 = [X[x][score] for x in X if X[x]['scenario'] == scenario and X[x]['misclassified'] == False]
                x2 = [X[x][score] for x in X if X[x]['scenario'] == scenario and X[x]['misclassified'] == True]
                ax[row_idx][col_idx].hist([np.array(x1), np.array(x2)], bins=100, stacked=True,
                                          label=[scenario, 'Misclassified'], color=[color, "#ff3636"])
    else:
        for col_idx, score in enumerate(['brisque', 'piqe', 'niqe']):
            for row_idx, (scenario, color) in enumerate(zip(scenarios, ['#88bc70', '#ffe069', '#5d8bba'])):
                x1 = [X[x][score] for x in X if
                      X[x]['scenario'] == scenario and X[x]['misclassified'] == False and X[x]['compression_type'] == c]
                x2 = [X[x][score] for x in X if
                      X[x]['scenario'] == scenario and X[x]['misclassified'] == True and X[x]['compression_type'] == c]
                ax[row_idx][col_idx].hist([np.array(x1), np.array(x2)], bins=100, stacked=True,
                                          label=[scenario, 'Misclassified'], color=[color, "#ff3636"])

    ax[2][0].set_xlabel('brisque score')
    ax[2][1].set_xlabel('piqe score')
    ax[2][2].set_xlabel('niqe score')
    ax[0][0].set_ylabel('Count')
    ax[1][0].set_ylabel('Count')
    ax[2][0].set_ylabel('Count')
    ax[0][0].legend(loc='upper left')
    ax[1][0].legend(loc='upper left')
    ax[2][0].legend(loc='upper left')
    fig.suptitle(f'Distribution of IQM scores for 50 I-frames in the test set \n(considering {c} videos)', fontsize=14)
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_filepath)


def analyze_data():
    metrics_file = Path(r'/scratch/p288722/datasets/vision/I_frame_metrics/all_I_frames.json')
    with open(metrics_file, 'r') as f:
        X = json.load(f)
    dest_dir = Path(r'/scratch/p288722/datasets/vision/I_frame_metrics')

    # plot_iqm_scores(X, dest_dir.joinpath('all_I_frames_combined.png'), 'all')
    # plot_iqm_scores(X, dest_dir.joinpath('all_I_frames_Native.png'), 'Native')
    # plot_iqm_scores(X, dest_dir.joinpath('all_I_frames_WhatsApp.png'), 'WhatsApp')
    # plot_iqm_scores(X, dest_dir.joinpath('all_I_frames_YouTube.png'), 'YouTube')

    import pandas as pd
    f = pd.read_csv(r'/scratch/p288722/runtime_data/scd_videos_first_revision/06_I_frames/50_frames_pred/mobile_net'
                    r'/models/MobileNet_50_I_frames_ccrop_run1/predictions_50_frames/frames/fm-e00020_F_predictions.csv')
    # Prepare data for plot
    data_for_plot = {}
    for idx, row in f.iterrows():
        filename = Path(row.File).name
        frame_properties = X[filename]
        frame_properties['misclassified'] = row['True Label'] != row['Predicted Label']
        data_for_plot[filename] = frame_properties

    plot_iqm_scores_with_predictions(data_for_plot, dest_dir.joinpath('50_I_frames_combined_test.png'), 'all')
    plot_iqm_scores_with_predictions(data_for_plot, dest_dir.joinpath('50_I_frames_Native_test.png'), 'Native')
    plot_iqm_scores_with_predictions(data_for_plot, dest_dir.joinpath('50_I_frames_WhatsApp_test.png'), 'WhatsApp')
    plot_iqm_scores_with_predictions(data_for_plot, dest_dir.joinpath('50_I_frames_YouTube_test.png'), 'YouTube')


def run_flow():
    args = parse_args()

    # files = [args.dest_dir.joinpath(f'all_{x}.json') for x in range(0, 28)]
    # combined_data = {}
    # for file in files:
    #     with open(file, 'r') as f:
    #         data = json.load(f)
    #     combined_data = combined_data | data
    # with open(args.dest_dir.joinpath('all_I_frames.json'), 'w+') as f:
    #     json.dump(combined_data, f, indent=2)

    iqms_dict = compute_iqms(args)
    analyze_data()


if __name__ == '__main__':
    """
    Make use of the three non-reference based metrics available in MATLAB to compute Image Quality Metrics (IQM)
    """
    run_flow()
