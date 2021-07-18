import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_distribution(view_file):
    with open(view_file) as f:
        view_dict = json.load(f)

    distribution = {}
    for device in view_dict:
        videos = set([Path(x).parent.name for x in view_dict[device]])
        num_frames_per_video = [(sum([v in x for x in view_dict[device]]), v) for v in videos]
        distribution[device] = num_frames_per_video

    return distribution


def plot_data_distribution(source_view_dir):
    train_dist = get_distribution(source_view_dir.joinpath('train.json'))
    val_dist = get_distribution(source_view_dir.joinpath('val.json'))
    test_dist = get_distribution(source_view_dir.joinpath('test.json'))

    # ticks = [x for x in train_dist]
    ticks = [idx + 1 for idx, _ in enumerate(train_dist)]

    data_a = [[y[0] for y in train_dist[x]] for x in train_dist]
    data_b = [[y[0] for y in test_dist[x]] for x in test_dist]

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure(figsize=(10, 4))

    bp_a = plt.boxplot(data_a, positions=np.array(range(len(data_a))) * 2.0 - 0.4, sym='', widths=0.6)
    bp_b = plt.boxplot(data_b, positions=np.array(range(len(data_b))) * 2.0 + 0.4, sym='', widths=0.6)
    set_box_color(bp_a, '#a1d99b')  # colors are from http://colorbrewer2.org/
    set_box_color(bp_b, '#2C7BB6')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#a1d99b', label='Train')
    plt.plot([], c='#2C7BB6', label='Test')
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.xlim(-2, len(ticks) * 2)
    # plt.ylim(0, 8)
    plt.xlabel('Device ID')
    plt.ylabel('Number of I-frames per video')
    plt.title('Distribution of I-frames')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_data_distribution(
        source_view_dir=Path(r'/scratch/p288722/datasets/vision/8_devices/bal_all_I_frames')
    )
