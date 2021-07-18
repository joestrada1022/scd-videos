from pathlib import Path

from matplotlib import pyplot as plt


def plot_histogram(frame_predictions_file_const, frame_predictions_file_conv, data_split=None):
    with open(frame_predictions_file_conv) as f1, open(frame_predictions_file_const) as f2:
        f1.__next__()
        f2.__next__()
        f1_flat_whole_frame, f1_others_whole_frame, f1_flat_resize_frame, f1_others_resize_frame = [], [], [], []
        for item in f1:
            parts = item.strip().split(',')
            if 'flat' in parts[0]:
                f1_flat_whole_frame.append(float(parts[4]))
                f1_flat_resize_frame.append(float(parts[4]))
            else:
                f1_others_whole_frame.append(float(parts[4]))
                f1_others_resize_frame.append(float(parts[4]))

        f2_flat_whole_frame, f2_others_whole_frame, f2_flat_resize_frame, f2_others_resize_frame = [], [], [], []
        for item in f2:
            parts = item.strip().split(',')
            if 'flat' in parts[0]:
                f2_flat_whole_frame.append(float(parts[4]))
                f2_flat_resize_frame.append(float(parts[4]))
            else:
                f2_others_whole_frame.append(float(parts[4]))
                f2_others_resize_frame.append(float(parts[4]))

    plt.figure()
    plt.hist(f1_flat_whole_frame, alpha=0.7, bins=100, label='flat')
    plt.hist(f1_others_whole_frame, alpha=0.7, bins=100, label='others')
    plt.title(f'Distribution of standard deviation - Whole frames \n {data_split} data')
    plt.xlabel('Standard deviation')
    plt.ylabel('Number of frames')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()

    plt.figure()
    plt.hist(f2_flat_whole_frame, alpha=0.7, bins=100, label='flat')
    plt.hist(f2_others_whole_frame, alpha=0.7, bins=100, label='others')
    plt.title(f'Distribution of standard deviation - Resized frames \n {data_split} data')
    plt.xlabel('Standard deviation')
    plt.ylabel('Number of frames')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


if __name__ == '__main__':
    plot_histogram(
        frame_predictions_file_const=
        Path(r'/scratch/p288722/runtime_data/scd_videos_tf/50_frames/mobile_net_1/models/ConvNet/'
             r'predictions_50_frames_val/frames/fm-e00018_F_predictions.csv'),
        frame_predictions_file_conv=
        Path(r'/scratch/p288722/runtime_data/scd_videos_tf/50_frames/mobile_net_2/models/ConstNet/'
             r'predictions_50_frames_val/frames/fm-e00014_F_predictions.csv'),
        data_split='50 frames per video on Validation'
    )

    plot_histogram(
        frame_predictions_file_const=
        Path(r'/scratch/p288722/runtime_data/scd_videos_tf/50_frames_pred/mobile_net_1/models/ConvNet/'
             r'predictions_50_frames/frames/fm-e00018_F_predictions.csv'),
        frame_predictions_file_conv=
        Path(r'/scratch/p288722/runtime_data/scd_videos_tf/50_frames_pred/mobile_net_2/models/ConstNet/'
             r'predictions_50_frames/frames/fm-e00014_F_predictions.csv'),
        data_split='50 frames per video on Test'
    )
