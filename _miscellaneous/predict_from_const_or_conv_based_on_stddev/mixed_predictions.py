from pathlib import Path


def mixed_predictions(conv_net_frame_predictions, const_net_frame_predictions, standard_deviation_file, threshold=0.1):
    std_dev_map = {}
    with open(standard_deviation_file) as f:
        f.__next__()
        for line in f:
            parts = line.split(',')
            std_dev_map[parts[0].split("'")[1]] = float(parts[4].strip())

    all_lines = []
    with open(conv_net_frame_predictions) as f1, open(const_net_frame_predictions) as f2:
        all_lines.append(','.join(f1.__next__().strip().split(',')[:4]) + '\n')
        for line in f1:
            parts = line.split(',')
            key = parts[0].split("'")[1]
            if std_dev_map[key] > threshold:  # use conv net
                all_lines.append(','.join(parts[:4]) + '\n')
        f2.__next__()
        for line in f2:
            parts = line.split(',')
            key = parts[0].split("'")[1]
            if std_dev_map[key] <= threshold:  # use const net
                all_lines.append(','.join(parts[:4]) + '\n')

    parts = list(conv_net_frame_predictions.parts)
    parts[-3] = 'predictions_mixed_50_frames'
    filename = Path('/'.join(parts))
    filename.parent.mkdir(exist_ok=True, parents=True)

    with open(filename, 'w+') as f:
        f.writelines(all_lines)


if __name__ == '__main__':
    # mixed_predictions(
    #     conv_net_frame_predictions=Path(r'/scratch/p288722/runtime_data/scd_videos_tf/50_frames_pred_bal_val/mobile_net_1/'
    #                                     r'models/ConvNet/predictions_50_frames/frames/fm-e00016_F_predictions.csv'),
    #     const_net_frame_predictions=Path(r'/scratch/p288722/runtime_data/scd_videos_tf/50_frames_pred_bal_val/mobile_net_2/'
    #                                      r'models/ConstNet/predictions_50_frames/frames/fm-e00013_F_predictions.csv'),
    #     standard_deviation_file=Path(r'/scratch/p288722/runtime_data/scd_videos_tf/50_frames_pred/mobile_net_2/'
    #                                  r'models/ConstNet/predictions_50_frames/frames/fm-e00014_F_predictions.csv'),
    #     threshold=0.1
    # )

    mixed_predictions(
        conv_net_frame_predictions=Path(
            r'/scratch/p288722/runtime_data/scd_videos_tf/200_frames_pred/mobile_net_3/'
            r'models/ConvNet/predictions_200_frames/frames/fm-e00006_F_predictions.csv'),
        const_net_frame_predictions=Path(
            r'/scratch/p288722/runtime_data/scd_videos_tf/200_frames_pred/mobile_net_3/'
            r'models/ConstNet/predictions_200_frames/frames/fm-e00008_F_predictions.csv'),
        standard_deviation_file=Path(r'/scratch/p288722/runtime_data/scd_videos_tf/200_frames_pred/mobile_net_2/'
                                     r'models/ConstNet/predictions_200_frames/frames/fm-e00007_F_predictions.csv'),
        threshold=0.1
    )
