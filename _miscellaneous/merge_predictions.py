from pathlib import Path

from tqdm import tqdm


def merge_predictions(source_dir, num_fragments, dest_dir):
    header_line = []
    lines =[]
    for index in tqdm(range(num_fragments)):
        predictions_file = source_dir.joinpath(f'predictions_all_frames_{index}/frames/fm-e00007_F_predictions.csv')
        with open(predictions_file, 'r') as f:
            header_line = [f.__next__()]
            lines += f.readlines()

    dest_file = dest_dir.joinpath(f'predictions_all_frames/frames/fm-e00007_F_predictions.csv')
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_file, 'w+') as f:
        f.writelines(header_line + lines)


if __name__ == '__main__':
    merge_predictions(
        source_dir=Path(r'/scratch/p288722/runtime_data/scd_videos_tf/200_frames_pred/mobile_net_2/models/ConvNet'),
        num_fragments=10,
        dest_dir=Path(r'/scratch/p288722/runtime_data/scd_videos_tf/200_frames_pred/mobile_net_2/models/ConvNet'),
    )
