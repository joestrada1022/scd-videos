from pathlib import Path


def process_data(num_frames, root_dest_dir, source_results_dir):
    for frame_count in num_frames:
        dest_dir = root_dest_dir.joinpath(f'predictions_{frame_count}_frames')
        dest_dir.mkdir(parents=True, exist_ok=True)

        source_frames_file = list(source_results_dir.joinpath('frames').glob('fm-e*'))[0]
        dest_frames_dir = dest_dir.joinpath(f'frames')
        dest_frames_dir.mkdir(parents=True, exist_ok=True)
        dest_frames_file = dest_dir.joinpath(f'frames/{source_frames_file.name}')

        dest_lines = []
        dict_videos = {}
        with open(source_frames_file, 'r') as f:
            dest_lines.append(f.__next__())  # Header
            for line in f:
                video_name = line.split(',')[0].split('-')[0]
                if video_name in dict_videos:
                    dict_videos[video_name] += [line]
                else:
                    dict_videos[video_name] = [line]

        for video in dict_videos:
            frame_predictions = sorted(dict_videos[video])
            num_source_frames = len(frame_predictions)
            for index in range(0, num_source_frames, int(num_source_frames/frame_count)):
                dest_lines.append(dict_videos[video][index])

        with open(dest_frames_file, 'w+') as f:
            f.writelines(dest_lines)


if __name__ == '__main__':
    process_data(
        num_frames=[1, 5, 10, 20, 50, 100, 400, 800],
        root_dest_dir=Path(r'/scratch/p288722/runtime_data/scd_videos_tf/200_frames_pred/mobile_net_2/models/ConvNet'),
        source_results_dir=Path(r'/scratch/p288722/runtime_data/scd_videos_tf/200_frames_pred/mobile_net_2/models/ConvNet/predictions_all_frames')
    )
