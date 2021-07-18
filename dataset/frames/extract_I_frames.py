import json
import subprocess
from pathlib import Path

from tqdm import tqdm


def create_I_frame_dataset(all_frames_views_dir, all_I_frames_views_dir, dataset_root_dir):
    module = 'module load FFmpeg/4.2.2-GCCcore-9.3.0'  # load the ffmpeg module on the Peregrine
    for split in all_frames_views_dir.glob('*'):
        with open(split) as f:
            all_frames_dict = json.load(f)

        all_frames_I_dict = {}
        for device in tqdm(all_frames_dict):
            all_frames_I_dict[device] = []
            videos = set([Path(x).parent.name for x in all_frames_dict[device]])
            for video in videos:
                if len(list(dataset_root_dir.glob(rf'*/videos/*/{video}*'))) != 1:
                    raise ValueError('Debug and check, why this is not equal to 1')
                video_path = list(dataset_root_dir.glob(rf'*/videos/*/{video}*'))[0]
                cmd = f"{module}\nffprobe {str(video_path)} -show_frames | grep -E 'pict_type'"
                frame_types = subprocess.run(cmd, shell=True, capture_output=True).stdout.decode('ascii').split('\n')
                frame_ids = set([str(idx + 1).zfill(5) for idx, x in enumerate(frame_types) if x == 'pict_type=I'])

                all_frame_paths = [Path(x) for x in all_frames_dict[device] if Path(x).parent.name == video]
                I_frame_paths = sorted([str(x) for x in all_frame_paths if x.name.split('-')[1][:5] in frame_ids])
                all_frames_I_dict[device].extend(I_frame_paths)

        all_I_frames_views_dir.mkdir(parents=True, exist_ok=True)
        with open(all_I_frames_views_dir.joinpath(split.name), 'w+') as f:
            json.dump(all_frames_I_dict, f, indent=2)


if __name__ == '__main__':
    create_I_frame_dataset(
        all_frames_views_dir=Path(r'/scratch/p288722/datasets/vision/bal_all_frames'),
        all_I_frames_views_dir=Path(r'/scratch/p288722/datasets/vision/bal_all_I_frames'),
        dataset_root_dir=Path(r'/data/p288722/VISION/dataset')
    )
