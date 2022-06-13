import json
from pathlib import Path


def count_train_val_test_videos():
    dataset_splits = Path(r'./../split/')
    for split in ['train', 'val', 'test']:
        with open(dataset_splits.joinpath(f'{split}.json')) as f:
            dataset = json.load(f)
        videos = {Path(y).parent.name for _, x in dataset.items() for y in x}
        frames = [y for _, x in dataset.items() for y in x]
        print(f'Number of videos in {split}: {len(videos)}')
        print(f'Number of video frames in {split}: {len(frames)}')


if __name__ == '__main__':
    count_train_val_test_videos()
