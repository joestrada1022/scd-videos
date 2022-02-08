import json
from pathlib import Path

if __name__ == '__main__':
    root_dir = Path(r'/data/p288722/datasets/vision/I_frame_splits')
    for file in root_dir.glob('*/*.json'):
        with open(file) as f:
            json_data = json.load(f)
        for device in json_data:
            json_data[device] = [x.replace('scratch', 'data') for x in json_data[device]]
        with open(file, 'w+') as f:
            json.dump(json_data, f, indent=2)
