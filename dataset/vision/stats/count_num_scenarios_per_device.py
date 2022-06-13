from pathlib import Path


def count_num_scenarios_per_device(root_dir):
    devices = sorted(list(root_dir.glob('*')))
    devices_count_dict = {}
    for device in devices:
        devices_count_dict[device] = {
            'native': 0,
            'WA': 0,
            'YT': 0,
            'native-flat': 0,
            'native-indoor': 0,
            'native-outdoor': 0,
            'WA-flat': 0,
            'WA-indoor': 0,
            'WA-outdoor': 0,
            'YT-flat': 0,
            'YT-indoor': 0,
            'YT-outdoor': 0
        }
        for video in device.glob('*'):
            if 'WA' in video.name:
                devices_count_dict[device]['WA'] += 1
                if 'flat' in video.name:
                    devices_count_dict[device]['WA-flat'] += 1
                elif 'indoor' in video.name:
                    devices_count_dict[device]['WA-indoor'] += 1
                elif 'outdoor' in video.name:
                    devices_count_dict[device]['WA-outdoor'] += 1
            elif 'YT' in video.name:
                devices_count_dict[device]['YT'] += 1
                if 'flat' in video.name:
                    devices_count_dict[device]['YT-flat'] += 1
                elif 'indoor' in video.name:
                    devices_count_dict[device]['YT-indoor'] += 1
                elif 'outdoor' in video.name:
                    devices_count_dict[device]['YT-outdoor'] += 1
            else:
                devices_count_dict[device]['native'] += 1
                if 'flat' in video.name:
                    devices_count_dict[device]['native-flat'] += 1
                elif 'indoor' in video.name:
                    devices_count_dict[device]['native-indoor'] += 1
                elif 'outdoor' in video.name:
                    devices_count_dict[device]['native-outdoor'] += 1

    lines = ['device_name,native,WA,YT,native-flat,native-indoor,native-outdoor,WA-flat,'
             'WA-indoor,WA-outdoor,YT-flat,YT-indoor,YT-outdoor\n']
    for device in devices_count_dict:
        lines.append(device.name + ',' + ','.join(
            [str(devices_count_dict[device][x]) for x in devices_count_dict[device]]) + '\n')
    with open(r'videos_stats.csv', 'w+') as f:
        f.writelines(lines)


if __name__ == '__main__':
    path_to_root_dir_of_video_frames = Path(r'')  # update this field
    assert path_to_root_dir_of_video_frames.exists(), 'The input path to dataset with frames does not exists'
    count_num_scenarios_per_device(path_to_root_dir_of_video_frames)
