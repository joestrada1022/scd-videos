import json
from pathlib import Path


def create_data_set(source_data_views, dest_data_views, device_names):
    for file_path in source_data_views.glob('*'):
        with open(file_path) as f:
            source_data_set = json.load(f)
        dest_data_set = {x: source_data_set[x] for x in device_names}

        # Pick videos from native indoor scenario
        # for item in dest_data_set:
        #     dest_data_set[item] = [x for x in dest_data_set[item] if '_indoor_' in x]

        dest_data_views.mkdir(parents=True, exist_ok=True)
        with open(dest_data_views.joinpath(file_path.name), 'w+') as f:
            json.dump(dest_data_set, f, indent=2)

        # for index, item in enumerate(data_set,1):
        #     print(index, item)


if __name__ == '__main__':
    create_data_set(
        source_data_views=Path(r'/scratch/p288722/datasets/vision/bal_all_I_frames'),
        dest_data_views=Path(r'/scratch/p288722/datasets/vision/8_devices/bal_all_I_frames'),
        # device_names=('D01_Samsung_GalaxyS3Mini', 'D34_Apple_iPhone5')
        device_names=('D01_Samsung_GalaxyS3Mini', 'D03_Huawei_P9', 'D04_LG_D290', 'D07_Lenovo_P70A',
                      'D12_Sony_XperiaZ1Compact', 'D24_Xiaomi_RedmiNote3', 'D25_OnePlus_A3000', 'D34_Apple_iPhone5')
    )
