from pathlib import Path
from sklearn import metrics


def video_predictions_to_model_level_results(video_predictions_file, devices_to_models_map):
    with open(video_predictions_file, newline='') as f:
        header = f.readline().strip().split(',')
        true_class_index = header.index('true_class')
        predicted_class_index = header.index('top1_class')
        gt, pr = [], []
        for row in f:
            values = row.strip().split(',')
            gt.append(int(values[true_class_index]))
            pr.append(int(values[predicted_class_index]))
    device_level_acc = metrics.accuracy_score(gt, pr)
    print(f'Device-level accuracy: {device_level_acc}')

    model_gt = [devices_to_models_map[x] for x in gt]
    model_pr = [devices_to_models_map[x] for x in pr]
    model_level_acc = metrics.accuracy_score(model_gt, model_pr)
    print(f'Model-level accuracy: {model_level_acc}')


if __name__ == '__main__':
    num_devices = 28
    d2m_map = {x: x for x in range(num_devices)}
    DEVICES = ['D01_Samsung_GalaxyS3Mini',
               'D02_Apple_iPhone4s',
               'D03_Huawei_P9',
               'D04_LG_D290',
               'D05_Apple_iPhone5c',
               'D06_Apple_iPhone6',
               'D07_Lenovo_P70A',
               'D08_Samsung_GalaxyTab3',
               'D09_Apple_iPhone4',
               'D10_Apple_iPhone4s',
               'D11_Samsung_GalaxyS3',
               'D12_Sony_XperiaZ1Compact',
               'D14_Apple_iPhone5c',
               'D15_Apple_iPhone6',
               'D16_Huawei_P9Lite',
               'D18_Apple_iPhone5c',
               'D19_Apple_iPhone6Plus',
               'D24_Xiaomi_RedmiNote3',
               'D25_OnePlus_A3000',
               'D26_Samsung_GalaxyS3Mini',
               'D27_Samsung_GalaxyS5',
               'D28_Huawei_P8',
               'D29_Apple_iPhone5',
               'D30_Huawei_Honor5c',
               'D31_Samsung_GalaxyS4Mini',
               'D32_OnePlus_A3003',
               'D33_Huawei_Ascend',
               'D34_Apple_iPhone5']
    d2m_map[DEVICES.index('D26_Samsung_GalaxyS3Mini')] = DEVICES.index('D01_Samsung_GalaxyS3Mini')
    d2m_map[DEVICES.index('D10_Apple_iPhone4s')] = DEVICES.index('D02_Apple_iPhone4s')
    d2m_map[DEVICES.index('D14_Apple_iPhone5c')] = DEVICES.index('D05_Apple_iPhone5c')
    d2m_map[DEVICES.index('D18_Apple_iPhone5c')] = DEVICES.index('D05_Apple_iPhone5c')
    d2m_map[DEVICES.index('D15_Apple_iPhone6')] = DEVICES.index('D06_Apple_iPhone6')
    d2m_map[DEVICES.index('D34_Apple_iPhone5')] = DEVICES.index('D29_Apple_iPhone5')

    video_predictions_to_model_level_results(
        video_predictions_file=Path(r'/scratch/p288722/runtime_data/scd-videos/i_frames/all_frames_28d_64_pred/mobile_net/models/h0_ConvNet/predictions_all_frames/videos/fm-e00010_V_predictions.csv'),
        devices_to_models_map=d2m_map,
    )
