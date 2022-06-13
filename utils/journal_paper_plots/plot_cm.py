import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_model_level_accuracy_vision(true_labels, pred_labels):
    devices = ['D01_Samsung_GalaxyS3Mini',
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
               'D34_Apple_iPhone5'
               ]
    models = sorted(set([x[4:] for x in devices]))
    device_to_model_map = {index: models.index(x[4:]) for index, x in enumerate(devices)}

    ground_truths = pd.Series(true_labels, copy=True)
    predictions = pd.Series(pred_labels, copy=True)

    for index, (tr, pr) in enumerate(zip(true_labels, pred_labels)):
        ground_truths[index] = device_to_model_map[tr]
        predictions[index] = device_to_model_map[pr]

    accuracy = sum(ground_truths == predictions) / len(true_labels)
    return accuracy


def get_model_level_accuracy_qufvd(true_labels, pred_labels):
    print(' ')
    gt = true_labels // 2
    pd = pred_labels // 2
    model_level_acc = sum(gt == pd) / len(gt)
    return model_level_acc


def create_cm_normalized(input_file, class_names, scenario=None, platform=None):
    # Credits to Guru
    from sklearn.metrics import confusion_matrix
    import seaborn as sn

    df = pd.read_csv(input_file)
    # e.g. flat/indoor/outdoor
    if scenario is not None:
        df = df[df["filename"].str.contains(scenario)]

    # e.g. original/WA/YT
    if platform is not None:
        if platform == "original":
            df = df[~df["filename"].str.contains("YT") & ~df["filename"].str.contains("WA")]
        else:
            df = df[df["filename"].str.contains(platform)]

    true_labels = df['true_class']
    pred_labels = df['top1_class']

    device_level_accuracy = sum(true_labels == pred_labels) / len(true_labels)
    # model_level_accuracy = get_model_level_accuracy_vision(true_labels, pred_labels)
    # stabilized_videos_accuracy = get_stabilized_videos_accuracy_vision(true_labels, pred_labels)
    model_level_accuracy_qufvd = get_model_level_accuracy_qufvd(true_labels, pred_labels)
    print(f'Device-level accuracy : {device_level_accuracy}')
    print(f'Model-level accuracy on QUFVD: {model_level_accuracy_qufvd}')
    # print(f'Model-level accuracy on VISION : {model_level_accuracy}')
    # print(f'Model-level accuracy on VISION : {stabilized_videos_accuracy}')

    cm_matrix = confusion_matrix(true_labels, pred_labels)

    # Creating labels for the plot
    x_ticks = [''] * len(cm_matrix)
    y_ticks = [''] * len(cm_matrix)
    for i in np.arange(0, len(cm_matrix), 1):
        x_ticks[i] = str(i + 1)
        y_ticks[i] = str(i + 1)

    colorbar_lbl = 'Normalized num videos per class'
    title = f"Overall Accuracy"

    if platform is not None:
        title = f"{title} - {platform}"
    if scenario is not None:
        title = f"{scenario.title()} Scenario"

    plt.figure(figsize=(8.2, 7.2), dpi=300)
    sn.set(font_scale=1.7)  # for label size

    # From the sklearn documentation (plot example)
    # Note: possible division by zero error
    norm_cm = cm_matrix.astype('float') / cm_matrix.sum(axis=1)[:, np.newaxis]
    # Round to 2 decimals
    norm_cm = np.around(norm_cm, 2)

    df_cm = pd.DataFrame(norm_cm, class_names, class_names)
    sn.heatmap(df_cm, annot=False, square=True, cmap="YlGnBu",
               cbar_kws={'label': colorbar_lbl},
               vmin=0, vmax=1)
    plt.yticks(rotation=0)
    plt.title(title, pad=30, fontsize=30)
    plt.ylabel('True Class', labelpad=10)
    plt.xlabel('Predicted Class', labelpad=10)
    plt.tight_layout()

    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


if __name__ == '__main__':
    # QUFVD data set
    create_cm_normalized(
        input_file=r'/scratch/p288722/runtime_data/scd_videos_first_revision/14_qufvd/all_frames_pred/mobile_net/'
                   r'models/MobileNet_all_I_frames_ccrop_run1/predictions_all_frames/videos/'
                   r'fm-e00020_V_predictions.csv',
        class_names=list(range(1, 21)),
        scenario=None,
        platform=None,
    )

    # VISION data set
    # The code needs to be adapted for getting the confusion matrix on the VISION data set
    # Check the commented out lines of code in create_cm_normalized(...)
    # Populate scenario and platform arguments as appropriate

