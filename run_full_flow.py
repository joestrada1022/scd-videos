"""
This file is used to run several scripts to run the train, validation, and test flows.
These scripts could be optionally be run from the command line with appropriate command line arguments.
"""
from run_evaluate import run_evaluate_flow
from run_train import run_train_flow
from utils.predict_utils.select_best_model import select_best_model
import tensorflow as tf

# To force eager execution of tf.data functions, please use `tf.data.experimental.enable_debug_mode()`
tf.data.experimental.enable_debug_mode()

def run_full_flow():
    # Set up the arguments
    num_frames = 12
    net = 'mobile'
    const_type = 'None'
    category = 'None'

    model_name = ''
    if net == 'mobile':
        model_name = f"MobileNet_{num_frames}"
    elif net == 'res':
        model_name = f"ResNet_{num_frames}"
    elif net == 'misl':
        model_name = f"MISLNet_{num_frames}"

    if const_type == 'derrick':
        model_name += '_Const'

    if category == 'native':
        model_name += '_na'
    elif category == 'whatsapp':
        model_name += '_wa'
    elif category == 'youtube':
        model_name += '_yt'

    # data set params
    all_I_frames_dir = "data"  # path to all-frames dir of the vision dataset
    all_frames_dir = "/data/p288722/datasets/vision/all_frames"  # path to I-frames dir of vision or qufvd dataset

    dataset_params = ['--dataset_name', 'qufvd',  # vision or qufvd
                      '--frame_selection', 'equally_spaced',
                      '--frame_type', 'I',  # I or all
                      '--fpv', f'{num_frames}',  # number of Frames to use Per Video (fpv)
                      '--height', '480',  # center crop dimensions
                      '--width', '800',  # center crop dimensions
                      '--all_I_frames_dir', all_I_frames_dir]  # change to all_frames_dir if using that

    # add the absolute path to the results dir
    results_dir = r'results'
    gpu_id = 0  # set the CUDA GPU ID

    # Step 1 - Run train
    args = ['--category', category, '--net_type', net, '--epochs', '20', '--lr', '0.1', '--batch_size', '32',
            '--use_pretrained', '1', '--gpu_id', gpu_id,
            '--const_type', const_type,
            '--model_name', model_name,
            '--global_results_dir', f'{results_dir}/{num_frames}_frames/{net}_net']
    run_train_flow(dataset_params + args)

    # Step 2 - Run validation
    args = ['--eval_set', 'val', '--batch_size', '64', '--gpu_id', gpu_id, '--suffix', f"{num_frames}_frames_val",
            '--input_dir', f'{results_dir}/{num_frames}_frames/{net}_net/models/{model_name}']
    run_evaluate_flow(dataset_params + args)

    # Step 3 - Select the best model based on the results of the validation set
    args = ['--val_summary', f'{results_dir}/{num_frames}_frames/{net}_net/models/{model_name}/'
                             f'predictions_{num_frames}_frames_val/videos/V_prediction_stats.csv']
    select_best_model(args)

    # Step 4 - Run test
    args = ['--eval_set', 'test', '--batch_size', '64', '--gpu_id', gpu_id, '--suffix', f"{num_frames}_frames",
            '--input_dir', f'{results_dir}/{num_frames}_frames_pred/{net}_net/models/{model_name}']
    run_evaluate_flow(dataset_params + args)

    print(f'Run finished. '
          f'Check the output files under '
          f'"{results_dir}/{num_frames}_frames_pred/{net}_net/models/{model_name}/predictions_{num_frames}_frames" '
          f'for the results.')


if __name__ == '__main__':
    run_full_flow()
