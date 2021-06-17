import argparse
import os
from pathlib import Path

from dataset.data_factory import DataFactory
from utils.predict_utils.frame_prediction_statistics import FramePredictionStatistics
from utils.predict_utils.frame_prediction_visualization import FramePredictionVis
from utils.predict_utils.predict_frames import FramePredictor
from utils.predict_utils.predict_videos import VideoPredictor
from utils.predict_utils.video_prediction_statistics import VideoPredictionStatistics
from utils.predict_utils.video_prediction_visualization import VideoPredictionVis


def get_dataset_path(data_set):
    return os.path.join("/data/p288722/from_f118170/vbdi", data_set)


def get_devices(data_set_path):
    test_dir = os.path.join(data_set_path, "test")
    devices = [item for item in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, item))]
    return devices


def get_models_files(input_dir, models):
    # Get all files (i.e. models) from input directory
    files_list = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    files_list = sorted(files_list)
    print(f"Found {len(files_list)} files in {input_dir}: {files_list}")

    if models:
        print(f"'Models' argument is set. Only the following models will be evaluated: {models.split(',')}")
        model_split = models.split(',')
        return [file for file in files_list if file in model_split]

    return files_list


def get_result_dir(input_dir, primary_suffix, secondary_suffix):
    if primary_suffix:
        output_dir = os.path.join(input_dir, f"predictions_{primary_suffix}")
    elif secondary_suffix:
        output_dir = os.path.join(input_dir, f"predictions_{secondary_suffix}")
    else:
        output_dir = os.path.join(input_dir, f"predictions")

    # Create output directory is not exists
    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(e)
            raise ValueError(f"Error during creation of output directory")

    frames_output_dir = os.path.join(output_dir, "frames")
    videos_output_dir = os.path.join(output_dir, "videos")
    plots_output_dir = os.path.join(output_dir, "plots")
    if not os.path.isdir(frames_output_dir):
        os.makedirs(frames_output_dir)
    if not os.path.isdir(videos_output_dir):
        os.makedirs(videos_output_dir)
    if not os.path.isdir(plots_output_dir):
        os.makedirs(plots_output_dir)

    return output_dir, frames_output_dir, videos_output_dir, plots_output_dir


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Make predictions with signature network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to directory consisting of .h5-models (to use for predicting)')
    parser.add_argument('--models', type=str, required=False,
                        help='Models within input dir (*.h5) to evaluate. Separate models by a ",". ')
    parser.add_argument('--constrained', type=int, required=False, default=1,
                        help='Constrained layer included (0=no, 1=yes)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use to make predictions')
    parser.add_argument('--batch_size', type=int, required=False, default=64, help='Batch size')
    parser.add_argument('--height', type=int, required=False, default=480,
                        help='Height of CNN input dimension [default=480]')
    parser.add_argument('--width', type=int, required=False, default=800,
                        help='Width of CNN input dimension [default=800]')
    parser.add_argument('--category', type=str, required=False, help='enter "native", "whatsapp", or "youtube"')
    parser.add_argument('--suffix', type=str, required=False, help='enter suffix string for the predictions folder')
    parser.add_argument('--gpu_id', type=int, required=False, default=0,
                        help='Choose the available GPU devices')
    parser.add_argument('--epoch', type=int, required=False, default=-1, help='Choose the epoch')

    args = parser.parse_args()
    model_input_dir = args.input_dir
    models_to_process = args.models
    dataset_path = get_dataset_path(args.dataset)
    batch_size = args.batch_size
    height = args.height
    width = args.width
    constrained = args.constrained == 1
    category = args.category
    gpu_id = args.gpu_id
    suffix = args.suffix
    epoch = args.epoch

    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices):
        tf.config.set_visible_devices(physical_devices[gpu_id], device_type='GPU')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    model_name = model_input_dir.split(os.path.sep)[-1]
    model_files = get_models_files(model_input_dir, models_to_process)
    _, frames_res_dir, videos_res_dir, plots_res_dir = get_result_dir(model_input_dir,
                                                                      primary_suffix=category,
                                                                      secondary_suffix=suffix)
    if epoch > 0:
        model_files = [x for x in model_files if str(epoch).zfill(5) in x]
    print(f"Found {len(model_files)} files for model {model_name}")

    # # Choose the epoch with the best accuracy to continue with predictions
    # predictions_file = Path(model_input_dir).joinpath('predictions/videos/V_prediction_stats.csv')
    # with open(predictions_file, 'r') as f:
    #     best_model, best_acc = None, 0.0
    #     f.__next__()  # Skip Header
    #     f.__next__()  # Skip First row
    #     for line in f:
    #         curr_model, curr_acc = line.split(',')[:2]
    #         if float(curr_acc) > best_acc:
    #             best_model = curr_model + '.h5'
    #             best_acc = float(curr_acc)
    #     model_files = [best_model]

    for model_file in model_files:

        # quick test to determine if both frame and video results are already computed:
        frames_results = Path(FramePredictor(model_dir=model_input_dir, input_dir=dataset_path,
                                             model_file_name=model_file, result_dir=frames_res_dir).get_output_file())
        videos_results = Path(VideoPredictor(model_file_name=model_file, result_dir=videos_res_dir).get_output_file())
        if not (frames_results.exists() and videos_results.exists()):

            print(f"{model_file} | Start prediction process")
            # Predict Frames
            frame_predictor = FramePredictor(model_dir=model_input_dir, model_file_name=model_file,
                                             result_dir=frames_res_dir,
                                             input_dir=dataset_path)
            if not Path(frame_predictor.get_output_file()).exists():
                # Re-create dataset for each model to make sure the test-generator does not mess up.
                dataset = DataFactory(input_dir=dataset_path, batch_size=batch_size, height=height, width=width)
                filename_ds, val_ds = dataset.get_tf_val_data(category=category)

                # List containing only the file names of items in test set
                test_ds_filenames = list(filename_ds.as_numpy_iterator())

                print(f"{model_file} | Start predicting frames")

                frame_pred_file = frame_predictor.start(test_ds=val_ds, filenames=test_ds_filenames)
                print(f"{model_file} | Predicting frames completed")

            # Predict Videos which is based on the predicted frames
            video_predictor = VideoPredictor(model_file_name=model_file, result_dir=videos_res_dir)
            if not Path(video_predictor.get_output_file()).exists():
                print(f"{model_file} | Start predicting videos")
                # Use frame prediction file as input
                video_pred_file = video_predictor.start(frame_predictor.get_output_file())
                print(f"{model_file} | Predicting videos completed")

    print(f"Creating Statistics and Visualizations ...")
    # Create Frame Prediction Statistics
    fps = FramePredictionStatistics(result_dir=frames_res_dir)
    frame_stats = fps.start()
    print(f"Frame Prediction Statistics Completed")

    vps = VideoPredictionStatistics(result_dir=videos_res_dir)
    video_stats = vps.start()
    print(f"Video Prediction Statistics Completed")

    fpv = FramePredictionVis(result_dir=plots_res_dir)
    fpv.start(frame_stats)
    print(f"Frame Prediction Visualization Completed")

    vpv = VideoPredictionVis(result_dir=plots_res_dir, model_name=model_name)
    vpv.start(video_stats)
    print(f"Video Prediction Visualization Completed")
