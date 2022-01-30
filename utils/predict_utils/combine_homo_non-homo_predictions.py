import pandas as pd

from utils.predict_utils import VideoPredictor


def combine_predictions():
    df_homo = pd.read_csv(r'/scratch/p288722/runtime_data/scd_videos_first_revision/03_2stream/50_frames_pred/'
                          r'mobile_net/models/MobileNet_homo/predictions_50_frames/frames/fm-e00019_F_predictions.csv')
    df_non_homo = pd.read_csv(r'/scratch/p288722/runtime_data/scd_videos_first_revision/03_2stream/50_frames_pred/'
                              r'mobile_net/models/MobileNet_non_homo/predictions_50_frames/frames/'
                              r'fm-e00019_F_predictions.csv')

    # Combine the frame predictions
    df_combined = df_homo + df_non_homo  # fixme

    # Generate & save video-level predictions
    output_file = r'/scratch/p288722/runtime_data/scd_videos_first_revision/03_2stream/50_frames_pred/mobile_net/' \
                  r'models/MobileNet_combined/combined_video_stats.csv'
    VideoPredictor()._predict_and_save(df_combined, output_file)


if __name__ == '__main__':
    combine_predictions()
