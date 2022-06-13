import os

import numpy as np
import pandas as pd


class VideoPredictor:

    def __init__(self, model_file_name=None, result_dir=None, dataset_name=None):
        self.model_file_name = model_file_name
        self.result_dir = result_dir
        self.dataset_name = dataset_name
        # Top k predicted classes will be saved in csv-file
        self.top_k_predictions = 3

    def start(self, frame_prediction_file):
        # Read file with frame predictions
        df_frame_predictions = pd.read_csv(frame_prediction_file)
        if len(df_frame_predictions) == 0:
            print(f"No frame predictions found in {str(frame_prediction_file)}")
            return
        return self._predict_and_save(df_frame_predictions)

    def _predict_and_save(self, df_frame_predictions, output_file=None):
        # Predict videos
        video_predictions = self.__get_predictions(df_frame_predictions=df_frame_predictions)

        # Create dataframe based on video_predictions
        df_results = pd.DataFrame(video_predictions, columns=self.__get_columns(self.top_k_predictions))

        if not output_file:
            output_file = self.get_output_file()
        df_results.to_csv(output_file, index=False)

        return output_file

    def get_output_file(self):
        # Remove .h5 extension
        output_file_name = self.model_file_name.split('.')[0]
        output_file = os.path.join(self.result_dir, f"{output_file_name}_V_predictions.csv")
        return output_file

    @staticmethod
    def get_platform(filename):
        if "YT" in filename:
            return "YT"
        elif "WA" in filename:
            return "WA"
        else:
            return "original"

    def __get_predictions(self, df_frame_predictions):
        # Determine unique videos
        if self.dataset_name == 'vision':
            videos = df_frame_predictions["File"].str.split("-").str[0].unique()
        elif self.dataset_name == 'qufvd':
            df_frame_predictions["File"] = df_frame_predictions["File"].str.replace('(', '<').str.replace(')', '>')
            videos = df_frame_predictions["File"].str.split("-").str[-5:-2].str.join('-').unique()
        else:
            raise ValueError(f'Invalid dataset name - {self.dataset_name}')

        print(f"Total number of videos to classify: {len(videos)}.")

        max_softmax = [[float(y) for y in x.strip('[]').split()] for x in df_frame_predictions['Softmax Scores']]
        max_softmax = [float(max(np.array(x))) for x in max_softmax]
        df_frame_predictions = df_frame_predictions.assign(max_softmax=max_softmax)

        video_predictions = []
        for video in videos:
            # Get frame predictions for current video
            df_video_predictions = df_frame_predictions[df_frame_predictions["File"].str.contains(video)]
            # Class index
            class_idx = df_video_predictions["True Label"].iloc[0]

            # Use all frames to classify a video
            n_frames = len(df_video_predictions)

            # Majority Vote
            df_device_vote = df_video_predictions["Predicted Label"].value_counts().rename_axis('class').reset_index(
                name='vote_count')

            # Select top 3 devices
            df_top_votes = df_device_vote.nlargest(self.top_k_predictions, ['vote_count'])

            # Get platform (i.e. original/Whatsapp (WA)/Youtube (YT) by using filename of first row.
            platform = self.get_platform(df_video_predictions["File"].iloc[0])

            video_result = [video, platform, n_frames, class_idx]
            for row_idx, row in df_top_votes.iterrows():
                label = int(row["class"])
                vote_count = int(row["vote_count"])
                confidence = round(vote_count / n_frames, 3)

                video_result.append(label)
                video_result.append(confidence)

            # If all frames predicted the same label, we only have one row in df_top_votes.
            if len(df_top_votes) < self.top_k_predictions:
                i = len(df_top_votes)
                while i < 3:
                    video_result.append(None)
                    video_result.append(None)
                    i += 1

            video_loss = df_video_predictions["Loss"].mean()
            video_result.append(video_loss)

            video_predictions.append(video_result)

        return video_predictions

    @staticmethod
    def __get_columns(k):
        columns = ['filename', 'platform', 'n_frames_for_prediction', 'true_class']

        for i in range(k):
            # e.g. top1_class
            columns.append(f"top{i + 1}_class")
            # e.g. top1_conf
            columns.append(f"top{i + 1}_conf")

        columns.append('Loss')

        return columns
