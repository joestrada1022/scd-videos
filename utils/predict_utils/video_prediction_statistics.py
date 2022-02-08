import os

import numpy as np
import pandas as pd


class VideoPredictionStatistics:

    def __init__(self, result_dir):
        self.result_dir = result_dir

    def start(self):
        files = self.__get_video_files()
        if len(files) == 0:
            raise ValueError(f"No csv-files found in {self.result_dir}")
        else:
            print(f"Found {len(files)} files in {self.result_dir}.")

        global_video_statistics = []

        for file in files:
            file_statistics = self.__get_statistics(file)
            global_video_statistics.append(file_statistics)

        df_global_statistics = pd.DataFrame(global_video_statistics, columns=self.__get_columns())

        df_global_statistics.loc[-1] = self.__empty_row()  # adding a row
        df_global_statistics.index = df_global_statistics.index + 1  # shifting index
        df_global_statistics = df_global_statistics.sort_index()  # sorting by index

        output_file = os.path.join(self.result_dir, f"V_prediction_stats.csv")
        df_global_statistics.to_csv(output_file, index=False)

        return output_file

    def __get_video_files(self):
        # Get all csv-files containing video predictions from input directory
        files = [f for f in os.listdir(self.result_dir) if os.path.isfile(os.path.join(self.result_dir, f))
                 and f.endswith("V_predictions.csv")]
        files = sorted(files)

        return files

    @staticmethod
    def __get_native_scenario_statistics(df_video_predictions):
        # Videos of flat objects (e.g. walls or skies)
        df_flat = df_video_predictions[df_video_predictions["filename"].str.contains("flat")]
        # Videos of indoor rooms for example
        df_indoor = df_video_predictions[df_video_predictions["filename"].str.contains("indoor")]
        # Videos of outdoor objects
        df_outdoor = df_video_predictions[df_video_predictions["filename"].str.contains("outdoor")]

        df_native_flat = df_flat[df_flat["platform"] == "original"]
        df_native_indoor = df_indoor[df_indoor["platform"] == "original"]
        df_native_outdoor = df_outdoor[df_outdoor["platform"] == "original"]

        # Accuracy for different scenarios
        acc_native_flat = round(df_native_flat["correct"].mean(), 3)
        acc_native_indoor = round(df_native_indoor["correct"].mean(), 3)
        acc_native_outdoor = round(df_native_outdoor["correct"].mean(), 3)

        vc_native_flat = len(df_native_flat)
        vc_native_indoor = len(df_native_indoor)
        vc_native_outdoor = len(df_native_outdoor)

        return [acc_native_flat, acc_native_indoor, acc_native_outdoor,
                vc_native_flat, vc_native_indoor, vc_native_outdoor]

    @staticmethod
    def __get_scenario_statistics(df_video_predictions):
        # Videos of flat objects (e.g. walls or skies)
        df_flat = df_video_predictions[df_video_predictions["filename"].str.contains("flat")]
        # Videos of indoor rooms for example
        df_indoor = df_video_predictions[df_video_predictions["filename"].str.contains("indoor")]
        # Videos of outdoor objects
        df_outdoor = df_video_predictions[df_video_predictions["filename"].str.contains("outdoor")]

        # Accuracy for different scenarios
        acc_flat = round(df_flat["correct"].mean(), 3)
        acc_indoor = round(df_indoor["correct"].mean(), 3)
        acc_outdoor = round(df_outdoor["correct"].mean(), 3)

        # Confidence per scenario
        conf_flat = round(df_flat["top1_conf"].mean(), 3)
        conf_indoor = round(df_indoor["top1_conf"].mean(), 3)
        conf_outdoor = round(df_outdoor["top1_conf"].mean(), 3)

        # Loss per scenario
        loss_flat = round(1 - conf_flat, 3)
        loss_indoor = round(1 - conf_indoor, 3)
        loss_outdoor = round(1 - conf_outdoor, 3)

        # Video count per scenario
        vc_flat = len(df_flat)
        vc_indoor = len(df_indoor)
        vc_outdoor = len(df_outdoor)

        return [acc_flat, acc_indoor, acc_outdoor,
                conf_flat, conf_indoor, conf_outdoor,
                loss_flat, loss_indoor, loss_outdoor,
                vc_flat, vc_indoor, vc_outdoor]

    @staticmethod
    def __get_platform_statistics(df_video_predictions):
        # Averages per platform, i.e. 'original', 'YT', 'WA'
        df_platform_results = df_video_predictions.groupby("platform").agg(
            acc=("correct", "mean"),
            conf=("top1_conf", "mean"),
            count=("platform", "count")
        )

        # Reset the index. Otherwise, the platform will function as index
        df_platform_results = df_platform_results.reset_index()

        df_WA = df_platform_results[df_platform_results["platform"] == "WA"]
        df_YT = df_platform_results[df_platform_results["platform"] == "YT"]
        df_original = df_platform_results[df_platform_results["platform"] == "original"]

        # Accuracy per platform
        acc_WA = round(df_WA["acc"].values[0], 3) if len(df_WA.index) > 0 else 0
        acc_YT = round(df_YT["acc"].values[0], 3) if len(df_YT.index) > 0 else 0
        acc_original = round(df_original["acc"].values[0], 3) if len(df_original.index) > 0 else 0

        # Confidence per platform
        conf_WA = round(df_WA["conf"].values[0], 3) if len(df_WA.index) > 0 else 0
        conf_YT = round(df_YT["conf"].values[0], 3) if len(df_YT.index) > 0 else 0
        conf_original = round(df_original["conf"].values[0], 3) if len(df_original.index) > 0 else 0

        # Loss per platform
        loss_WA = round(1 - conf_WA, 3)
        loss_YT = round(1 - conf_YT, 3)
        loss_original = round(1 - conf_original, 3)

        # Video count per platform
        vc_WA = df_WA["count"].values[0] if len(df_WA.index) > 0 else 0
        vc_YT = df_YT["count"].values[0] if len(df_YT.index) > 0 else 0
        vc_original = df_original["count"].values[0] if len(df_original.index) > 0 else 0

        return [acc_WA, acc_YT, acc_original,
                conf_WA, conf_YT, conf_original,
                loss_WA, loss_YT, loss_original,
                vc_WA, vc_YT, vc_original]

    @staticmethod
    def __get_whatsapp_youtube_scenario_statistics(df_video_predictions):

        df_flat = df_video_predictions[df_video_predictions["filename"].str.contains("flat")]
        df_indoor = df_video_predictions[df_video_predictions["filename"].str.contains("indoor")]
        df_outdoor = df_video_predictions[df_video_predictions["filename"].str.contains("outdoor")]

        df_whatsapp_flat = df_flat[df_flat["platform"] == "WA"]
        df_whatsapp_indoor = df_indoor[df_indoor["platform"] == "WA"]
        df_whatsapp_outdoor = df_outdoor[df_outdoor["platform"] == "WA"]

        acc_whatsapp_flat = round(df_whatsapp_flat["correct"].mean(), 3)
        acc_whatsapp_indoor = round(df_whatsapp_indoor["correct"].mean(), 3)
        acc_whatsapp_outdoor = round(df_whatsapp_outdoor["correct"].mean(), 3)

        df_youtube_flat = df_flat[df_flat["platform"] == "YT"]
        df_youtube_indoor = df_indoor[df_indoor["platform"] == "YT"]
        df_youtube_outdoor = df_outdoor[df_outdoor["platform"] == "YT"]

        acc_youtube_flat = round(df_youtube_flat["correct"].mean(), 3)
        acc_youtube_indoor = round(df_youtube_indoor["correct"].mean(), 3)
        acc_youtube_outdoor = round(df_youtube_outdoor["correct"].mean(), 3)

        return [acc_whatsapp_flat, acc_whatsapp_indoor, acc_whatsapp_outdoor,
                acc_youtube_flat, acc_youtube_indoor, acc_youtube_outdoor]

    def __get_statistics(self, file):
        file_path = os.path.join(self.result_dir, file)
        df_video_predictions = pd.read_csv(file_path)

        # Add column 'correct' with value 1 if prediction == true class, else 0.
        df_video_predictions["correct"] = np.where(
            df_video_predictions["true_class"] == df_video_predictions["top1_class"], 1, 0)

        # Averages total
        grand_acc = round(df_video_predictions["correct"].mean(), 4)
        grand_conf = round(df_video_predictions["top1_conf"].mean(), 4)
        grand_loss = round(df_video_predictions["Loss"].mean(), 4)

        filename = file
        filename_split = file.split("_")
        for split in filename_split:
            if "fm-" in split:
                filename = split

        result = [filename, grand_acc, grand_conf, grand_loss]
        # Get statistics per platform
        platform_statistics = self.__get_platform_statistics(df_video_predictions)
        # Append statistics to result set
        result.extend(platform_statistics)
        # Get statistics per scenario
        scenario_statistics = self.__get_scenario_statistics(df_video_predictions)
        # Append statistics to result set
        result.extend(scenario_statistics)

        # Get statistics per scenario for native/original videos only
        native_scenario_statistics = self.__get_native_scenario_statistics(df_video_predictions)
        # Append statistics to result set
        result.extend(native_scenario_statistics)

        # Get global metrics as well as metrics per scenario.
        # Metrics include precision, recall and f1
        metrics = self.__get_metrics(df_video_predictions)
        # Append metrics to result set
        result.extend(metrics)

        # Get statistics per scenario for whatsapp/youtube videos
        whatsapp_youtube_scenario_statistics = self.__get_whatsapp_youtube_scenario_statistics(df_video_predictions)
        # Append statistics to result set
        result.extend(whatsapp_youtube_scenario_statistics)

        return result

    @staticmethod
    def __get_metrics(df):
        from sklearn.metrics import precision_recall_fscore_support
        result = []

        # Calculate global values first
        true_labels = df['true_class']
        predicted_labels = df['top1_class']

        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average="weighted")
        precision = round(precision, 2)
        recall = round(recall, 2)
        f1 = round(f1, 2)
        result.extend([precision, recall, f1])

        scenarios = ['flat', 'indoor', 'outdoor']
        for scenario in scenarios:
            df_scenario = df[df["filename"].str.contains(scenario)]
            true_labels = df_scenario['true_class']
            pred_labels = df_scenario['top1_class']

            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average="weighted")
            precision = round(precision, 2)
            recall = round(recall, 2)
            f1 = round(f1, 2)

            result.extend([precision, recall, f1])

        return result

    @staticmethod
    def __get_columns():
        # do NOT change the order of the columns cause the entire construction is build upon this order.
        return ["model", "acc", "conf", "loss",
                "acc_platform_WA", "acc_platform_YT", "acc_platform_original",
                "conf_platform_WA", "conf_platform_YT", "conf_platform_original",
                "loss_platform_WA", "loss_platform_YT", "loss_platform_original",
                "vc_platform_WA", "vc_platform_YT", "vc_platform_original",
                "acc_scenario_flat", "acc_scenario_indoor", "acc_scenario_outdoor",
                "conf_scenario_flat", "conf_scenario_indoor", "conf_scenario_outdoor",
                "loss_scenario_flat", "loss_scenario_indoor", "loss_scenario_outdoor",
                "vc_scenario_flat", "vc_scenario_indoor", "vc_scenario_outdoor",
                "acc_native_flat", "acc_native_indoor", "acc_native_outdoor",
                "vc_native_flat", "vc_native_indoor", "vc_native_outdoor",
                "precision", "recall", "f1",
                "precision_flat", "recall_flat", "f1_flat",
                "precision_indoor", "recall_indoor", "f1_indoor",
                "precision_outdoor", "recall_outdoor", "f1_outdoor",
                "acc_WA_flat", "acc_WA_indoor", "acc_WA_outdoor",
                "acc_YT_flat", "acc_YT_indoor", "acc_YT_outdoor"]

    def __empty_row(self):
        num_cols = len(self.__get_columns())
        empty_row = []

        for _ in range(num_cols):
            empty_row.append(0)

        return empty_row
