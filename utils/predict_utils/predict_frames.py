import os

import numpy as np
import pandas as pd
import tensorflow as tf


class FramePredictor:

    def __init__(self, model_dir, model_file_name, result_dir):
        self.model_dir = model_dir
        self.model_file_name = model_file_name
        self.result_dir = result_dir

        # Load model
        from models import Constrained3DKernelMinimal
        model_path = os.path.join(model_dir, model_file_name)
        custom_objects = {
            'Constrained3DKernelMinimal': Constrained3DKernelMinimal,
        }
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    def start(self, test_ds, filenames, onehot_ground_truths):
        output_file = self.get_output_file()
        return self._predict_and_save(test_ds, filenames, onehot_ground_truths, output_file)

    def get_output_file(self):
        output_file = f"{self.model_file_name.split('.')[0]}_F_predictions.csv"
        return os.path.join(self.result_dir, output_file)

    def __predict_frames(self, test_ds, onehot_ground_truths):
        softmax_scores = self.model.predict(test_ds, verbose=1)

        # Use reduction type 'None' to create array of losses for each prediction
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        cce_losses = cce(onehot_ground_truths, softmax_scores).numpy()

        # Both actual and predicted labels are in one-hot vector form
        onehot_ground_truths = [np.argmax(x) for x in onehot_ground_truths]
        predictions = [np.argmax(x) for x in softmax_scores]
        prediction_losses = [x for x in cce_losses]

        return onehot_ground_truths, predictions, prediction_losses, softmax_scores

    def _predict_and_save(self, test_ds, filenames, onehot_ground_truths, output_file):
        true_labels, predicted_labels, losses, softmax_scores = self.__predict_frames(test_ds, onehot_ground_truths)
        df = pd.DataFrame(list(zip(filenames, true_labels, predicted_labels, losses, softmax_scores)),
                          columns=["File", "True Label", "Predicted Label", "Loss", "Softmax Scores"])
        df.to_csv(output_file, index=False)
        return output_file
