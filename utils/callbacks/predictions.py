import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import keras


class PredictionsCallback(keras.callbacks.Callback):

    def __init__(self, train_ds=None, val_ds=None, test_ds=None):
        super(PredictionsCallback, self).__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

    def run_predictions(self, ds):
        softmax_scores = self.model.predict(ds)
        labels_ds = ds.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices(y))
        one_hot_encoded_ground_truths = np.array(list(labels_ds.as_numpy_iterator())).astype(np.int32)

        # Use reduction type 'None' to create array of losses for each prediction
        cce = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        cce_losses = cce(one_hot_encoded_ground_truths, softmax_scores).numpy()

        # Both actual and predicted labels are in one-hot vector form
        ground_truths = [np.argmax(x) for x in one_hot_encoded_ground_truths]
        predictions = [np.argmax(x) for x in softmax_scores]
        prediction_losses = [x for x in cce_losses]

        avg_acc = accuracy_score(ground_truths, predictions)
        avg_loss = np.average(prediction_losses)

        return avg_acc, avg_loss

    def on_epoch_end(self, epoch, logs=None):

        if self.train_ds:
            acc, loss = self.run_predictions(self.train_ds)
            print(f'Epoch %d: train loss - %.4f, acc - %.4f' % (epoch + 1, loss, acc), flush=True)
        if self.val_ds:
            acc, loss = self.run_predictions(self.val_ds)
            print(f'Epoch %d:   val loss - %.4f, acc - %.4f' % (epoch + 1, loss, acc), flush=True)
        if self.test_ds:
            acc, loss = self.run_predictions(self.test_ds)
            print(f'Epoch %d:  test loss - %.4f, acc - %.4f' % (epoch + 1, loss, acc), flush=True)

        print('Summary of metrics, computed by tensorflow:')
