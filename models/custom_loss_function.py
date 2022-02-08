import tensorflow as tf
import tensorflow_addons as tfa


class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=1.0, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        labels = tf.argmax(labels, axis=1)  # converting one-hot labels to integer indices
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


class PPCCELoss(tf.keras.losses.Loss):
    def __init__(self, distance_matrix, name=None):
        super(PPCCELoss, self).__init__(name=name)
        self.D = distance_matrix
        self.eps = 1e-14

    def __call__(self, y_true, y_pred, sample_weight=None):
        """
        :param y_true: One hot encoded ground truths
        :param y_pred: The output of the model (softmax scores)
        :param sample_weight:
        :return: scalar loss
        """

        loss = []
        true_idx = tf.math.argmax(y_true, axis=1)
        pred_idx = tf.math.argmax(y_pred, axis=1)
        # distance_pred = []

        for idx, (p, k) in enumerate(zip(y_pred, true_idx)):
            if pred_idx[idx] == k:  # correct prediction
                loss.append(-tf.math.log(p[k] + self.eps))
            else:  # incorrect prediction
                # loss.append(-tf.math.log(p[k] * tf.linalg.tensordot(self.D[k], p, axes=1) + self.eps))
                loss.append(-tf.math.log(p[k] * self.D[k][pred_idx[idx]] + self.eps))

            # distance_pred.append(self.D[k][pred_idx[idx]])
            # print(','.join([str(float(x)) for x in p]))
            # print(','.join([str(float(x)) for x in self.D[k]]))

        # print(','.join([str(float(x)) for x in loss]))
        # print(','.join([str(float(x)) for x in distance_pred]))
        # print(','.join([str(int(x)) for x in true_idx]))
        # print(','.join([str(int(x)) for x in pred_idx]))

        return tf.math.reduce_mean(loss)
