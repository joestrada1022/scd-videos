from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D

from . import BaseNet, Constrained3DKernelMinimal, SupervisedContrastiveLoss


# tf.data.experimental.enable_debug_mode()


# Credits - https://keras.io/examples/vision/supervised-contrastive-learning/
# Essentially, training an image classification model with Supervised Contrastive Learning is performed in two phases:
# 1. Training an encoder to learn to produce vector representations of input images such that representations of images
#       in the same class will be more similar compared to representations of images in different classes.
# 2. Training a classifier on top of the frozen encoder.


class ResNetContrastive(BaseNet):
    def __init__(self, num_batches, global_results_dir, const_type, model_path=None, lr=0.1):
        super().__init__(num_batches, global_results_dir, const_type, model_path, lr)

    @staticmethod
    def __create_encoder(input_shape, pretrained=False):
        """
        :param input_shape: tuple of ints - (height, width, num_channels)
        :param weights: 'imagenet' or None
        :return: tensorflow model
        """
        assert type(pretrained) is bool, f'Invalid value for pretrained - {pretrained}'
        weights = 'imagenet' if pretrained else None
        encoder = tf.keras.applications.resnet_v2.ResNet50V2(input_shape=input_shape, include_top=False,
                                                             weights=weights, pooling='avg')
        return encoder

    @staticmethod
    def __add_projection_head(encoder, input_shape, projection_units=128):
        inputs = tf.keras.Input(shape=input_shape)
        features = encoder(inputs)
        outputs = tf.keras.layers.Dense(projection_units, activation="relu")(features)
        outputs = tf.keras.layers.Flatten()(outputs)
        model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="encoder_with_projection-head"
        )
        return model

    def get_encoder(self):
        if hasattr(self, 'encoder'):
            return self.encoder
        else:
            raise AttributeError('Must call create_model with `model_type == encoder` before accessing the encoder')

    @staticmethod
    def create_classifier(encoder, input_shape, num_classes, encoder_trainable=True, dropout_rate=0.5,
                          hidden_units=512):
        for layer in encoder.layers:
            layer.trainable = encoder_trainable

        inputs = tf.keras.Input(shape=input_shape)
        features = encoder(inputs)
        features = tf.keras.layers.Flatten()(features)
        features = tf.keras.layers.Dropout(dropout_rate)(features)
        features = tf.keras.layers.Dense(hidden_units, activation="relu")(features)
        features = tf.keras.layers.Dropout(dropout_rate)(features)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(features)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="classifier")
        return model

    def create_model(self, num_outputs, height, width, model_type, model_name=None, use_pretrained=True):
        self.model_name = model_name
        input_shape = (height, width, 3)

        if model_type == 'encoder':
            self.encoder = self.__create_encoder(input_shape, use_pretrained)
            if self.const_type:
                self.encoder = Sequential([
                    tf.keras.layers.InputLayer(input_shape=input_shape),
                    Conv2D(filters=3, kernel_size=5, strides=(1, 1), padding="same",
                           kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.0001, maxval=1, seed=108),
                           kernel_constraint=Constrained3DKernelMinimal(self.const_type),
                           name="constrained_layer"),
                    self.__create_encoder(input_shape, use_pretrained)
                ])

            encoder_with_projection_head = self.__add_projection_head(self.encoder, input_shape)
            self.model = encoder_with_projection_head
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(self.lr),
                loss=SupervisedContrastiveLoss(temperature=0.05),
            )

        elif model_type == 'classifier':
            encoder = self.get_encoder()
            self.model = self.create_classifier(encoder, input_shape, num_outputs, encoder_trainable=False)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(self.lr),
                loss=tf.keras.losses.categorical_crossentropy,
            )

        else:
            raise ValueError(f'Invalid model_type {model_type}')

        return self.model

    def compile(self):
        raise RuntimeWarning('The model is already being compiled at the time of creation - `create_model`')


if __name__ == '__main__':
    net = ResNetContrastive(num_batches=10, global_results_dir=Path('.'), const_type=None, model_path=None)
    net.create_model(28, 480, 800, model_type='encoder', model_name=None, use_pretrained=True)
    net.create_model(28, 480, 800, model_type='classifier', model_name=None, use_pretrained=True)
    print(' ')
