from pathlib import Path

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

from . import BaseNet, Constrained3DKernelMinimal


class ResNet(BaseNet):
    def __init__(self, global_results_dir, model_name, const_type, lr):
        super().__init__(global_results_dir, model_name, const_type, lr)

    def create_model(self, num_classes, height, width, use_pretrained=True):
        input_shape = (height, width, 3)
        self.model = ResNet50V2(include_top=True, weights=None, input_shape=input_shape, classes=num_classes)
        if use_pretrained:
            pretrained = ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
            for idx in range(len(pretrained.layers)):
                self.model.layers[idx].set_weights(pretrained.layers[idx].get_weights())

        if self.const_type:
            self.model = Sequential([
                tf.keras.layers.InputLayer(input_shape=input_shape),
                Conv2D(filters=3, kernel_size=5, strides=(1, 1), padding="same",
                       kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.0001, maxval=1, seed=108),
                       kernel_constraint=Constrained3DKernelMinimal(self.const_type),
                       name="constrained_layer"),
                self.model
            ])

        self.compile()
        return self.model


if __name__ == '__main__':
    net = ResNet(global_results_dir=Path('.'), model_name='ResNet', const_type=None, lr=0.1)
    net.create_model(num_classes=28, height=480, width=800, use_pretrained=True)
