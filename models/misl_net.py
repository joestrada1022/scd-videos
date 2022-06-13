import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Softmax
from tensorflow.keras.models import Sequential

from . import BaseNet, Constrained3DKernelMinimal


class MISLNet(BaseNet):
    def __init__(self, global_results_dir, model_name, const_type, lr):
        super().__init__(global_results_dir, model_name, const_type, lr)

    def create_model(self, num_classes, height, width):

        input_shape = (height, width, 3)
        model = Sequential()

        if self.const_type:
            cons_layer = Conv2D(
                filters=self.constrained_n_filters,
                kernel_size=self.constrained_kernel_size,
                strides=(1, 1),
                input_shape=input_shape,
                padding="same",  # Intentionally
                kernel_constraint=Constrained3DKernelMinimal(self.const_type),
                kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.0001, maxval=1, seed=108),
                name="constrained_layer")
            model.add(cons_layer)
            model.add(Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2), padding="same"))
        else:
            model.add(Conv2D(filters=96, kernel_size=(7, 7), strides=(2, 2), padding="same", input_shape=input_shape))

        model.add(BatchNormalization())
        model.add(Activation(tf.keras.activations.tanh))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation(tf.keras.activations.tanh))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation(tf.keras.activations.tanh))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation(tf.keras.activations.tanh))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Flatten())
        fc_layers = 2
        fc_size = 1024
        for _ in range(fc_layers):
            model.add(Dense(fc_size, activation=tf.keras.activations.tanh))

        model.add(Dense(num_classes, activation=None))
        model.add(Softmax())

        self.model = model
        self.compile()
        return model
