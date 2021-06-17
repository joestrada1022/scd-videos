"""MobileNet v3 Large models for Keras.
# Reference
    [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)
"""

import os

import tensorflow as tf
from keras import backend as K
from keras.layers import Activation, BatchNormalization, Add, Multiply, Reshape
from keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D
from keras.layers import Input
from keras.models import Model
from tensorflow.keras.layers import Conv2D

from models.base_net import BaseNet
from models.constrained_layer import Constrained3DKernelMinimal

"""MobileNet v3 models for Keras.
# Reference
    [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244?context=cs)
"""


class MobileNetBase:
    def __init__(self, shape, n_class, alpha=1.0):
        """Init

        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
        """
        self.shape = shape
        self.n_class = n_class
        self.alpha = alpha

    @staticmethod
    def _relu6(x):
        """Relu 6
        """
        return K.relu(x, max_value=6.0)

    @staticmethod
    def _hard_swish(x):
        """Hard swish
        """
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0

    def _return_activation(self, x, nl):
        """Convolution Block
        This function defines a activation choice.

        # Arguments
            x: Tensor, input tensor of conv layer.
            nl: String, non linearity activation type.

        # Returns
            Output tensor.
        """
        if nl == 'HS':
            x = Activation(self._hard_swish)(x)
        if nl == 'RE':
            x = Activation(self._relu6)(x)

        return x

    def _conv_block(self, inputs, filters, kernel, strides, nl):
        """Convolution Block
        This function defines a 2D convolution operation with BN and activation.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
            nl: String, non linearity activation type.

        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
        x = BatchNormalization(axis=channel_axis)(x)

        return self._return_activation(x, nl)

    @staticmethod
    def _squeeze(inputs):
        """Squeeze and Excitation.
        This function defines a squeeze structure.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
        """
        input_channels = int(inputs.shape[-1])

        x = GlobalAveragePooling2D()(inputs)
        x = Dense(input_channels, activation='relu')(x)
        x = Dense(input_channels, activation='hard_sigmoid')(x)
        x = Reshape((1, 1, input_channels))(x)
        x = Multiply()([inputs, x])

        return x

    def _bottleneck(self, inputs, filters, kernel, e, s, squeeze, nl):
        """Bottleneck
        This function defines a basic bottleneck structure.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            e: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            squeeze: Boolean, Whether to use the squeeze.
            nl: String, non linearity activation type.

        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        input_shape = K.int_shape(inputs)

        t_channel = int(e)
        c_channel = int(self.alpha * filters)

        r = s == 1 and input_shape[3] == filters

        x = self._conv_block(inputs, t_channel, (1, 1), (1, 1), nl)

        x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = self._return_activation(x, nl)

        if squeeze:
            x = self._squeeze(x)

        x = Conv2D(c_channel, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)

        if r:
            x = Add()([x, inputs])

        return x

    def build(self):
        pass


# class MobileNetV3LargeWrapper(MobileNetBase):
#     def __init__(self, shape, n_class, alpha=1.0, include_top=True):
#         """Init.
#
#         # Arguments
#             input_shape: An integer or tuple/list of 3 integers, shape
#                 of input tensor.
#             n_class: Integer, number of classes.
#             alpha: Integer, width multiplier.
#             include_top: if include classification layer.
#
#         # Returns
#             MobileNet v3 model.
#         """
#         super(MobileNetV3LargeWrapper, self).__init__(shape, n_class, alpha)
#         self.include_top = include_top
#
#     def build(self, plot=False):
#         """build MobileNetV3 Large.
#
#         # Arguments
#             plot: Boolean, weather to plot model.
#
#         # Returns
#             model: Model, model.
#         """
#         inputs = Input(shape=self.shape)
#
#         x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')
#
#         x = self._bottleneck(x, 16, (3, 3), e=16, s=1, squeeze=False, nl='RE')
#         x = self._bottleneck(x, 24, (3, 3), e=64, s=2, squeeze=False, nl='RE')
#         x = self._bottleneck(x, 24, (3, 3), e=72, s=1, squeeze=False, nl='RE')
#         x = self._bottleneck(x, 40, (5, 5), e=72, s=2, squeeze=True, nl='RE')
#         x = self._bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
#         x = self._bottleneck(x, 40, (5, 5), e=120, s=1, squeeze=True, nl='RE')
#         x = self._bottleneck(x, 80, (3, 3), e=240, s=2, squeeze=False, nl='HS')
#         x = self._bottleneck(x, 80, (3, 3), e=200, s=1, squeeze=False, nl='HS')
#         x = self._bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
#         x = self._bottleneck(x, 80, (3, 3), e=184, s=1, squeeze=False, nl='HS')
#         x = self._bottleneck(x, 112, (3, 3), e=480, s=1, squeeze=True, nl='HS')
#         x = self._bottleneck(x, 112, (3, 3), e=672, s=1, squeeze=True, nl='HS')
#         x = self._bottleneck(x, 160, (5, 5), e=672, s=2, squeeze=True, nl='HS')
#         x = self._bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')
#         x = self._bottleneck(x, 160, (5, 5), e=960, s=1, squeeze=True, nl='HS')
#
#         x = self._conv_block(x, 960, (1, 1), strides=(1, 1), nl='HS')
#         x = GlobalAveragePooling2D()(x)
#         x = Reshape((1, 1, 960))(x)
#
#         x = Conv2D(1280, (1, 1), padding='same')(x)
#         x = self._return_activation(x, 'HS')
#
#         if self.include_top:
#             x = Conv2D(self.n_class, (1, 1), padding='same', activation='softmax')(x)
#             x = Reshape((self.n_class,))(x)
#
#         model = Model(inputs, x)
#
#         return model


class MobileNetV3SmallWrapper(MobileNetBase):
    def __init__(self, shape, n_class, alpha=1.0, include_top=True):
        """Init.

        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
            include_top: if include classification layer.

        # Returns
            MobileNet v3 model.
        """
        super(MobileNetV3SmallWrapper, self).__init__(shape, n_class, alpha)
        self.include_top = include_top

    def build(self, is_constrained=False):
        """build MobileNetV3 Small.

        # Arguments
            plot: Boolean, weather to plot model.

        # Returns
            model: Model, model.
        """
        inputs = Input(shape=self.shape)
        if is_constrained:
            x = Conv2D(filters=3, kernel_size=5, strides=(1, 1), padding="valid",
                       kernel_initializer=tf.keras.initializers.RandomUniform(minval=0.0001, maxval=1, seed=108),
                       kernel_constraint=Constrained3DKernelMinimal(),
                       name="constrained_layer")(inputs)
            x = self._conv_block(x, 16, (3, 3), strides=(2, 2), nl='HS')
        else:
            x = self._conv_block(inputs, 16, (3, 3), strides=(2, 2), nl='HS')

        x = self._bottleneck(x, 16, (3, 3), e=16, s=2, squeeze=True, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=72, s=2, squeeze=False, nl='RE')
        x = self._bottleneck(x, 24, (3, 3), e=88, s=1, squeeze=False, nl='RE')
        x = self._bottleneck(x, 40, (5, 5), e=96, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 40, (5, 5), e=240, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 48, (5, 5), e=120, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 48, (5, 5), e=144, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=288, s=2, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')
        x = self._bottleneck(x, 96, (5, 5), e=576, s=1, squeeze=True, nl='HS')

        x = self._conv_block(x, 576, (1, 1), strides=(1, 1), nl='HS')
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 576))(x)

        x = Conv2D(1280, (1, 1), padding='same')(x)
        x = self._return_activation(x, 'HS')

        if self.include_top:
            x = Conv2D(self.n_class, (1, 1), padding='same', activation='softmax')(x)
            x = Reshape((self.n_class,))(x)

        model = Model(inputs, x)

        return model


class MobileNet(BaseNet):
    def __init__(self, constrained_net, num_batches, global_results_dir, model_path=None):
        super().__init__(constrained_net, num_batches, global_results_dir, model_path)

    def set_model(self, model_path):
        # Path is e.g. ~/constrained_net/fm-e00001.h5
        path_splits = model_path.split(os.sep)
        model_name = path_splits[-2]

        self.model_path = model_path
        self.model_name = model_name

        # noinspection PyProtectedMember
        self.model = tf.keras.models.load_model(model_path, custom_objects={
            'Constrained3DKernelMinimal': Constrained3DKernelMinimal,
            '_hard_swish': MobileNetBase._hard_swish,
            '_relu6': MobileNetBase._relu6})

        if self.model is None:
            raise ValueError(f"Model could not be loaded from location {model_path}")

    def create_model(self, num_output, height=480, width=800, model_name=None):
        input_shape = (height, width, 3)
        net = MobileNetV3SmallWrapper(shape=input_shape, n_class=num_output)
        model = net.build(is_constrained=self.constrained_net)
        self.model_name = model_name
        self.model = model
        self.compile()
        return model
