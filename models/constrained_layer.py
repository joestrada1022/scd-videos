import numpy as np
import tensorflow as tf
from tensorflow.keras.constraints import Constraint


class Constrained3DKernelMinimal(Constraint):
    def __init__(self, const_type='derrick'):
        super(Constrained3DKernelMinimal, self).__init__()
        self.const_type = const_type

    def __call__(self, w):
        """
        =========================================================================================
        === Same functionality as Constrained3DKernel but without extensive in-line comments. ===
        =========================================================================================

        This custom weight constraint implements the constrained convolutional layer for RGB-images.
        However, this is not as straightforward as it should be. Since TensorFlow prohibits us from
        assigning new values directly to the weight matrix, we need a trick to update its values.
        This trick consists of multiplying the weight matrix by so-called 'mask matrices'
        to get the desired results.

        For example, if we want to set the center values of the weight matrix to zero, we first create a
        mask matrix of the same size consisting of only ones, except at the center cells. The center cells
        will have as value one. After multiplying the weight matrix with this mask matrix,
        we obtain a 'new' weight matrix, where the center values are set to zero but with the remaining values
        untouched.

        More information about this problem:
        #https://github.com/tensorflow/tensorflow/issues/14132

        The incoming weight matrix 'w' is a 4D array of shape (x, y, z, n_filters) where (normally):
        x = 5
        y = 5
        z = 3, since we're using RGB images
        n_filters = 3

        This means there are 3 filters in total with each filter being 3-dimensional.
       """
        if self.const_type == 'derrick':
            return self._constraint_derrick_et_al(w)
        else:
            raise ValueError('Invalid constraint type')

    def get_config(self):
        return {}

    @staticmethod
    def _constraint_derrick_et_al(w):
        w_original_shape = w.shape

        # 1. Reshaping of 'w'
        x, y, z, n_kernels = w_original_shape[0], w_original_shape[1], w_original_shape[2], w_original_shape[3]
        center = x // 2  # Determine the center cell on the xy-plane.
        new_shape = [n_kernels, z, x, y]
        w = tf.transpose(w, [3, 2, 0, 1])

        # 2. Set center values of 'w' to zero by multiplying 'w' with mask-matrix
        center_zero_mask = np.ones(new_shape)
        center_zero_mask[:, :, center, center] = 0
        w *= center_zero_mask

        # 3. Normalize values w.r.t xy-planes
        xy_plane_sum = tf.reduce_sum(w, [2, 3], keepdims=True)  # Recall new shape of w: (n_kernels, z, y, x).
        w = tf.math.divide(w, xy_plane_sum) * 10000  # Divide each element by its corresponding xy-plane sum-value

        # 4. Set center values of 'w' to negative one by subtracting mask-matrix from 'w'
        center_one_mask = np.zeros(new_shape)
        center_one_mask[:, :, center, center] = 10000
        w = tf.math.subtract(w, center_one_mask)

        # Reshape 'w' to original shape and return
        w = tf.transpose(w, [2, 3, 1, 0])
        return w
