from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def load_model(model_path):
    from models import Constrained3DKernelMinimal, CombineInputsWithConstraints, PPCCELoss
    custom_objects = {
        'Constrained3DKernelMinimal': Constrained3DKernelMinimal,
        'CombineInputsWithConstraints': CombineInputsWithConstraints,
        'PPCCELoss': PPCCELoss
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


if __name__ == '__main__':
    # 2a. Load the model
    model = load_model(model_path=Path(r'/scratch/p288722/runtime_data/scd-videos/dev_combine_layer/'
                                       r'50_frames_28d_64_pred/mobile_net/models/ConstNet_guru/fm-e00020.h5'))

    weights = []
    biases = []
    for i, layer in enumerate(model.layers):
        # Weights of the first constrained layer filter
        if 'batch_normalization' not in layer.name and 'reshape' not in layer.name:
            if layer.get_weights():
                w, b = layer.get_weights()
                if w.all():
                    weights.append(np.ravel(w))
                if b.all():
                    biases.append(np.ravel(b))

    weights = np.concatenate(weights)
    biases = np.concatenate(biases)

    plt.figure()
    plt.hist(np.ravel(weights), bins=500, log=True)
    plt.title(f'All weights')
    plt.ylabel('count')
    plt.xlabel('weights')
    plt.show()
    plt.close()

    plt.figure()
    plt.hist(np.ravel(biases), bins=500, log=True)
    plt.title(f'All biases')
    plt.ylabel('count')
    plt.xlabel('biases')
    plt.show()
    plt.close()
