import sys
from pathlib import Path

import matplotlib
import numpy as np
import tensorflow as tf
from keract import keract
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
from tqdm import tqdm

sys.path.insert(1, '/home/p288722/git_code/scd-videos')

from dataset.data_factory import DataFactory
from models.constrained_layer import Constrained3DKernelMinimal
from models.mobile_net import MobileNetBase


def load_model(model_path):
    # noinspection PyProtectedMember
    custom_objects = {
        'Constrained3DKernelMinimal': Constrained3DKernelMinimal,
        '_hard_swish': MobileNetBase._hard_swish,
        '_relu6': MobileNetBase._relu6
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)


def get_constrained_activations():
    maps = []
    for element in eval_ds:
        maps.append(keract.get_activations(model, element[0].numpy(), layer_names=['constrained_layer'],
                                           nodes_to_evaluate=None, output_format='simple', nested=False,
                                           auto_compile=True)['constrained_layer'][0]
                    )
    return maps


def compute_correlation_map():
    num_filters = 3
    pcc_map = np.zeros(shape=(len(sorted_data), len(sorted_data), num_filters))

    # Memoization for PCC Map
    data = np.array([x[1] for x in sorted_data])
    data_std = data.std(axis=(1, 2), keepdims=True)
    data_mean_sub = data - data.mean(axis=(1, 2), keepdims=True)

    for channel_index in range(num_filters):
        for row_index, (r_class_id, r_feature_map) in tqdm(enumerate(sorted_data)):
            for col_index, (c_class_id, c_feature_map) in enumerate(sorted_data):
                pcc_map[row_index][col_index][channel_index] = stats.pearsonr(
                    np.ravel(r_feature_map[:, :, channel_index]), np.ravel(c_feature_map[:, :, channel_index]))[0]

    return pcc_map


def plot_correlations(plot_name):
    matplotlib.rcParams.update({'font.size': 15})
    fig, axs = plt.subplots(1, 3, figsize=(5, 15), dpi=300, sharey=True, sharex=True)
    # Set the range for the plots
    # v_min = np.min(filters)
    # v_max = np.max(filters)
    v_min = v_max = None
    for i in range(3):
        # Plot the filter weights as an image
        im = axs[i].imshow(pcc[:, :, i], vmin=v_min, vmax=v_max, cmap='viridis')
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    # axs[0].set_ylabel('Constrained Filters')
    axs[0].title.set_text('Channel 1')
    axs[1].title.set_text('Channel 2')
    axs[2].title.set_text('Channel 3')
    plt.tight_layout()
    plt.savefig(str(Path(r'/scratch/p288722/runtime_data/scd-videos/dev_const_layer/plots').joinpath(plot_name)))
    plt.show()
    plt.cla()
    plt.clf()
    plt.close()


if __name__ == '__main__':
    # 1. Load the dataset
    dataset = DataFactory(input_dir=Path(r'/scratch/p288722/datasets/vision/2_devices/bal_50_frames'),
                          batch_size=1, height=480, width=800)
    filename_ds, eval_ds = dataset.get_tf_evaluation_data(category=None, mode='train')

    # 2a. Load the model
    model = load_model(model_path=Path(r'/scratch/p288722/runtime_data/scd-videos/dev_const_layer_/'
                                       r'50_frames_2d/mobile_net/models/ConstNet_bayar/fm-e00004.h5'))

    # Weights of the first constrained layer filter
    filters, biases = model.layers[1].get_weights()

    # 3. Run the predictions and save the results (along with class labels)
    activations = get_constrained_activations()

    # 4. Compute pair-wise pearson's co-relation
    labels = [int(str(x.numpy())[3:5]) for x in filename_ds]
    sorted_data = sorted(zip(labels, activations), key=lambda x: x[0])
    pcc = compute_correlation_map()

    # 5. Plot the results
    plot_correlations(plot_name='cor_map_mobile_net_const_bayar_train.png')
