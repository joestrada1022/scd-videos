import argparse
from pathlib import Path

import tensorflow as tf

import dataset
from models import MISLNet, MobileNet, ResNet


def none_or_str(value):
    if value == 'None':
        return None
    return str(value)


def none_or_bool(value):
    if value == 'None':
        return None
    return bool(int(value))


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Train the CNNs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset params
    parser.add_argument('--dataset_name', type=str, choices=['vision', 'qufvd'])
    parser.add_argument('--all_I_frames_dir', type=Path, help='Input directory of extracted I frames')
    parser.add_argument('--all_frames_dir', type=Path, help='Input directory of extracted frames')
    parser.add_argument('--frame_selection', type=str, default='equally_spaced', choices=['equally_spaced', 'first_N'])
    parser.add_argument('--frame_type', type=str, default='I', choices=['I', 'all'])
    parser.add_argument('--fpv', type=int, default=50, help='max number of frames per video (set -1 for all frames)')
    parser.add_argument('--category', type=str, choices=["native", "whatsapp", "youtube", "None"])

    # ConvNet params
    parser.add_argument('--const_type', type=none_or_str, default=None, help='Constraint type',
                        choices=['derrick', 'None', None])
    parser.add_argument('--model_name', type=str, help='Name for the model')
    parser.add_argument('--use_pretrained', type=none_or_bool, default=True, help='Use pretrained net from ImageNet')
    parser.add_argument('--model_path', type=str, help='Path to model to continue training (*.h5)')
    parser.add_argument('--height', type=int, default=480, help='Height of CNN input dimension [default: 480]')
    parser.add_argument('--width', type=int, default=800, help='Width of CNN input dimension [default: 800]')
    parser.add_argument('--net_type', type=str, default='mobile', choices=['mobile', 'misl', 'res'])

    # Optimization params
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')

    # Session params
    parser.add_argument('--gpu_id', type=int, default=None, help='Choose the available GPU devices')
    parser.add_argument('--global_results_dir', type=Path, required=True, help='Path to results dir')

    if args:
        p = parser.parse_args(args)  # read from custom input
    else:
        p = parser.parse_args()  # read from command line

    print(f'Using pre-trained model - {p.use_pretrained}')

    if p.gpu_id is not None:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(physical_devices[p.gpu_id], device_type='GPU')

    return p


def run_train_flow(args=None):
    p = parse_args(args)

    if p.dataset_name == 'vision':
        data_factory = dataset.vision.DataFactory(p)
    elif p.dataset_name == 'qufvd':
        data_factory = dataset.qufvd.DataFactory(p)
    else:
        raise ValueError(f'Invalid option {p.dataset_name}')

    num_classes = len(data_factory.class_names)
    train_ds, num_batches = data_factory.get_tf_train_data(category=p.category)  # fixme: remove num_batches
    val_ds = None  # validation is being performed in a separate script (run_evaluate.py)

    if p.net_type == 'mobile':
        net = MobileNet(p.global_results_dir, p.model_name, p.const_type, p.lr)
        net.create_model(num_classes, p.height, p.width, p.model_name, p.use_pretrained)
    elif p.net_type == 'misl':
        net = MISLNet(p.global_results_dir, p.model_name, p.const_type, p.lr)
        net.create_model(num_classes, p.height, p.width)
    elif p.net_type == 'res':
        net = ResNet(p.global_results_dir, p.model_name, p.const_type, p.lr)
        net.create_model(num_classes, p.height, p.width, p.use_pretrained)
    else:
        raise ValueError('Invalid net type')

    if p.model_path:  # to continue the training
        net.set_model(p.model_path)
        net.compile()

    net.print_model_summary()
    net.train(train_ds=train_ds, val_ds=val_ds, epochs=p.epochs)


if __name__ == "__main__":
    run_train_flow()
