import argparse
from pathlib import Path

import tensorflow as tf

# tf.config.run_functions_eagerly(True)

from dataset import DataFactory
from models import MISLNet, MobileNet, EfficientNet


def none_or_str(value):
    if value == 'None':
        return None
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train the CNNs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--fc_layers', type=int, default=2, help='Number of fully-connected layers [default: 2]')
    parser.add_argument('--fc_size', type=int, default=1024, help='Number of neurons in Fully Connected layers')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model_name', type=str, help='Name for the model')
    parser.add_argument('--use_pretrained', type=bool, default=True, help='Use pretrained weights from ImageNet')
    parser.add_argument('--model_path', type=str, help='Path to model to continue training (*.h5)')
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset to train the constrained_net')
    parser.add_argument('--height', type=int, default=480, help='Height of CNN input dimension [default: 480]')
    parser.add_argument('--width', type=int, default=800, help='Width of CNN input dimension [default: 800]')
    parser.add_argument('--gpu_id', type=int, default=0, help='Choose the available GPU devices')
    parser.add_argument('--category', type=str, help='enter "native", "whatsapp", or "youtube"')
    parser.add_argument('--global_results_dir', type=Path, required=True, help='Path to results dir')
    parser.add_argument('--const_type', type=none_or_str, default=None, help='Constraint type')
    parser.add_argument('--net_type', type=str, default='mobile', choices=['mobile', 'eff', 'misl'])
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')

    args = parser.parse_args()
    fc_size = args.fc_size
    fc_layers = args.fc_layers
    n_epochs = args.epochs
    cnn_height = args.height
    cnn_width = args.width
    batch_size = args.batch_size
    model_path = args.model_path
    model_name = args.model_name
    dataset_path = args.dataset
    gpu_id = args.gpu_id
    category = args.category
    global_results_dir = args.global_results_dir
    const_type = args.const_type
    net_type = args.net_type
    use_pretrained = args.use_pretrained

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[gpu_id], device_type='GPU')

    data_factory = DataFactory(input_dir=dataset_path,
                               batch_size=batch_size,
                               height=cnn_height,
                               width=cnn_width)

    num_classes = len(data_factory.get_class_names())
    train_ds, num_batches = data_factory.get_tf_train_data(category=category)
    filename_ds, val_ds = data_factory.get_tf_val_data(category=category)

    if net_type == 'mobile':
        net = MobileNet(num_batches, global_results_dir, const_type, lr=args.lr)
        if model_path:
            net.set_model(model_path)
        else:
            net.create_model(num_classes, cnn_height, cnn_width, model_name, use_pretrained)

    elif net_type == 'eff':
        net = EfficientNet(num_batches, global_results_dir, const_type)
        net.create_model(num_classes, cnn_height, cnn_width, model_name, use_pretrained)
        if model_path:
            net.set_model(model_path)

    elif net_type == 'misl':
        net = MISLNet(num_batches, global_results_dir, const_type)
        net.create_model(num_classes, fc_layers, fc_size, cnn_height, cnn_width, model_name)
        if model_path:
            net.set_model(model_path)
    else:
        raise ValueError('Invalid net type')

    net.print_model_summary()
    net.train(train_ds=train_ds, val_ds=val_ds, epochs=n_epochs)
