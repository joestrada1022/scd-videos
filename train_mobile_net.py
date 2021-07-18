import argparse

from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train the constrained_net',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--fc_layers', type=int, default=2, required=False,
                        help='Number of fully-connected layers [default: 2].')
    parser.add_argument('--fc_size', type=int, default=1024, required=False,
                        help='Number of neurons in Fully Connected layers')
    parser.add_argument('--epochs', type=int, required=False, default=1, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, required=False, default=32, help='Batch size')
    parser.add_argument('--model_name', type=str, required=False, help='Name for the model')
    parser.add_argument('--model_path', type=str, required=False, help='Path to model to continue training (*.h5)')
    parser.add_argument('--dataset', type=str, required=False, help='Path to dataset to train the constrained_net')
    parser.add_argument('--height', type=int, required=False, default=480, help='Input Height [default: 480]')
    parser.add_argument('--width', type=int, required=False, default=800,
                        help='Width of CNN input dimension [default: 800]')
    parser.add_argument('--constrained', type=int, required=False, default=1, help='Include constrained layer')
    parser.add_argument('--gpu_id', type=int, required=False, default=0,
                        help='Choose the available GPU devices')
    parser.add_argument('--category', type=str, required=False, help='enter "native", "whatsapp", or "youtube"')
    parser.add_argument('--global_results_dir', type=str, required=True, help='Path to results dir')
    parser.add_argument('--const_type', type=str, required=False, default='guru', help='Constraint type')

    args = parser.parse_args()
    fc_size = args.fc_size
    fc_layers = args.fc_layers
    n_epochs = args.epochs
    cnn_height = args.height
    cnn_width = args.width
    batch_size = args.batch_size
    use_constrained_layer = args.constrained == 1
    model_path = args.model_path
    model_name = args.model_name
    dataset_path = args.dataset
    gpu_id = args.gpu_id
    category = args.category
    global_results_dir = Path(args.global_results_dir)
    const_type = args.const_type

    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[gpu_id], device_type='GPU')

    # from constrained_net.constrained_net import ConstrainedNet
    from models.mobile_net import MobileNet
    from dataset.data_factory import DataFactory

    # Network = ConstrainedNet
    Network = MobileNet

    data_factory = DataFactory(input_dir=dataset_path,
                               batch_size=batch_size,
                               height=cnn_height,
                               width=cnn_width)

    num_classes = len(data_factory.get_class_names())
    train_ds, num_batches = data_factory.get_tf_train_data(category=category)
    filename_ds, val_ds = data_factory.get_tf_val_data(category=category)

    net = Network(constrained_net=use_constrained_layer,
                  num_batches=num_batches,
                  global_results_dir=global_results_dir,
                  const_type=const_type,
                  )
    if model_path:
        net.set_model(model_path)
    else:
        # Create new model
        net.create_model(num_output=num_classes, height=cnn_height, width=cnn_width, model_name=model_name)

    net.print_model_summary()
    net.train(train_ds=train_ds, val_ds=val_ds, epochs=n_epochs)
