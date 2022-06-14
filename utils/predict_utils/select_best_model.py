import argparse
import shutil
from pathlib import Path


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_summary', type=Path, required=True, help='Path to validation summary')

    if args:
        p = parser.parse_args(args)  # read from custom input
    else:
        p = parser.parse_args()  # read from command line

    assert p.val_summary.exists(), f'The video statistics file is missing : {p.val_summary}'
    return p


def select_best_model(args=None):
    """
    Select the best epoch with maximum validation accuracy.
    Ties are resolved by choosing the model with the least validation loss.
    Further, ties are resolved by choosing the earliest such epoch.

    :return:
    """
    args = parse_args(args)

    # Fetch the epoch-wise accuracy and loss from the .csv file
    with open(args.val_summary, 'r') as f:
        lines = sorted(f.readlines()[2:])  # skipping first 2 header rows
    eval_loss = [float(x.split(',')[3]) for x in lines]
    eval_acc = [float(x.split(',')[1]) for x in lines]
    eval_models = [x.split(',')[0] for x in lines]

    # Select epoch with maximum val accuracy. If there is a tie, pick the one with least val loss
    max_elem = max(zip(eval_acc, [-x for x in eval_loss]))
    max_acc, min_loss = (max_elem[0], -max_elem[1])

    # Determine all epochs with max_acc and min_loss
    max_acc_indices = set([i for i, x in enumerate(eval_acc) if x == max_acc])
    min_loss_indices = set([i for i, x in enumerate(eval_loss) if x == min_loss])
    best_epoch_indices = max_acc_indices.intersection(min_loss_indices)

    # Choose the earliest seen epoch.
    # Ignoring other epochs as they would have recovered from under-fitting or over-fitting
    index = min(best_epoch_indices)

    # Copy the model corresponding to the best epoch for predictions on the test set
    model_path = args.val_summary.parent.parent.parent.joinpath(f'{eval_models[index]}.h5')
    tmp = list(model_path.parts)[:-1]
    tmp[-4] += '_pred'
    dest_path = Path('/'.join(tmp))
    dest_path.mkdir(exist_ok=True, parents=True)
    shutil.copy(model_path, dest_path)


if __name__ == '__main__':
    select_best_model()
