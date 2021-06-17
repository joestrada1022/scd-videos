from pathlib import Path


def verify_videos(dir_1=None, dir_2=None, frame_predictions=None):
    dir_1_videos = set([x.name.split('-')[0] for x in dir_1.glob('*/*.jpg')])
    dir_2_videos = set([x.name.split('-')[0] for x in dir_2.glob('*/*.jpg')])
    dir_1_images = set([x.name for x in dir_1.glob('*/*.jpg')])
    dir_2_images = set([x.name for x in dir_2.glob('*/*.jpg')])

    with open(frame_predictions, 'r') as f:
        f.__next__()
        vides_from_predictions = set([x.split("'")[1].split('-')[0] for x in f.readlines()])

    print('')


if __name__ == '__main__':
    verify_videos(
        # dir_1 = Path(r'/data/p288722/from_f118170/vbdi/datasets/Exp.III/balanced_ds_28D/test'), # icpram
        dir_1 = Path(r'/scratch/p288722/datasets/VISION/bal_28_devices_derrick/test'),
        dir_2 = Path(r'/scratch/p288722/datasets/VISION/bal_28_devices_all_frames/test'),
        frame_predictions = Path(r'/data/p288722/from_f118170/vbdi/models/Exp.III/ExpIII-ConstrCNN-FC2x1024_balanced_28D/predictions/frames/fm-e00001_F_predictions.csv')
    )

    # dir_1 = Path(r'/scratch/p288722/datasets/VISION/bal_28_devices_all_frames/test')
    # dir_1_videos = set([x.name.split('-')[0] for x in dir_1.glob('D08*/*.jpg')])

