"""
This script is adapted from: https://github.com/polimi-ispl/prnu-python/blob/master/example.py
"""
import json
import os
from multiprocessing import cpu_count, Pool

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

import prnu
import cv2


def get_img_dims(img_path):
    im_arr = cv2.imread(img_path)
    # im = Image.open(img_path)
    # im = ImageOps.exif_transpose(im)
    # im_arr = np.array(im)

    # some videos are not
    shape = im_arr.shape
    if shape[0] > shape[1]:
        im_arr = cv2.rotate(im_arr, cv2.ROTATE_90_CLOCKWISE)
        # print(img_path)

    shape = im_arr.shape
    if shape[0] > shape[1]:
        print(img_path)

    return shape


def main():
    """
    For each device compute the fingerprint from all the training images of that device
    For each test video frame compute the noise residual, and predict the source camera device based on CC & PCE
    Furthermore, combine the frame-level predictions to device level predictions by taking a majority vote
    :return:
    """
    train_filename = r'/scratch/p288722/datasets/vision/old_baseline_split/bal_50_frames/train.json'
    test_filename = r'/scratch/p288722/datasets/vision/old_baseline_split/bal_50_frames/test.json'
    with open(train_filename) as f1, open(test_filename) as f2:
        train_data = json.load(f1)
        test_data = json.load(f2)
    train_images = np.array(sorted([img for device in train_data for img in train_data[device] if 'flat' in img]))
    train_device = np.array([img.split(os.sep)[-3] for img in train_images])

    test_images = np.array(sorted([img for device in test_data for img in test_data[device]]))
    test_device = np.array([img.split(os.sep)[-3] for img in test_images])
    test_videos = np.array([img.split(os.sep)[-2] for img in test_images])

    print('Computing fingerprints')
    fingerprint_device = sorted(np.unique(train_device))
    k = []
    for device in tqdm(fingerprint_device):
        pool = Pool(cpu_count())
        imgs = pool.map(prnu.preprocessing_wrapper, train_images[train_device == device])
        pool.close()
        k += [prnu.extract_multiple_aligned(imgs, processes=cpu_count())]
    k = np.stack(k, 0)

    # print('Computing residuals')
    # pool = Pool(cpu_count())
    # imgs = pool.map(prnu.preprocessing_wrapper, test_images)
    # w = pool.map(prnu.extract_single, imgs)
    # pool.close()
    # w = np.stack(w, 0)

    np.save(r'/scratch/p288722/runtime_data/scd_videos_first_revision/sota_prnu_old_baseline_split/k_flat_50_frames_resize.npy', k)
    # np.save(r'/scratch/p288722/runtime_data/scd_videos_first_revision/sota_prnu_old_baseline_split/w_all_50_frames_resize.npy', w)

    # k = np.load(r'/scratch/p288722/runtime_data/scd_videos_first_revision/sota_prnu_old_baseline_split/k.npy')
    w = np.load(r'/scratch/p288722/runtime_data/scd_videos_first_revision/sota_prnu_old_baseline_split/w_all_50_frames_resize.npy')

    # Computing Ground Truth
    gt = prnu.gt(fingerprint_device, test_device)

    print('Computing cross correlation')
    w_parts = np.array_split(w, len(w) // 100)  # Computing in parts to save memory
    cc_aligned_rot = [prnu.aligned_cc(k, item)['cc'] for item in tqdm(w_parts)]
    cc_aligned_rot = np.hstack(cc_aligned_rot)

    print('Computing statistics cross correlation')
    stats_cc = prnu.stats(cc_aligned_rot, gt)

    # print('Computing PCE')
    # pce_rot = np.zeros((len(fingerprint_device), len(test_device)))
    #
    # for fingerprint_idx, fingerprint_k in enumerate(k):
    #     for natural_idx, natural_w in enumerate(w):
    #         cc2d = prnu.crosscorr_2d(fingerprint_k, natural_w)
    #         pce_rot[fingerprint_idx, natural_idx] = prnu.pce(cc2d)['pce']
    #
    # print('Computing statistics on PCE')
    # stats_pce = prnu.stats(pce_rot, gt)

    print('FRAME LEVEL STATS')

    print('AUC on CC {:.2f}, expected {:.2f}'.format(stats_cc['auc'], 0.98))
    # print('AUC on PCE {:.2f}, expected {:.2f}'.format(stats_pce['auc'], 0.81))

    ground_truth = np.argmax(gt, axis=0)
    pred_cc = np.argmax(cc_aligned_rot, axis=0)
    accuracy_cc = sum(ground_truth == pred_cc) / len(pred_cc)
    # pred_pce = np.argmax(pce_rot, axis=0)
    # accuracy_pce = sum(ground_truth == pred_pce) / len(pred_pce)

    print('Accuracy PCC {:.2f}'.format(accuracy_cc))
    # print('Accuracy PCE {:.2f}'.format(accuracy_pce))

    print('\nVIDEO LEVEL STATS')

    video_level_gt = []
    video_level_pred_cc = []
    video_level_pred_pce = []

    for video in tqdm(np.unique(test_videos)):
        values, counts = np.unique(pred_cc[video == test_videos], return_counts=True)
        pred_cc_class = values[np.argmax(counts)]  # most frequent class
        # values, counts = np.unique(pred_pce[video == test_videos], return_counts=True)
        # pred_pce_class = values[np.argmax(counts)]  # most frequent class
        values, counts = np.unique(ground_truth[video == test_videos], return_counts=True)
        gt_class = values[np.argmax(counts)]  # most frequent class
        assert len(values) == 1  # Sanity check - All the frame labels in the ground truths must belong to one class

        video_level_gt.append(gt_class)
        video_level_pred_cc.append(pred_cc_class)
        # video_level_pred_pce.append(pred_pce_class)

    from sklearn.metrics import accuracy_score
    print('Accuracy PCC {:.3f}'.format(accuracy_score(video_level_gt, video_level_pred_cc)))
    # print('Accuracy PCE {:.3f}'.format(accuracy_score(video_level_gt, video_level_pred_pce)))


if __name__ == '__main__':
    main()
