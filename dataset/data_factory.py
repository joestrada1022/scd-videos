import itertools
import json
import os
import random
import time
from pathlib import Path

import math
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataFactory:

    def __init__(self, width=800, height=480, batch_size=32,
                 input_dir=None):
        with open(Path(input_dir).joinpath('train.json'), 'r') as f:
            self.train_data = json.load(f)
            self.class_names = np.array(sorted(self.train_data.keys()))
            self.train_data = list(itertools.chain.from_iterable(self.train_data.values()))
            random.seed(108)
            random.shuffle(self.train_data)
        with open(Path(input_dir).joinpath('val.json'), 'r') as f:
            self.val_data = json.load(f)
            self.val_data = list(itertools.chain.from_iterable(self.val_data.values()))
            random.seed(108)
            random.shuffle(self.val_data)
        with open(Path(input_dir).joinpath('test.json'), 'r') as f:
            self.test_data = json.load(f)
            self.test_data = list(itertools.chain.from_iterable(self.test_data.values()))
            random.seed(108)
            random.shuffle(self.test_data)

        self.batch_size = batch_size
        self.img_width = width
        self.img_height = height
        self.channels = 3

        # To allow reproducibility
        self.seed = 108

    def get_class_names(self):
        return self.class_names

    def get_tf_input_dim(self):
        return tf.constant((self.img_height, self.img_width), tf.dtypes.int32)

    def process_path(self, file_path):
        label = self.get_label(file_path)
        img = self.load_img(file_path)
        return img, label

    def get_label(self, file_path):
        file_parts = tf.strings.split(file_path, os.path.sep)
        class_name = file_parts[-3]
        one_hot_vec = tf.cast(class_name == self.class_names, dtype=tf.dtypes.float32, name="labels")
        return one_hot_vec

    @staticmethod
    def get_file_name(file_path):
        file_parts = tf.strings.split(file_path, os.path.sep)
        file_name = file_parts[-1]
        return file_name

    def load_img(self, file_path, resize_dim=None, ):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.dtypes.float32)

        # Set to CNN Input Dimensions
        if resize_dim is None:
            resize_dim = self.get_tf_input_dim()
        img = tf.image.resize(img, size=resize_dim)

        # if self.center_crop:
        #     height, width, _ = img.get_shape().as_list()
        #     img = tf.image.crop_to_bounding_box(image=img,
        #                                         offset_height=int(height / 2 - self.img_height / 2),
        #                                         offset_width=int(width / 2 - self.img_width / 2),
        #                                         target_height=480,
        #                                         target_width=800)

        return img

    def get_tf_train_data(self, category):
        t_start = time.time()
        file_path_ds = tf.data.Dataset.from_tensor_slices(self.train_data)
        if category == 'native':
            ds_list = [x for x in file_path_ds if
                       ('WA' not in x.numpy().decode("utf-8")) and ('YT' not in x.numpy().decode("utf-8"))]
            file_path_ds = tf.data.Dataset.from_tensor_slices(ds_list)
        elif category == 'whatsapp':
            ds_list = [x for x in file_path_ds if 'WA' in x.numpy().decode("utf-8")]
            file_path_ds = tf.data.Dataset.from_tensor_slices(ds_list)
        elif category == 'youtube':
            ds_list = [x for x in file_path_ds if 'YT' in x.numpy().decode("utf-8")]
            file_path_ds = tf.data.Dataset.from_tensor_slices(ds_list)

        print(f"Found {len(list(file_path_ds))} images in ({int(time.time() - t_start)} sec.)")

        print(f"\nPrinting first 10 elements of dataset:\n")
        for element in file_path_ds.take(10):
            k = self.process_path(element)
            print(k, element)
            break

        # Load actual images and create labels accordingly
        labeled_ds = file_path_ds.map(self.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        print(f"\nFinished creating labeled dataset ({int(time.time() - t_start)} sec.)\n")

        # Determine number of total elements
        num_elements = tf.data.experimental.cardinality(labeled_ds).numpy()
        print(f"\ntotal number elements: {num_elements} ({int(time.time() - t_start)} sec.)\n")

        # Set batch and prefetch preferences
        labeled_ds = labeled_ds.batch(self.batch_size, drop_remainder=False)
        labeled_ds = labeled_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        num_batches = math.ceil(num_elements / self.batch_size)
        return labeled_ds, num_batches

    def get_tf_evaluation_data(self, category, mode):
        t_start = time.time()

        if mode == 'test':
            file_path_ds = tf.data.Dataset.from_tensor_slices(self.test_data)
        elif mode == 'val':
            file_path_ds = tf.data.Dataset.from_tensor_slices(self.val_data)
        elif mode == 'train':
            file_path_ds = tf.data.Dataset.from_tensor_slices(self.train_data)
        else:
            raise ValueError('Invalid mode')

        if category == 'native':
            ds_list = [x for x in file_path_ds if
                       ('WA' not in x.numpy().decode("utf-8")) and ('YT' not in x.numpy().decode("utf-8"))]
            file_path_ds = tf.data.Dataset.from_tensor_slices(ds_list)
        elif category == 'whatsapp':
            ds_list = [x for x in file_path_ds if 'WA' in x.numpy().decode("utf-8")]
            file_path_ds = tf.data.Dataset.from_tensor_slices(ds_list)
        elif category == 'youtube':
            ds_list = [x for x in file_path_ds if 'YT' in x.numpy().decode("utf-8")]
            file_path_ds = tf.data.Dataset.from_tensor_slices(ds_list)

        print(f"Found {len(list(file_path_ds))} images in ({int(time.time() - t_start)} sec.)")

        # Create labeled dataset by loading the image and estimating the label
        labeled_ds = file_path_ds.map(self.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        labeled_ds = labeled_ds.batch(self.batch_size, drop_remainder=False)
        labeled_ds = labeled_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        print(f"Finished loading test frames ({int(time.time() - t_start)} sec.)")

        # Create dataset of file names which is necessary for evaluation
        filename_ds = file_path_ds.map(self.get_file_name, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return filename_ds, labeled_ds

    def get_tf_val_data(self, category):
        return self.get_tf_evaluation_data(category, mode='val')

    def get_tf_test_data(self, category):
        return self.get_tf_evaluation_data(category, mode='test')

    @staticmethod
    def get_labels(ds):
        # Credits to Guru
        labels_ds = ds.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices(y))
        ground_truth_labels = np.array(list(labels_ds.as_numpy_iterator())).astype(np.int32)
        return ground_truth_labels
