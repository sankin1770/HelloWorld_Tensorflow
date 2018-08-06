"""
    Convert Fashion60  to TFRecords of TF-Example protos.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import sys
from scipy import misc
from datasets.dataset_utils import *

# resize all the re-id images into the same size
#for CIFAR100
_IMAGE_HEIGHT = 250
_IMAGE_WIDTH = 200
_IMAGE_CHANNELS = 3

def _add_to_tfrecord(list_filename_1, list_filename_2, tfrecord_writer):
    """Loads images and writes files to a TFRecord.

    Args:
      image_dir: The image directory where the raw images are stored.
      list_filename: The list file of images.deep
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    num_images_1 = len(tf.gfile.FastGFile(list_filename_1, 'r').readlines())
    num_images_2 = len(tf.gfile.FastGFile(list_filename_2, 'r').readlines())
    assert num_images_1 == num_images_2

    # def set():
    #     label_1 = ""
    #     img = ''
    #     label_2 = ''
    #
    #     label_coat = tf.where(tf.equal(label_1, 1))
    #     img = tf.gather(img, label_coat)
    #     label_2 = tf.gather(label_2, label_coat)

    shape = (_IMAGE_HEIGHT, _IMAGE_WIDTH, _IMAGE_CHANNELS)
    with tf.Graph().as_default():
        image = tf.placeholder(dtype=tf.uint8, shape=shape)
        encoded_png = tf.image.encode_jpeg(image)
        j = 0
        count = 0
        with tf.Session('') as sess:
            for line1, line2 in zip(
                    tf.gfile.FastGFile(list_filename_1, 'r').readlines(),
                    tf.gfile.FastGFile(list_filename_2, 'r').readlines()):
                count += 1
                if count % 100 == 0:
                    print('>> Converting image %d/%d' % (j + 1, num_images_1))

                j += 1
                imagename, label_1 = line1.split(' ')
                imagename, label_2 = line2.split(' ')
                label_1 = int(label_1)
                label_2 = int(label_2)
                file_path = imagename
                image_data = misc.imread(file_path)
                image_data = misc.bytescale(image_data)
                image_data = misc.imresize(image_data, [_IMAGE_HEIGHT, _IMAGE_WIDTH])

                if len(image_data.shape) < 3:
                    image_data = np.reshape(image_data, [_IMAGE_HEIGHT, _IMAGE_WIDTH, 1])
                    image_data = np.tile(image_data, [1, 1, 3])
                elif image_data.shape[-1] == 1:
                    image_data = np.tile(image_data, [1, 1, 3])

                png_string = sess.run(encoded_png, feed_dict={image: image_data})
                example = image_to_tfexample(png_string, label_1,
                                             label_2, bytes(imagename,'utf-8'), _IMAGE_HEIGHT, _IMAGE_WIDTH, b'jpg')
                tfrecord_writer.write(example.SerializeToString())


def run(label_1_dir, label_2_dir, tfrecord_dir):
    """Convert images to tfrecords.
    Args:
    image_dir: The image directory where the raw images are stored.
    output_dir: The directory where the lists and tfrecords are stored.
    split_name: The split name of dataset.
    """
    list1_filename = label_1_dir
    list2_filename = label_2_dir
    # tf_filename = tfrecord_dir
    tf_filename = os.path.join(tfrecord_dir, 'test.tfrecord')
    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        _add_to_tfrecord(list1_filename, list2_filename, tfrecord_writer)

    print(" Done! \n")

