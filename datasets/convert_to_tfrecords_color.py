"""
    Convert Market-1501 to TFRecords of TF-Example protos.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
import sys
from scipy import misc
from datasets.dataset_utils import *

# resize all the re-id images into the same size
#for Market1501
_IMAGE_HEIGHT = 256
_IMAGE_WIDTH = 256
_IMAGE_CHANNELS = 3

def _add_to_tfrecord(image_dir, tfrecord_writer, split_name):
    """Loads images and writes files to a TFRecord.

    Args:
      image_dir: The image directory where the raw images are stored.
      list_filename: The list file of images.
      tfrecord_writer: The TFRecord writer to use for writing.
    """
    # filenames = tf.train.match_filenames_once(os.path.join(image_dir,'\*.jpg'))
    list=os.listdir(image_dir)
    i=0
    count = 0
    for  filename in list:
        i+=1

        filenames=os.listdir(os.path.join(image_dir+"/",filename))

        nums=len(filenames)
        count+=nums
        shape = (_IMAGE_HEIGHT, _IMAGE_WIDTH, _IMAGE_CHANNELS)
        with tf.Graph().as_default():
            image = tf.placeholder(dtype=tf.uint8, shape=shape)
            encoded_png = tf.image.encode_jpeg(image)
            j = 0

            with tf.Session('') as sess:
                for line in filenames:
                    sys.stdout.write('\r>> Converting %s%s image %d/%d  now%d images' % (filename,split_name, j + 1,nums,count))
                    sys.stdout.flush()
                    j += 1
                    image_data = misc.imread(os.path.join(image_dir+"/"+filename+"/",line))
                    label=i-1
                    image_data=misc.bytescale(image_data)
                    image_data = misc.imresize(image_data, [_IMAGE_HEIGHT, _IMAGE_WIDTH])
                    png_string = sess.run(encoded_png, feed_dict={image: image_data})
                    example = image_to_tfexample(png_string, label, bytes(line,'utf-8'), _IMAGE_HEIGHT, _IMAGE_WIDTH, b'jpg')
                    tfrecord_writer.write(example.SerializeToString())


def run(image_dir, output_dir, split_name):
    """Convert images to tfrecords.
    Args:
    image_dir: The image directory where the raw images are stored.
    output_dir: The directory where the lists and tfrecords are stored.
    split_name: The split name of dataset.
    """
    tf_filename = os.path.join(output_dir, '%s.tfrecord' % split_name)

    if tf.gfile.Exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
        _add_to_tfrecord(image_dir, tfrecord_writer, split_name)

    print(" Done! \n")

