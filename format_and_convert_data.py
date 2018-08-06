"""
    Format Market-1501 training images and convert all the splits into TFRecords
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import convert_to_tfrecords
from datasets import format_market_train
from datasets import make_filename_list
from datasets.utils import *

FLAGS = tf.app.flags.FLAGS
split_name='bounding_box_test'#bounding_box_train bounding_box_test gt_bbox query
tf.app.flags.DEFINE_string('image_dir', 'D:/杭电/范老师/最新/Deep-Mutual-Learning-master/Market-1501-v15.09.15/'+split_name, None)
tf.app.flags.DEFINE_string('output_dir', 'D:/杭电/范老师/最新/Deep-Mutual-Learning-master/tfrecords_output/', None)
tf.app.flags.DEFINE_string('split_name', split_name, None)


def main(_):

    #mkdir_if_missing(FLAGS.output_dir)

    if FLAGS.split_name == 'bounding_box_train':
        format_market_train.run(image_dir=FLAGS.image_dir)

    make_filename_list.run(image_dir=FLAGS.image_dir,
                           output_dir=FLAGS.output_dir,
                           split_name=FLAGS.split_name)

    convert_to_tfrecords.run(image_dir=FLAGS.image_dir,
                             output_dir=FLAGS.output_dir,
                             split_name=FLAGS.split_name)


if __name__ == '__main__':
    tf.app.run()

