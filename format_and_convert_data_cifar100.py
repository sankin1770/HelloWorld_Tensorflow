"""
    Format Market-1501 training images and convert all the splits into TFRecords
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import convert_to_tfrecords_cifar100
from datasets.utils import *

FLAGS = tf.app.flags.FLAGS
#CIFAR100
split_name='train'#test train
tf.app.flags.DEFINE_string('image_dir', 'D:/杭电/范老师/最新/Deep-Mutual-Learning-master/cifar100/'+split_name, None)
tf.app.flags.DEFINE_string('output_dir', 'D:/杭电/范老师/最新/Deep-Mutual-Learning-master/cifar100/tfrecord/', None)
tf.app.flags.DEFINE_string('split_name', split_name, None)

def main(_):

    convert_to_tfrecords_cifar100.run(image_dir=FLAGS.image_dir,
                             output_dir=FLAGS.output_dir,
                             split_name=FLAGS.split_name)


if __name__ == '__main__':
    tf.app.run()

