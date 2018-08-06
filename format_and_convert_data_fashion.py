"""
    Format Market-1501 training images and convert all the splits into TFRecords
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import convert_to_tfrecords_fashion
from datasets.utils import *

FLAGS = tf.app.flags.FLAGS
#CIFAR100
# split_name='coat'#test train
# tf.app.flags.DEFINE_string('image_dir', '/home/sankin/MyFiles/Fashion60/'+split_name, None)
# tf.app.flags.DEFINE_string('output_dir', '/home/sankin/MyFiles/Fashion60/', None)
# tf.app.flags.DEFINE_string('split_name', split_name, None)

def main(_):

    convert_to_tfrecords_fashion.run(label_1_dir='/home/sankin/MyFiles/Fashion60/Clothes-test-coarse-shuffle.txt',
                                     label_2_dir='/home/sankin/MyFiles/Fashion60/Clothes-test-fine-shuffle.txt',
                                     tfrecord_dir='/home/sankin/MyFiles/Fashion/')


if __name__ == '__main__':
    tf.app.run()

