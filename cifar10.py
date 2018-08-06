# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the CIFAR-10 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

 # Compute inference on the model inputs to make a prediction.
 predictions = inference(inputs)

 # Compute the total loss of the prediction with respect to the labels.
 loss = loss(predictions, labels)

 # Create a graph to run one step of training with respect to the loss.
 train_op = train(loss, global_step)
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import cifar10_input

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 64,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', 'tmp/cifar10_data',
                           """Path to the CIFAR-10 data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.001  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels


def inputs(eval_data):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  if not FLAGS.data_dir:
    raise ValueError('Please supply a data_dir')
  data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
  images, labels = cifar10_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=FLAGS.batch_size)
  if FLAGS.use_fp16:
    images = tf.cast(images, tf.float16)
    labels = tf.cast(labels, tf.float16)
  return images, labels

def alexnet(image, keepprob=0.5):

    # 定义卷积层1，卷积核大小，偏置量等各项参数参考下面的程序代码，下同。
    with tf.name_scope("conv1") as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1, name="weights"))
        conv = tf.nn.conv2d(image, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64]), trainable=True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)

        pass

    # LRN层
    lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name="lrn1")

    # 最大池化层
    pool1 = tf.nn.max_pool(lrn1, ksize=[1,3,3,1], strides=[1,2,2,1],padding="VALID", name="pool1")

    # 定义卷积层2
    with tf.name_scope("conv2") as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,64,192], dtype=tf.float32, stddev=1e-1, name="weights"))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[192]), trainable=True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        pass

    # LRN层
    lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name="lrn2")

    # 最大池化层
    pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID", name="pool2")

    # 定义卷积层3
    with tf.name_scope("conv3") as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,192,384], dtype=tf.float32, stddev=1e-1, name="weights"))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[384]), trainable=True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        pass

    # 定义卷积层4
    with tf.name_scope("conv4") as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,384,256], dtype=tf.float32, stddev=1e-1, name="weights"))
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]), trainable=True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        pass

    # 定义卷积层5
    with tf.name_scope("conv5") as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,256,256], dtype=tf.float32, stddev=1e-1, name="weights"))
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding="SAME")
        biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]), trainable=True, name="biases")
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        pass

    # 最大池化层
    pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding="VALID", name="pool5")
    # 全连接层
    flatten = tf.reshape(pool5, [-1, 2*2*256])

    weight1 = tf.Variable(tf.truncated_normal([2*2*256, 1024], mean=0, stddev=0.01))

    fc1 = tf.nn.sigmoid(tf.matmul(flatten, weight1))

    dropout1 = tf.nn.dropout(fc1, keepprob)

    weight2 = tf.Variable(tf.truncated_normal([1024, 1024], mean=0, stddev=0.01))

    fc2 = tf.nn.sigmoid(tf.matmul(dropout1, weight2))

    dropout2 = tf.nn.dropout(fc2, keepprob)

    weight3 = tf.Variable(tf.truncated_normal([1024, 10], mean=0, stddev=0.01))

    fc3 = tf.nn.sigmoid(tf.matmul(dropout2, weight3))

    return fc3
def inference(images):
  """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 3, 64],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = _variable_with_weight_decay('weights',
                                         shape=[5, 5, 64, 64],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    _activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1], padding='SAME', name='pool2')

  # local3
  with tf.variable_scope('local3') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1])
    dim = reshape.get_shape()[1].value
    weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    _activation_summary(local3)

  # local4
  with tf.variable_scope('local4') as scope:
    weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                          stddev=0.04, wd=0.004)
    biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
    _activation_summary(local4)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                          stddev=1/192.0, wd=None)
    biases = _variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
    _activation_summary(softmax_linear)

  return softmax_linear


def conv2d(_x, _w, _b):
    return tf.nn.bias_add(tf.nn.conv2d(_x, _w, [1, 1, 1, 1], padding='SAME'), _b)


def max_pool(_x, f):
    return tf.nn.max_pool(_x, [1, f, f, 1], [1, 1, 1, 1], padding='SAME')


def lrn(_x):
    return tf.nn.lrn(_x, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


def init_w(namespace, shape, wd, stddev, reuse=False):
    with tf.variable_scope(namespace, reuse=reuse):
        initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=stddev)
        w = tf.get_variable("w", shape=shape, initializer=initializer)

        if wd:
            weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
    return w


def init_b(namespace, shape, reuse=False):
    with tf.variable_scope(namespace, reuse=reuse):
        initializer = tf.constant_initializer(0.0)
        b = tf.get_variable("b", shape=shape, initializer=initializer)
    return b


def batch_normal(xs, out_size):
    axis = list(range(len(xs.get_shape()) - 1))
    n_mean, n_var = tf.nn.moments(xs, axes=axis)
    scale = tf.Variable(tf.ones([out_size]))
    shift = tf.Variable(tf.zeros([out_size]))
    epsilon = 0.001
    ema = tf.train.ExponentialMovingAverage(decay=0.9)

    def mean_var_with_update():
        ema_apply_op = ema.apply([n_mean, n_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(n_mean), tf.identity(n_var)

    mean, var = mean_var_with_update()

    bn = tf.nn.batch_normalization(xs, mean, var, shift, scale, epsilon)
    return bn


def alexnet_cifar(images, reuse=False):
    '''Build the network model and return logits'''

    # conv1
    w1 = init_w("conv1", [3, 3, 3, 24], None, 0.01, reuse)
    bw1 = init_b("conv1", [24], reuse)
    conv1 = conv2d(images, w1, bw1)
    bn1 = batch_normal(conv1, 24)
    c_output1 = tf.nn.relu(bn1)
    pool1 = max_pool(c_output1, 2)

    # conv2
    w2 = init_w("conv2", [3, 3, 24, 96], None, 0.01, reuse)
    bw2 = init_b("conv2", [96], reuse)
    conv2 = conv2d(pool1, w2, bw2)
    bn2 = batch_normal(conv2, 96)
    c_output2 = tf.nn.relu(bn2)
    pool2 = max_pool(c_output2, 2)

    # conv3
    w3 = init_w("conv3", [3, 3, 96, 192], None, 0.01, reuse)
    bw3 = init_b("conv3", [192], reuse)
    conv3 = conv2d(pool2, w3, bw3)
    bn3 = batch_normal(conv3, 192)
    c_output3 = tf.nn.relu(bn3)

    # conv4
    w4 = init_w("conv4", [3, 3, 192, 192], None, 0.01, reuse)
    bw4 = init_b("conv4", [192], reuse)
    conv4 = conv2d(conv3, w4, bw4)
    bn4 = batch_normal(conv4, 192)
    c_output4 = tf.nn.relu(bn4)

    # conv5
    w5 = init_w("conv5", [3, 3, 192, 96], None, 0.01, reuse)
    bw5 = init_b("conv5", [96], reuse)
    conv5 = conv2d(conv4, w5, bw5)
    bn5 = batch_normal(conv5, 96)
    c_output5 = tf.nn.relu(bn5)
    pool5 = max_pool(c_output5, 2)
    print(c_output5.shape)
    # FC1
    wfc1 = init_w("fc1", [96 * 24 * 24, 1024], None, 1e-2, reuse)
    bfc1 = init_b("fc1", [1024], reuse)
    shape = pool5.get_shape()
    reshape = tf.reshape(pool5, [-1, shape[1].value * shape[2].value * shape[3].value])
    w_x1 = tf.matmul(reshape, wfc1) + bfc1
    bn6 = batch_normal(w_x1, 1024)
    fc1 = tf.nn.relu(bn6)

    # FC2
    wfc2 = init_w("fc2", [1024, 1024], None, 1e-2, reuse)
    bfc2 = init_b("fc2", [1024], reuse)
    w_x2 = tf.matmul(fc1, wfc2) + bfc2
    bn7 = batch_normal(w_x2, 1024)
    fc2 = tf.nn.relu(bn7)

    # FC3
    wfc3 = init_w("fc3", [1024, 10], None, 1e-2, reuse)
    bfc3 = init_b("fc3", [10], reuse)
    softmax_linear = tf.add(tf.matmul(fc2, wfc3), bfc3)


    return softmax_linear
def alexnet_cifar_attention(images, reuse=False):
    '''Build the network model and return logits'''

    # conv1
    w1 = init_w("conv1", [3, 3, 3, 24], None, 0.01, reuse)
    bw1 = init_b("conv1", [24], reuse)
    conv1 = conv2d(images, w1, bw1)
    bn1 = batch_normal(conv1, 24)
    c_output1 = tf.nn.relu(bn1)
    pool1 = max_pool(c_output1, 2)

    # conv2
    w2 = init_w("conv2", [3, 3, 24, 96], None, 0.01, reuse)
    bw2 = init_b("conv2", [96], reuse)
    conv2 = conv2d(pool1, w2, bw2)
    bn2 = batch_normal(conv2, 96)
    c_output2 = tf.nn.relu(bn2)
    pool2 = max_pool(c_output2, 2)

    # conv3
    w3 = init_w("conv3", [3, 3, 96, 192], None, 0.01, reuse)
    bw3 = init_b("conv3", [192], reuse)
    conv3 = conv2d(pool2, w3, bw3)
    bn3 = batch_normal(conv3, 192)
    c_output3 = tf.nn.relu(bn3)

    # conv4
    w4 = init_w("conv4", [3, 3, 192, 192], None, 0.01, reuse)
    bw4 = init_b("conv4", [192], reuse)
    conv4 = conv2d(conv3, w4, bw4)
    bn4 = batch_normal(conv4, 192)
    c_output4 = tf.nn.relu(bn4)

    # conv5
    w5 = init_w("conv5", [3, 3, 192, 96], None, 0.01, reuse)
    bw5 = init_b("conv5", [96], reuse)
    conv5 = conv2d(conv4, w5, bw5)
    bn5 = batch_normal(conv5, 96)
    c_output5 = tf.nn.relu(bn5)
    pool5 = max_pool(c_output5, 2)
    #attention
    w_1 = init_w("conv_1", [1, 1, 96, 48], None, 0.01, reuse)
    b_1 = init_b("conv_1", [48], reuse)
    conv_1 = conv2d(pool5, w_1, b_1)
    c_output_1 = tf.nn.relu(conv_1)

    w_2 = init_w("conv_2", [1, 1, 48, 24], None, 0.01, reuse)
    b_2 = init_b("conv_2", [24], reuse)
    conv_2 = conv2d(c_output_1, w_2, b_2)
    c_output_2 = tf.nn.relu(conv_2)

    w_3 = init_w("conv_3", [1, 1, 24, 2], None, 0.01, reuse)
    b_3 = init_b("conv_3", [2], reuse)
    c_output_3 = conv2d(c_output_2, w_3, b_3)
    # x = pool5.reshape(pool5.size(0), pool5.size(3), -1, 1)
    x=tf.reshape(pool5,[pool5.shape[0], pool5.shape[3], -1, 1])
    y=c_output_3
    y = tf.reshape(y, [x.shape[0]*2, -1])
    y = tf.nn.softmax(y)
    # y = y.view(x.size(0), 2, -1, 1)
    y = tf.reshape(y, [x.shape[0], 2, -1, 1])
    feature_list = []
    for i in range(2):
        y_mask = tf.slice(y,begin=[0,i,0,0],size=[x.shape[0],1,x.shape[2],1])
        y_mask = y_mask * x
        # y_mask = torch.sum(y_mask, 2, keepdim=True)
        feature_list.append(y_mask)
    feature_list.append(x)
    y_feature_concat = tf.concat(feature_list, 1)
    y_feature_concat = tf.squeeze(y_feature_concat)
    # y_feature_concat = y_feature_concat.view(x.size(0), 768 * 6 * 6)
    y_feature_concat = tf.reshape(y_feature_concat, [x.shape[0],288*24*24])
    # print(y_feature_concat.size())



    # FC1
    wfc1 = init_w("fc1", [288 * 24 * 24, 1024], None, 1e-2, reuse)
    bfc1 = init_b("fc1", [1024], reuse)
    w_x1 = tf.matmul(y_feature_concat, wfc1) + bfc1
    bn6 = batch_normal(w_x1, 1024)
    fc1 = tf.nn.relu(bn6)

    # FC2
    wfc2 = init_w("fc2", [1024, 1024], None, 1e-2, reuse)
    bfc2 = init_b("fc2", [1024], reuse)
    w_x2 = tf.matmul(fc1, wfc2) + bfc2
    bn7 = batch_normal(w_x2, 1024)
    fc2 = tf.nn.relu(bn7)

    # FC3
    wfc3 = init_w("fc3", [1024, 10], None, 1e-2, reuse)
    bfc3 = init_b("fc3", [10], reuse)
    softmax_linear = tf.add(tf.matmul(fc2, wfc3), bfc3)


    return softmax_linear

def alexnet_cifar_FC(images, reuse=False):
        '''Build the network model and return logits'''

        # conv1
        w1 = init_w("conv1", [3, 3, 3, 24], None, 0.01, reuse)
        bw1 = init_b("conv1", [24], reuse)
        conv1 = conv2d(images, w1, bw1)
        bn1 = batch_normal(conv1, 24)
        c_output1 = tf.nn.relu(bn1)
        pool1 = max_pool(c_output1, 2)

        # conv2
        w2 = init_w("conv2", [3, 3, 24, 96], None, 0.01, reuse)
        bw2 = init_b("conv2", [96], reuse)
        conv2 = conv2d(pool1, w2, bw2)
        bn2 = batch_normal(conv2, 96)
        c_output2 = tf.nn.relu(bn2)
        pool2 = max_pool(c_output2, 2)

        # conv3
        w3 = init_w("conv3", [3, 3, 96, 192], None, 0.01, reuse)
        bw3 = init_b("conv3", [192], reuse)
        conv3 = conv2d(pool2, w3, bw3)
        bn3 = batch_normal(conv3, 192)
        c_output3 = tf.nn.relu(bn3)

        # conv4
        w4 = init_w("conv4", [3, 3, 192, 192], None, 0.01, reuse)
        bw4 = init_b("conv4", [192], reuse)
        conv4 = conv2d(conv3, w4, bw4)
        bn4 = batch_normal(conv4, 192)
        c_output4 = tf.nn.relu(bn4)

        # conv5
        w5 = init_w("conv5", [3, 3, 192, 96], None, 0.01, reuse)
        bw5 = init_b("conv5", [96], reuse)
        conv5 = conv2d(conv4, w5, bw5)
        bn5 = batch_normal(conv5, 96)
        c_output5 = tf.nn.relu(bn5)
        pool5 = max_pool(c_output5, 2)
        w6 = init_w("conv6", [3, 3, 96, 48], None, 0.01, reuse)
        bw6 = init_b("conv6", [48], reuse)
        conv6=tf.nn.bias_add(tf.nn.conv2d(pool5, w6, [1, 2, 2, 1], padding='VALID'),bw6)

        w7 = init_w("conv7", [11, 11, 48, 10], None, 0.01, reuse)
        bw7 = init_b("conv7", [10], reuse)
        conv7 = tf.nn.bias_add(tf.nn.conv2d(conv6, w7, [1, 2, 2, 1], padding='VALID'),bw7)

        w8 = init_w("conv8", [1, 1, 10, 10], None, 0.01, reuse)
        bw8 = init_b("conv8", [10], reuse)
        net = tf.nn.bias_add(tf.nn.conv2d(conv7, w8, [1, 1, 1, 1], padding='VALID'),bw8)
        logits = tf.squeeze(net, [1,2], name='logits')
        logits=tf.nn.softmax(logits)


        return logits
def weight_variable(shape):
        """weight_variable generates a weight variable of a given shape."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
def identity_block(X_input, kernel_size, in_filter, out_filters, stage, block, training):
        # defining name basis
        block_name = 'res' + str(stage) + block
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input

            #first
            W_conv1 = weight_variable([1, 1, in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #second
            W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #third

            W_conv3 = weight_variable([1, 1, f2, f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            #final step
            add = tf.add(X, X_shortcut)
            add_result = tf.nn.relu(add)

        return add_result


def convolutional_block( X_input, kernel_size, in_filter,
                        out_filters, stage, block, training, stride=2):
    # defining name basis
    block_name = 'res' + str(stage) + block
    with tf.variable_scope(block_name):
        f1, f2, f3 = out_filters

        x_shortcut = X_input
        # first
        W_conv1 = weight_variable([1, 1, in_filter, f1])
        X = tf.nn.conv2d(X_input, W_conv1, strides=[1, stride, stride, 1], padding='VALID')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        # second
        W_conv2 = weight_variable([kernel_size, kernel_size, f1, f2])
        X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
        X = tf.layers.batch_normalization(X, axis=3, training=training)
        X = tf.nn.relu(X)

        # third
        W_conv3 = weight_variable([1, 1, f2, f3])
        X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
        X = tf.layers.batch_normalization(X, axis=3, training=training)

        # shortcut path
        W_shortcut = weight_variable([1, 1, in_filter, f3])
        x_shortcut = tf.nn.conv2d(x_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

        # final
        add = tf.add(x_shortcut, X)
        add_result = tf.nn.relu(add)

    return add_result

def resnet_50(x_input, classes=10,is_training=True):
        """
        Implementation of the popular ResNet50 the following architecture:
        CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
        -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

        Arguments:

        Returns:
        """
        x = tf.pad(x_input, tf.constant([[0, 0], [3, 3, ], [3, 3], [0, 0]]), "CONSTANT")
        with tf.variable_scope('reference'):
            training =is_training

            # stage 1
            w_conv1 = weight_variable([7, 7, 3, 64])
            x = tf.nn.conv2d(x, w_conv1, strides=[1, 2, 2, 1], padding='VALID')
            x = tf.layers.batch_normalization(x, axis=3, training=training)
            x = tf.nn.relu(x)

            x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='VALID')
            assert (x.get_shape() == (x.get_shape()[0], 15, 15, 64))

            # stage 2
            x = convolutional_block(x, 3, 64, [64, 64, 256], 2, 'a', training, stride=1)
            x = identity_block(x, 3, 256, [64, 64, 256], stage=2, block='b', training=training)
            x = identity_block(x, 3, 256, [64, 64, 256], stage=2, block='c', training=training)

            # stage 3
            x = convolutional_block(x, 3, 256, [128, 128, 512], 3, 'a', training)
            x = identity_block(x, 3, 512, [128, 128, 512], 3, 'b', training=training)
            x = identity_block(x, 3, 512, [128, 128, 512], 3, 'c', training=training)
            x = identity_block(x, 3, 512, [128, 128, 512], 3, 'd', training=training)

            # stage 4
            x = convolutional_block(x, 3, 512, [256, 256, 1024], 4, 'a', training)
            x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'b', training=training)
            x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'c', training=training)
            x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'd', training=training)
            x = identity_block(x, 3, 1024, [256, 256, 1024], 4, 'e', training=training)
            x =identity_block(x, 3, 1024, [256, 256, 1024], 4, 'f', training=training)

            # stage 5
            x = convolutional_block(x, 3, 1024, [512, 512, 2048], 5, 'a', training)
            x = identity_block(x, 3, 2048, [512, 512, 2048], 5, 'b', training=training)
            x = identity_block(x, 3, 2048, [512, 512, 2048], 5, 'c', training=training)
            print(x.shape)
            x = tf.nn.avg_pool(x, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
            flatten = tf.layers.flatten(x)
            x = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)
            # Dropout - controls the complexity of the model, prevents co-adaptation of
            # features.
            with tf.name_scope('dropout'):
                keep_prob = 0.5
                x = tf.nn.dropout(x, keep_prob)

            logits = tf.layers.dense(x, units=10, activation=tf.nn.softmax)

        return logits
def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  # logits=tf.nn.softmax(logits)
  predictions=tf.argmax(logits,axis=1)
  acc= tf.reduce_mean(tf.to_float(tf.equal(predictions,labels)))
  tf.add_to_collection('losses', cross_entropy_mean)
  tf.summary.scalar('train_acc', acc)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss'),acc

def my_accuracy(logits, labels):
    labels = tf.cast(labels, tf.int64)
    predictions = tf.argmax(logits, axis=1)
    acc = tf.reduce_mean(tf.to_float(tf.equal(predictions, labels)))
    return acc
def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name + ' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op


def train(total_loss, global_step):
  """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
  # Variables that affect learning rate.
  num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
  decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

  # Decay the learning rate exponentially based on the number of steps.
  lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                  global_step,
                                  decay_steps,
                                  LEARNING_RATE_DECAY_FACTOR,
                                  staircase=True)
  tf.summary.scalar('learning_rate', lr)
  tf.add_to_collection('learning_rate', lr)
  # Generate moving averages of all losses and associated summaries.
  loss_averages_op = _add_loss_summaries(total_loss)

  # Compute gradients.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies([loss_averages_op]):
      with tf.control_dependencies(update_ops):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  # Add histograms for trainable variables.
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name, var)

  # Add histograms for gradients.
  for grad, var in grads:
    if grad is not None:
      tf.summary.histogram(var.op.name + '/gradients', grad)

  # Track the moving averages of all trainable variables.
  variable_averages = tf.train.ExponentialMovingAverage(
      MOVING_AVERAGE_DECAY, global_step)
  with tf.control_dependencies([apply_gradient_op]):
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

  return variables_averages_op

def read_and_decode(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                   features={'label': tf.FixedLenFeature([], tf.int64),
                                             'image': tf.FixedLenFeature([], tf.string),})
    img = tf.decode_raw(features['image'], tf.uint8)
    img = tf.reshape(img, [32, 32, 3])
    img = tf.random_crop(img, [24, 24, 3])

    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return img, label


def maybe_download_and_extract():
  """Download and extract the tarball from Alex's website."""
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
