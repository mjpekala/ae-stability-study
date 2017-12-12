
import os

import numpy as np

import tensorflow as tf
import cifar10
from cifar10_input import read_cifar10

import pdb


class Cifar10:
  def __init__(self, sess, checkpoint_dir='./Weights/cifar10_tf'):
    # Note: the images are cropped prior to training.
    #       Hence, the non-standard CIFAR10 image sizes below.
    self.batch_shape = [128, 24, 24, 3]
    self.num_classes = 10

    #self.x_tf, self.y_tf = cifar10.inputs(eval_data='test')
    self.x_tf = tf.placeholder(tf.float32, shape=self.batch_shape)
    self.y_tf = tf.placeholder(tf.int32, shape=[self.batch_shape[0],])  # evidently these are not one-hot
    self.output = cifar10.inference(self.x_tf)

    self.loss = cifar10.loss(self.output, self.y_tf)
    self.loss_x = tf.gradients(self.loss, self.x_tf)[0]

    # load weights
    # note: the directory must also contain the file "checkpoint"
    #       for this to work
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
      variables_to_restore = variable_averages.variables_to_restore()
      saver = tf.train.Saver(variables_to_restore)
      saver.restore(sess, ckpt.model_checkpoint_path)
      print('[CIFAR10] restored model weighs')
    else:
      print('[CIFAR10] model checkpoint path "%s" is not valid' % checkpoint_dir)



#def _read_cifar_data(filename):
#  "Loads binary CIFAR-10 data into a numpy tensor"
#  with open(filename, 'rb') as f:
#    raw_data = f.read()
#  bytes_per_record = 32*32*3 + 1 # the +1 is for the label
#  arr = array.array('B', raw_data)
#  n_examples = int(len(arr) / bytes_per_record)
#  assert(n_examples * bytes_per_record == len(arr))
#
#  y = np.zeros((n_examples,))
#  x = np.zeros((n_examples, 32, 32, 3))
#
#  for idx, raw_idx in enumerate(range(0, len(raw_data), bytes_per_record)):
#    data_block = raw_data[raw_idx:(raw_idx+bytes_per_record)]
#    y[idx] = data_block[0]
#    xi = np.frombuffer(data_block[1:], dtype=np.uint8)
#    xi = np.reshape(xi, (3, 32, 32)).astype(np.float32)
#    xi = np.transpose(xi, [1,2,0])
#    xi = np.roll(xi, 2, axis=2)
#    x[idx,...] = xi
#
#  return x, y


def _load_cifar10_python(filename, preprocess=False):
  """ Loads CIFAR10 data from file (python format).

   Reference:
     https://www.cs.toronto.edu/~kriz/cifar.html
  """
  import pickle
  with open(test_data_file, 'rb') as f:
    d = pickle.load(f, encoding='bytes')
    x = d[b'data']
    y = np.array(d[b'labels'])

  x = np.reshape(x, (x.shape[0], 3, 32, 32))  # R, G, B
  x = np.transpose(x, [0,2,3,1])              # channels last
  x = x.astype(np.float32) / 255.

  assert(np.max(y) == 9)
  assert(np.min(y) == 0)

  if preprocess:
    # data preprocessing:
    #  1. crop the data to appropriate size for CNN
    #  2. zero mean, unit variance
    x = x[:,4:-4, 4:-4, :]
    x -= np.mean(x, axis=0, keepdims=True)
    x = x / np.std(x, axis=0, keepdims=True)

  return x,y


def _eval_model(sess, model, x, y):
  n_in_batch = model.batch_shape[0]
  n_batches = int(x.shape[0] / n_in_batch)

  y_hat = np.zeros((n_in_batch*n_batches,))

  for ii in range(n_batches):
    a, b = ii*n_in_batch, (ii+1)*n_in_batch
    x_mb = x[a:b,...]
    y_mb = y[a:b]

    pred = sess.run(model.output, feed_dict={model.x_tf : x_mb})
    y_hat[a:b] = np.argmax(pred,axis=1)

  print('accuracy on CIFAR10 test: %0.2f%%' % (100.*np.sum(y_hat == y[:n_in_batch*n_batches]) / y_hat.size))

  return y_hat



if __name__ == "__main__":

  with tf.Graph().as_default(), tf.Session() as sess:
    model = Cifar10(sess, '../../Weights/cifar10_tf')

    # test model on some data
    # https://www.cs.toronto.edu/~kriz/cifar.html
    test_data_file = '/home/pekalmj1/Data/CIFAR10/cifar-10-batches-py/test_batch'
    x,y = _load_cifar10_python(test_data_file, preprocess=True)

    _eval_model(sess, model, x, y)

