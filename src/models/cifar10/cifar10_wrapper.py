"""
  !!! NOTE !!!
   1.  This script assumes you ahve already trained the model
       (using cifar10_train.py).
   2.  This script assumes you are running from the project root directory
       e.g. via
         python -m models.cifar10.cifar10_wrapper.py

"""

__author__ = "mjp"
__date__ = 'december, 2017'


import sys, os
import pdb
import h5py

import numpy as np

import tensorflow as tf

from . import cifar10
from .cifar10_input import read_cifar10

from ae_utils import to_one_hot


class Cifar10(object):
  """ Wrapper around CIFAR-10 model.
  """

  def __init__(self, sess, checkpoint_file_or_dir='./Weights/cifar10_tf/model.ckpt-970664'):
    # Note: the images are cropped prior to training.
    #       Hence, the non-standard CIFAR10 image sizes below.
    #
    self.batch_shape = [128, 24, 24, 3]
    self.num_classes = 10

    #self.x_tf, self.y_tf = cifar10.inputs(eval_data='test')
    self.x_tf = tf.placeholder(tf.float32, shape=self.batch_shape)
    self.y_tf = tf.placeholder(tf.int32, shape=[self.batch_shape[0],10])  

    self.output = cifar10.inference(self.x_tf)  # logits!

    # Note: we do *not* use the original network's loss here, since that contains 
    #       weight decay terms that do not really apply here.
    #
    # Instead, we build our own custom loss function for use with AE.
    #
    self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y_tf)
    if False:
      self.loss = tf.reduce_mean(self.loss)

    self.loss_x = tf.gradients(self.loss, self.x_tf)[0]

    # load weights
    # note: the directory must also contain the file "checkpoint"
    #       for this to work
    variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    if not os.path.isdir(checkpoint_file_or_dir):
      saver.restore(sess, checkpoint_file_or_dir)
    else:
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
      saver.restore(sess, ckpt.model_checkpoint_path)


  def __call__(self, x_input):
    "This method exists for use by cleverhans."
    return self.output



def load_cifar10_python(filename, preprocess=True):
  """ Loads CIFAR10 data from file (python format).

   Reference:
     https://www.cs.toronto.edu/~kriz/cifar.html
  """
  import pickle
  with open(filename, 'rb') as f:
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
  # Note: In the following, we toss the last few examples if 
  #       the data set size is not a multiple of the batch size.
  #
  n_in_batch = model.batch_shape[0]
  n_batches = int(x.shape[0] / n_in_batch)
  if n_in_batch * n_batches < x.shape[0]:
    n_batches += 1

  #--------------------------------------------------
  # Test on clean/original data
  # The accuracy here should be ~83% or so
  #--------------------------------------------------
  y_hat = np.zeros(y.shape)
  x_mb = np.zeros((n_in_batch,) + x.shape[1:], dtype=np.float32)

  for ii in range(n_batches):
    a = ii * n_in_batch
    b = min(x.shape[0], (ii+1) * n_in_batch)
    x_mb[0:(b-a),...] = x[a:b,...]

    pred = sess.run(model.output, feed_dict={model.x_tf : x_mb})
    y_hat[a:b] = np.argmax(pred[0:(b-a),:],axis=1)

  acc = 100. * np.sum(y_hat == y) / y.size

  return y_hat, acc



def _fgsm_attack(sess, model, x, y, eps, use_cleverhans=False):
  """
   Craft adversarial examples using Fast Gradient Sign Method (FGSM)
  """
  n_in_batch = model.batch_shape[0]
  n_batches = int(x.shape[0] / n_in_batch)
  if n_in_batch * n_batches < x.shape[0]:
    n_batches += 1

  x_adv = np.zeros(x.shape)

  if use_cleverhans:
    from cleverhans import utils_tf
    from cleverhans.model import CallableModelWrapper
    from cleverhans.attacks import FastGradientMethod

    fgsm = FastGradientMethod(model, sess=sess)
    x_adv_tf = fgsm.generate(model.x_tf, eps=.05, clip_min=0, clip_max=1.)

    # TODO: finish this!
    #eval_params = {'batch_size': n_in_batch}
    #x_adv, = batch_eval(sess, [model.x_tf], [adv_x_tf], [x], args=eval_params)

  else:
    x_mb = np.zeros((n_in_batch,) + x.shape[1:], dtype=np.float32)
    y_mb = np.zeros((n_in_batch,10), dtype=np.float32)

    for ii in range(n_batches):
      a = ii * n_in_batch
      b = min(x.shape[0], (ii+1) * n_in_batch)

      x_mb[0:(b-a),...] = x[a:b,...]
      y_mb[0:(b-a), :] = to_one_hot(y[a:b], 10)

      # Note: we are not concerned with label leaking here because we
      #       will not use these examples for adversarial training.
      #       See also: https://arxiv.org/pdf/1611.01236.pdf
      #
      grad = sess.run(model.loss_x, feed_dict={model.x_tf : x_mb, model.y_tf : y_mb})
      grad = grad[0:(b-a),...]
      x_adv[a:b,...] = x[a:b] + np.sign(grad) * eps

  return x_adv




if __name__ == "__main__":
  epsilon_values = [.02, .03, .05, .1, .15, .2, .25]
  output_file = 'cifar10_AE.h5'

  with tf.Graph().as_default(), tf.Session() as sess:
    model = Cifar10(sess)

    #--------------------------------------------------
    # Evaluate on clean data.
    #  (data from: https://www.cs.toronto.edu/~kriz/cifar.html)
    #--------------------------------------------------
    test_data_file = '/home/pekalmj1/Data/CIFAR10/cifar-10-batches-py/test_batch'
    x,y = load_cifar10_python(test_data_file, preprocess=True)
    print('[cifar10_wrapper]: x min/max:  %0.2f / %0.2f' % (np.min(x), np.max(x)))
    print('[cifar10_wrapper]: x mu/sigma: %0.2f / %0.2f' % (np.mean(x), np.std(x)))
    print('[cifar10_wrapper]: using epsilon: ',epsilon_values)

    y_hat, acc = _eval_model(sess, model, x, y)
    print('[cifar10_wrapper]: network accuracy on original/clean CIFAR10 (%d examples): %0.2f%%' % (y_hat.size,acc))

    with h5py.File(output_file, 'w') as h5:
      grp = h5.create_group('cifar10')
      grp['x'] = x
      grp['y'] = y
      grp['y_hat'] = y_hat  # estimates on clean data

      #----------------------------------------
      # Fast gradient sign attacks
      #----------------------------------------
      for eps in epsilon_values:
        x_adv = _fgsm_attack(sess, model, x, y, eps=eps)
        y_hat_adv, acc_adv = _eval_model(sess, model, x_adv, y)
        print('[cifar10_wrapper]: network accuracy on FGSM(eps=%0.2f) CIFAR10: %0.2f%%' % (eps, acc_adv))

        grp2 = grp.create_group('FGM-%0.2f' % eps)
        grp2['x'] = x_adv
        grp2['y_hat'] = y_hat_adv 
        grp2['epsilon'] = eps


  # The following is optional - just shows how to access file contents.
  with h5py.File(output_file, 'r') as h5:
    y_true = h5['cifar10']['y'].value

    for name in h5['cifar10']:
      if name.startswith('FGM'):
        y_hat = h5['cifar10'][name]['y_hat'].value
        acc_adv = 100. * np.sum(y_hat == y_true) / y_hat.size
        print('[cifar10_wrapper]: network accuracy on %s: %0.2f%%' % (name, acc_adv))
