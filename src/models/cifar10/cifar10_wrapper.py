"""
  !!! NOTE !!!
   1.  This script assumes you have already trained the model
       (using cifar10_train.py).
   2.  This script assumes you are running from the project root directory
       See the Makefile for an example of how this should be invoked.

"""

__author__ = "mjp"
__date__ = 'december, 2017'


import sys, os
import pdb
import h5py

import numpy as np

import tensorflow as tf

from cleverhans import utils_tf
from cleverhans.model import Model as CleverhansModel
from cleverhans.attacks import FastGradientMethod, MomentumIterativeMethod

from . import cifar10
from .cifar10_input import read_cifar10

from ae_utils import to_one_hot, run_in_batches



class Cifar10(CleverhansModel):
  """ Wrapper around CIFAR-10 model.

      Some of what we do here is to support Cleverhans; some is for our own purposes/attacks.
  """

  def __init__(self, sess, checkpoint_file_or_dir, num_models=1):
    # Note: the images are cropped prior to training.
    #       Hence, the non-standard CIFAR10 image sizes below.
    #
    self.batch_shape = [128, 24, 24, 3]
    self.num_classes = 10
    self.num_models = num_models

    self.x_tf = tf.placeholder(tf.float32, shape=self.batch_shape)
    self.y_tf = tf.placeholder(tf.int32, shape=[self.batch_shape[0],10])  

    # This serves two purposes:
    #   1) initialize the model. Note the use of reuse=False here (vs everywhere else)
    #   2) create a symbolic variable that we *might* use (CH will do its own thing)

    if self.num_models == 1:
      self.logits = cifar10.inference(self.x_tf, reuse=False)

      # Note: we do *not* use the original network's loss here, since that contains 
      #       weight decay terms that do not apply here.
      #       Instead, we build our own custom loss function for use with AE.
      self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_tf)
    else:
      logits_list, self.logits = cifar10.inference_n_models(self.x_tf, reuse=False, n=self.num_models)
      # ***  Note: this loss is (currently) agnostic of the orthogonality constraint!
      self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_tf)

    self.loss_x = tf.gradients(self.loss, self.x_tf)[0]

    self.__load_weights(checkpoint_file_or_dir)


  def __load_weights(self, checkpoint_file_or_dir):
    # load model weights
    # note: the directory must also contain the file with the name "checkpoint"
    #       in order for this to work
    variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    if not os.path.isdir(checkpoint_file_or_dir):
      saver.restore(sess, checkpoint_file_or_dir)
    else:
      ckpt = tf.train.get_checkpoint_state(checkpoint_file_or_dir)
      saver.restore(sess, ckpt.model_checkpoint_path)


  def get_logits(self, x):
    "Part of cleverhans Model API"
    if self.num_models == 1:
      return cifar10.inference(x, reuse=True)
    else:
      _, logits_agg = cifar10.inference_n_models(self.x_tf, reuse=True, n=self.num_models)
      return logits_agg


  def get_probs(self, x):
    "Part of cleverhans Model API"
    return tf.nn.softmax(self.get_logits(x))




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

    pred = sess.run(model.logits, feed_dict={model.x_tf : x_mb})
    y_hat[a:b] = np.argmax(pred[0:(b-a),:],axis=1)

  acc = 100. * np.sum(y_hat == y) / y.size

  return y_hat, acc



def _fgsm_attack(sess, model, x, y, eps):
  """
   Craft adversarial examples using Fast Gradient Sign Method (FGSM)
  """
  attack = FastGradientMethod(model, sess=sess)
  x_adv_tf = attack.generate(model.x_tf, eps=eps, clip_min=np.min(x), clip_max=np.max(x))

  y_oh = to_one_hot(y, 10)
  x_adv = run_in_batches(sess, model.x_tf, model.y_tf, x_adv_tf, x, y_oh, model.batch_shape[0])

  return x_adv



def _iterative_ell_infty_attack(sess, model, x, y, eps):
  """
   Use some iterative method with ell infty constraints here.
  """
  attack = MomentumIterativeMethod(model, sess=sess)
  x_adv_tf = attack.generate(model.x_tf, eps=eps, clip_min=np.min(x), clip_max=np.max(x))

  y_oh = to_one_hot(y, 10)
  x_adv = run_in_batches(sess, model.x_tf, model.y_tf, x_adv_tf, x, y_oh, model.batch_shape[0])

  return x_adv




if __name__ == "__main__":
  epsilon_values = [.02, .03, .05, .1, .15, .2, .25]
  output_file = 'cifar10_AE_CH.h5'
  cnn_weights = './Weights_n01'
  cnn_weights = './Weights_n02' # TEMP
  test_data_file = os.path.expanduser('~/Data/CIFAR10/cifar-10-batches-py/test_batch')

  with tf.Graph().as_default(), tf.Session() as sess:
    model = Cifar10(sess, cnn_weights, num_models=2)

    #--------------------------------------------------
    # Evaluate on clean data.
    #  (data from: https://www.cs.toronto.edu/~kriz/cifar.html)
    #--------------------------------------------------
    x,y = load_cifar10_python(test_data_file, preprocess=True)
    print('[cifar10_wrapper]: x min/max:         %0.2f / %0.2f' % (np.min(x), np.max(x)))
    print('[cifar10_wrapper]: x mu/sigma:        %0.2f / %0.2f' % (np.mean(x), np.std(x)))
    print('[cifar10_wrapper]: using epsilon:    ', epsilon_values)
    print('[cifar10_wrapper]: using tensorflow: ', tf.__version__)

    y_hat, acc = _eval_model(sess, model, x, y)
    print('[cifar10_wrapper]: network accuracy on original/clean CIFAR10 (%d examples): %0.2f%%' % (y_hat.size,acc))

    with h5py.File(output_file, 'w') as h5:
      grp = h5.create_group('cifar10')
      grp['x'] = x
      grp['y'] = y
      grp['y_hat'] = y_hat  # estimates on clean data

      #----------------------------------------
      # One-step \ell_\infty attack
      #----------------------------------------
      for eps in epsilon_values:
        x_adv = _fgsm_attack(sess, model, x, y, eps=eps)
        y_hat_adv, acc_adv = _eval_model(sess, model, x_adv, y)
        print('[cifar10_wrapper]: network accuracy on FGM(eps=%0.2f) CIFAR10: %0.2f%%' % (eps, acc_adv))

        grp2 = grp.create_group('FGM-%0.2f' % eps)
        grp2['x'] = x_adv
        grp2['y_hat'] = y_hat_adv 
        grp2['epsilon'] = eps

      #----------------------------------------
      # Iterative \ell_\infty attack
      #----------------------------------------
      for eps in epsilon_values:
        x_adv = _iterative_ell_infty_attack(sess, model, x, y, eps=eps)
        y_hat_adv, acc_adv = _eval_model(sess, model, x_adv, y)
        print('[cifar10_wrapper]: network accuracy on I-FGM(eps=%0.2f) CIFAR10: %0.2f%%' % (eps, acc_adv))

        grp2 = grp.create_group('I-FGM-%0.2f' % eps)
        grp2['x'] = x_adv
        grp2['y_hat'] = y_hat_adv 
        grp2['epsilon'] = eps


  print('[cifar10_wrapper]: results saved to file "%s"' % output_file)


  #--------------------------------------------------
  # The following is optional - provides an example of how one 
  # can access the HDF5 contents.
  #--------------------------------------------------
  with h5py.File(output_file, 'r') as h5:
    y_true = h5['cifar10']['y'].value

    for name in h5['cifar10']:
      if name.startswith('FGM') or name.startswith('I-FGM'):
        y_hat = h5['cifar10'][name]['y_hat'].value
        acc_adv = 100. * np.sum(y_hat == y_true) / y_hat.size
        print('[cifar10_wrapper]: network accuracy on %s (from saved file): %0.2f%%' % (name, acc_adv))
