""" This code uses the Gradient Aligned Adversarial Subspace (GAAS) 
    of [tra17] to analyze AE.


  REFERENCES:
  [tra17] Tramer et al. "The Space of Transferable Adversarial Examples," 2017.

"""

__author__ = "mjp"
__date__ = "november, 2017"


import sys, os
import unittest
import pdb

import numpy as np
from numpy.linalg import norm
from scipy.linalg import block_diag as blkdiag
from scipy.misc import imread, imsave
from scipy.io import savemat

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception, resnet_v2
slim = tf.contrib.slim

from gaas import gaas


#-------------------------------------------------------------------------------
# Functions for dealing with data and tensorflow models
#-------------------------------------------------------------------------------


def _input_filenames(input_dir):
  all_files = tf.gfile.Glob(os.path.join(input_dir, '*.png'))
  all_files.sort()
  return all_files


def _load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      length of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]

  for filepath in _input_filenames(input_dir):
    with tf.gfile.Open(filepath, mode='rb') as f:
      image = imread(f, mode='RGB').astype(np.float) / 255.0

    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))

    idx += 1
    if idx == batch_size:
      yield filenames, images
      filenames = []
      images = np.zeros(batch_shape)
      idx = 0

  # This is a partial batch left over at end.
  # Note that images will still have the proper size.
  if idx > 0:
    yield filenames, images


def smooth_one_hot_predictions(p, num_classes):
  """Given a vector (*not* a full matrix) of predicted class labels p, 
     generates a 'smoothed' one-hot prediction *matrix*.
  """
  out = (1./num_classes) * np.ones((p.size, num_classes), dtype=np.float32)
  for ii in range(p.size):
    out[ii,p[ii]] = 0.9
  return out




class InceptionV3:
  """ Bare-bones interface to the InceptionV3 model.
  """

  def __init__(self, sess):
    self.batch_shape = [1, 299, 299, 3]  # for now, we attack one image at a time
    self._num_classes = 1001
    self._scope = 'InceptionV3'
    self._weights_file = './Weights/inception_v3.ckpt'

    #
    # network inputs
    #
    self.x_tf = tf.placeholder(tf.float32, shape=self.batch_shape)
    self.y_tf = tf.placeholder(tf.float32, shape=[self.batch_shape[0], self._num_classes])

    #
    # network outputs
    #
    with slim.arg_scope(inception.inception_v3_arg_scope()): 
      logits, end_points = inception.inception_v3(self.x_tf, num_classes=self._num_classes, is_training=False, scope=self._scope)

      # output 1: the raw network predictions (a function of x only)
      self.output = end_points['Predictions']

      # output 2: the gradient of a loss function (a function of both x and y)
      cross_entropy_loss = tf.losses.softmax_cross_entropy(self.y_tf, 
                                                           logits, 
                                                           label_smoothing=0.1, 
                                                           weights=1.0)

      cross_entropy_loss += tf.losses.softmax_cross_entropy(self.y_tf, 
                                                            end_points['AuxLogits'], 
                                                            label_smoothing=0.1, 
                                                            weights=0.4) 

      self.loss = cross_entropy_loss
      self.nabla_x_loss = tf.gradients(self.loss, self.x_tf)[0] 

    #
    # load weights
    #
    saver = tf.train.Saver(slim.get_model_variables(scope=self._scope))
    saver.restore(sess, self._weights_file)



#-------------------------------------------------------------------------------


def fgsm_attack(sess, model, epsilon, input_dir, output_dir):
  """ Simple implementation of a fast gradient sign attack.
      This is just for us to test our code and is not of particular interest.
  """

  n_changed = 0
  n_total = 0

  for batch_id, (filenames, x0) in enumerate(_load_images(input_dir, model.batch_shape)):
    n = len(filenames)

    # for now, take predictions on clean data as truth.
    pred0 = sess.run(model.output, feed_dict={model.x_tf : x0})
    y0 = smooth_one_hot_predictions(np.argmax(pred0, axis=1), model._num_classes)

    # compute the gradient
    feed_dict = {model.x_tf : x0, model.y_tf : y0}
    grad = sess.run(model.nabla_x_loss, feed_dict=feed_dict)

    # construct AE
    x_adv = x0 + epsilon * np.sign(grad)
    x_adv = np.clip(x_adv, -1, 1)  # make sure AE respects box contraints

    # predicted label of AE
    pred1 = sess.run(model.output, feed_dict={model.x_tf : x_adv})
    y1 = smooth_one_hot_predictions(np.argmax(pred1, axis=1), model._num_classes)

    is_same = np.argmax(y1[:n,:],axis=1) == np.argmax(y0[:n,:],axis=1)
    print('[FGSM]: batch %02d: %d of %d predictions changed' % (batch_id, np.sum(~is_same), n))

    n_changed += np.sum(~is_same)
    n_total += n

  print('[FGSM]: overall AE success rate: %0.2f' % (100.*n_changed/n_total))



def _sample_adversarial_direction(model, x0, y0, g, epsilon_max):
  """ Evaluates model loss and predictions at multiple points along 
      a (presumed) adversarial direction.
  """
  g_normalized = g / norm(g.flatten(),2)  # unit \ell_2 norm

  v = [.01, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]

  out = np.zeros((len(v),3))

  for idx, pct in enumerate(v):
    x_adv = np.clip(x0 + pct * epsilon_max * g_normalized, -1, 1)

    loss = sess.run(model.loss, feed_dict={model.x_tf : x_adv, model.y_tf : y0})
    pred = sess.run(model.output, feed_dict={model.x_tf : x_adv})

    out[idx,0] = pct * epsilon_max
    out[idx,1] = loss
    out[idx,2] = np.argmax(pred,axis=1)

  return out, x_adv



def gaas_attack(sess, model, epsilon_frac, input_dir, output_dir):
  """ Computes subset of attacks using GAAS method.
  """
  n_images, n_successful = 0, 0

  print('[WARNING] this code is still under development!!!')

  for batch_id, (filenames, x0) in enumerate(_load_images(input_dir, model.batch_shape)):
    n = len(filenames)
    assert(n==1) # for now, we assume batch size is 1

    # translate relative energy constraint into an ell2 constraint
    epsilon = epsilon_frac * norm(x0.flatten(),2)

    #--------------------------------------------------
    # Use predictions on clean data as truth.
    #--------------------------------------------------
    pred0 = sess.run(model.output, feed_dict={model.x_tf : x0})
    y0_scalar = np.argmax(pred0, axis=1)
    y0 = smooth_one_hot_predictions(y0_scalar, model._num_classes)

    # also compute the loss and its gradient
    feed_dict = {model.x_tf : x0, model.y_tf : y0}
    loss0, g = sess.run([model.loss, model.nabla_x_loss], feed_dict=feed_dict)
    g_normalized = g / norm(g.flatten(),2)

    #--------------------------------------------------
    # Determine whether moving epsilon along the gradient produces an AE.
    # If not, then the subset of r_i is unlikely to be useful.
    #--------------------------------------------------
    out, x_adv = _sample_adversarial_direction(model, x0, y0, g_normalized, epsilon)
    loss_g = out[-1,1]
    was_ae_successful = np.any(out[:,2] != y0_scalar)

    delta_x = x_adv - x0

    print('[GAAS]: image %d successful? %d, y0=%d, ||x||_2 = %0.3f, ||x - x_g||_2 = %0.3f, ||x - x_g||_\inf = %0.3f, delta_loss=%0.3f' % (batch_id, was_ae_successful, y0_scalar, norm(x0.flatten(),2), norm(delta_x.flatten(), 2), norm(delta_x.flatten(), np.inf), loss_g - loss0))
    sys.stdout.flush()

    # save results for subsequent analysis
    fn = os.path.join(output_dir, 'grad_%d_' % was_ae_successful + filenames[0])
    imsave(fn,  x_adv[0,...])  
    fn = os.path.join(output_dir, 'gradient_samps_' + filenames[0].replace('.png', '.mat'))
    savemat(fn, {'out' : out})

    n_images += 1
    if not was_ae_successful:
      continue
    n_successful += 1

    print(out) 

    #--------------------------------------------------
    # Check whether the r_i are also adversarial directions
    # (if one moves epsilon along that direction)
    #--------------------------------------------------
    for gamma_pct in [.8, .9, .99]:
      # We have to choose how much the loss should change (i.e. gamma)
      # Currently, this is based on the maximum change determined above.
      # It may be there is a better approach...
      gamma = gamma_pct * (loss_g - loss0)

      # Compute k ("subspace" dimension)
      alpha = gamma / (epsilon * norm(g.flatten(),2))
      k = min(g.size, np.floor(1.0/(alpha*alpha)))
      k = int(max(k, 1))

      print('k=%d' % k) # TEMP
      k = min(k, 1000)  # put a limit on k for practical (computational) reasons

      R = gaas(g.flatten(),k)

      # count how many of the r_i are also successful AE directions
      n_successful = 0
      for ii in range(k):
        r_i = R[:,ii] * epsilon
        r_i = np.reshape(r_i, x0.shape)
        x_adv = np.clip(x0 + r_i, -1, 1)
        pred_ri = sess.run(model.output, feed_dict={model.x_tf : x_adv})
        n_successful += np.argmax(pred_ri,axis=1) != y0_scalar

      print('    gamma=%0.2f, epsilon=%0.2f, k=%d, ri_succ_rate=%0.3f%%' % (gamma, epsilon, k, 100.*n_successful / k))

  print('[GAAS]: AE success rate: %0.2f%%' % (100.* n_successful / n_images))




#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

if __name__ == "__main__":
  np.set_printoptions(precision=4, suppress=True)  # make output easier to read

  if len(sys.argv) < 2:
    # if called without arguments, run some unit tests
    unittest.main()
  else:
    # otherwise, attack some data
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    epsilon_l2 = 0.1   
    epsilon_linf = 0.4

    print('[info]: epsilon_l2=%0.3f, epsilon_linf=%0.3f' % (epsilon_l2, epsilon_linf))

    with tf.Graph().as_default(), tf.Session() as sess:
      model = InceptionV3(sess)
      gaas_attack(sess, model, epsilon_l2, input_dir, output_dir)    # TODO: this is under construction
      #fgsm_attack(sess, model, epsilon_linf, input_dir, output_dir)

