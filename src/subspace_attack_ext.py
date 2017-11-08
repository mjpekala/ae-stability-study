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
sys.path.append('./')
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
def _sample_adversarial_direction(model, x0, y0, g, epsilon_max):
  """ Evaluates model loss and predictions at multiple points along 
      a (presumed) adversarial direction g.
  """
  g_normalized = g / norm(g.flatten(),2)  # unit \ell_2 norm

  v = [.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]

  starting_loss = sess.run(model.loss, feed_dict={model.x_tf : x0, model.y_tf : y0})

  for idx, pct in enumerate(v):
    x_adv = np.clip(x0 + pct * epsilon_max * g_normalized, -1, 1)

    loss = sess.run(model.loss, feed_dict={model.x_tf : x_adv, model.y_tf : y0})
    pred = sess.run(model.output, feed_dict={model.x_tf : x_adv})

    success = (np.argmax(y0, axis=1) != np.argmax(pred, axis=1)) #test if class changed
    loss_chg = loss - starting_loss
    energy = pct * epsilon_max
    if success:
      return (success, x_adv, loss_chg, energy)

  return (success, x_adv, loss_chg, energy) # still need to return something if fail


def _test_perturbation(model, x0, y0, delta):
  # same as above, but using fixed change of delta rather than different epsilon
  starting_loss = sess.run(model.loss, feed_dict={model.x_tf: x0, model.y_tf: y0})
  x_adv = np.clip(x0 + delta, -1, 1)
  loss = sess.run(model.loss, feed_dict={model.x_tf: x_adv, model.y_tf: y0})
  pred = sess.run(model.output, feed_dict={model.x_tf: x_adv})
  success = (np.argmax(y0, axis=1) != np.argmax(pred, axis=1))  # test if class changed
  loss_chg = loss - starting_loss
  return (success, x_adv, loss_chg)



def gaas_attack(sess, model, epsilon_max, input_dir, output_dir):
  """ Computes subset of attacks using GAAS method.
    epsilon_set is list of epsilons to try (radius in L2)
  """


  print('[WARNING] this code is still under development!!!')
  n_images = 0
  n_total_success = 0
  for batch_id, (filenames, x0) in enumerate(_load_images(input_dir, model.batch_shape)):
    n = len(filenames)
    assert(n==1) # for now, we assume batch size is 1
    #--------------------------------------------------
    # Use predictions on clean data as truth.  Note that x0 is single example we're feeding
    #--------------------------------------------------
    pred0 = sess.run(model.output, feed_dict={model.x_tf : x0})
    y0_scalar = np.argmax(pred0, axis=1)
    y0 = smooth_one_hot_predictions(y0_scalar, model._num_classes)

    # also compute the loss and its gradient
    feed_dict = {model.x_tf : x0, model.y_tf : y0}
    loss0, g = sess.run([model.loss, model.nabla_x_loss], feed_dict=feed_dict)
    #g_normalized = g / norm(g.flatten(),2)

    #--------------------------------------------------
    # Determine whether moving epsilon along the gradient produces an AE.
    # If not, then the subset of r_i is unlikely to be useful.
    #--------------------------------------------------
    success, x_adv, loss_chg, energy = _sample_adversarial_direction(model, x0, y0, g, epsilon_max)
    delta_x = x_adv - x0

    print('[GAAS]: image %d successful? %d, y0=%d, epsilon = %0.3f, ||x - x_g||_2 = %0.3f, ||x - x_g||_\inf = %0.3f, delta_loss=%0.3f' % (batch_id, success, y0_scalar, energy, norm(delta_x.flatten(), 2), norm(delta_x.flatten(), np.inf), loss_chg))
    sys.stdout.flush()

    # save results for subsequent analysis
    fn = os.path.join(output_dir, 'grad_%d_' % success + filenames[0])
    imsave(fn,  x_adv[0,...])  
    fn = os.path.join(output_dir, 'gradient_samps_' + filenames[0].replace('.png', '.mat'))
    savemat(fn, {'out' : out}) #TODO: decide how to save relevant info

    if not success: #skip to next example if gradient doesn't work
      continue
    n_images += 1

    #--------------------------------------------------
    # Check whether the r_i are also adversarial directions
    # (if one moves epsilon along that direction)
    #--------------------------------------------------
    n_successful, n_loss_change = 0, 0
    any_success = False
    for gamma_pct in [.01, .1, .2, .5, 1]:
      # We have to choose how much the loss should change (i.e. gamma)
      # This is now based on the minimal movement in the gradient direction that switched classes
      gamma = gamma_pct * loss_chg

      # Compute k ("subspace" dimension)
      alpha = gamma / (energy * norm(g.flatten(),2)) #use energy required to perturn adversarial example.  Keeping this close to min should keep k smaller (easy computations)


      k = min(g.size, np.floor(1.0/(alpha*alpha)))
      k = int(max(k, 1))

      print('k=%d' % k) # TEMP
      k = min(k, 1000)  # put a limit on k for practical (computational) reasons

      R = gaas(g.flatten(),k)

      # count how many of the r_i are also successful AE directions
      n_successful = 0
      for ii in range(k):
        r_i = R[:,ii] * energy
        r_i = np.reshape(r_i, x0.shape)
        success, x_adv, loss_chg = _test_perturbation(model, x0, y0, r_i)
        n_successful += success
        if success:
          any_success = True
        n_loss_change += (loss_chg > gamma) #if the construction is correct, loss should have increased by at least gamma

      print('    gamma={0.2f}, epsilon={0.2f}, k={d}, ri_succ_rate={0.3f%%}, ri_loss_rate={0.3f%%}'.format(gamma, energy, k, 100.*n_successful / k), 100.*n_loss_change / k)
    n_total_success += any_success
  print('[GAAS]: AE success rate: %0.2f%%' % (100.* n_total_success / n_images))




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

    epsilon_max = 100 #made up; not sure how to tweak this yet
    print('[info]: maximum l2 distance is {}'.format(epsilon_max))

    with tf.Graph().as_default(), tf.Session() as sess:
      model = InceptionV3(sess)
      gaas_attack(sess, model, epsilon_max, input_dir, output_dir)    # TODO: this is under construction

