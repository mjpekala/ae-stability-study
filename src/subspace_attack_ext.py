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
sys.path.append('./')  # note: probably better to set your PYTHONPATH

from gaas import gaas
import nets




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
  """

  n_total_success = 0

  for batch_id, (filenames, x0) in enumerate(nets.load_images(input_dir, model.batch_shape)):
    n = len(filenames)
    assert(n==1) # for now, we assume batch size is 1

    #--------------------------------------------------
    # Use predictions on clean data as truth.  Note that x0 is single example we're feeding
    #--------------------------------------------------
    pred0 = sess.run(model.output, feed_dict={model.x_tf : x0})
    y0_scalar = np.argmax(pred0, axis=1)
    y0 = nets.smooth_one_hot_predictions(y0_scalar, model._num_classes)

    # also compute the loss and its gradient
    feed_dict = {model.x_tf : x0, model.y_tf : y0}
    loss0, g = sess.run([model.loss, model.nabla_x_loss], feed_dict=feed_dict)

    #--------------------------------------------------
    # Determine whether moving epsilon_max along the gradient produces an AE.
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
    savemat(fn, {'x_adv' : x_adv}) 

    if not success: #skip to next example if gradient doesn't work
      continue

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

      print('    gamma={0}, epsilon={1}, k={2}, ri_succ_rate={3}, ri_loss_rate={4}'.format(gamma, energy, k, 100.*n_successful / k, 100.*n_loss_change / k))
    n_total_success += any_success

  
  print('[GAAS]: AE success rate: %0.2f%%' % (100.* n_total_success / batch_id))




#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

if __name__ == "__main__":
  np.set_printoptions(precision=4, suppress=True)  # make output easier to read

  # otherwise, attack some data
  input_dir = sys.argv[1]
  output_dir = sys.argv[2]

  epsilon_max = 100 #made up; not sure how to tweak this yet
  print('[info]: maximum l2 distance is {}'.format(epsilon_max))

  with tf.Graph().as_default(), tf.Session() as sess:
    model = nets.InceptionV3(sess)
    gaas_attack(sess, model, epsilon_max, input_dir, output_dir) 

