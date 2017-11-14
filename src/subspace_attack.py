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
from numpy.linalg import norm, matrix_rank
from scipy.linalg import block_diag as blkdiag
from scipy.misc import imread, imsave
from scipy.io import savemat

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception, resnet_v2
slim = tf.contrib.slim

import nets
from gaas import gaas


SEED = 1099


def tf_run(sess, outputs, feed_dict, seed=1099):
  tf.set_random_seed(SEED)
  return sess.run(outputs, feed_dict=feed_dict)



def fgsm_attack(sess, model, epsilon, input_dir, output_dir):
  """ Simple implementation of a fast gradient sign attack.
      This serves primarily as a sanity check of our tensorflow code.
  """

  n_changed = 0
  n_total = 0

  for batch_id, (filenames, x0) in enumerate(nets.load_images(input_dir, model.batch_shape)):
    n = len(filenames)

    # for now, take predictions on clean data as truth.
    pred0 = sess.run(model.output, feed_dict={model.x_tf : x0})
    y0 = nets.smooth_one_hot_predictions(np.argmax(pred0, axis=1), model._num_classes)

    # compute the gradient
    feed_dict = {model.x_tf : x0, model.y_tf : y0}
    grad = sess.run(model.loss_x, feed_dict=feed_dict)

    # construct AE
    x_adv = x0 + epsilon * np.sign(grad)
    x_adv = np.clip(x_adv, -1, 1)  # make sure AE respects box contraints

    # predicted label of AE
    pred1 = sess.run(model.output, feed_dict={model.x_tf : x_adv})
    y1 = nets.smooth_one_hot_predictions(np.argmax(pred1, axis=1), model._num_classes)

    is_same = np.argmax(y1[:n,:],axis=1) == np.argmax(y0[:n,:],axis=1)
    print('[FGSM]: batch %02d: %d of %d predictions changed' % (batch_id, np.sum(~is_same), n))

    n_changed += np.sum(~is_same)
    n_total += n

  print('[FGSM]: overall AE success rate: %0.2f' % (100.*n_changed/n_total))


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


def linearity_test(sess, model, input_dir, output_dir, epsilon=1):
  """  Here we check to see if the GAAS subspace construction seems to
       be working with representative loss functions.
  """

  overall_result = []

  for batch_id, (filenames, x0) in enumerate(nets.load_images(input_dir, model.batch_shape)):
    n = len(filenames)
    assert(n==1) # for now, we assume batch size is 1
    print('EXAMPLE %3d' % batch_id)

    #--------------------------------------------------
    # Use predictions on original example as ground truth.
    #--------------------------------------------------
    pred0 = tf_run(sess, model.output, feed_dict={model.x_tf : x0})
    y0_scalar = np.argmax(pred0, axis=1)
    y0 = nets.smooth_one_hot_predictions(y0_scalar, model._num_classes)

    #--------------------------------------------------
    # compute the loss and its gradient
    #--------------------------------------------------
    feed_dict = {model.x_tf : x0, model.y_tf : y0}
    loss0, g = tf_run(sess, [model.loss, model.loss_x], feed_dict=feed_dict)
    l2_norm_g = norm(g.flatten(),2)

    #--------------------------------------------------
    # see how much the loss changes if we move along g by epsilon
    #--------------------------------------------------
    x_step = x0 + epsilon * (g / l2_norm_g)
    loss_end, pred_end = tf_run(sess, [model.loss, model.output], feed_dict={model.x_tf : x_step, model.y_tf : y0})

    was_ae_successful = np.argmax(pred_end,axis=1) != y0_scalar
    print('   loss on clean example: %2.3f' % loss0)
    print('   was \ell_2 gradient-step AE successful? {}'.format(was_ae_successful))

    # if moving by epsilon fails to increase the loss, this is unexpected
    if loss_end <= loss0:
      print('[info]: moving along gradient failed to increase loss; skipping...')
      continue

    #--------------------------------------------------
    # Pick some admissible gamma and compute the corresponding value of k.
    #--------------------------------------------------

    # Note: lemma 1 requires alpha live in [0,1]!
    # This limits how large one can make gamma.
    # Note the formula in [tra17] is actually for alpha^{-1}
    gamma = loss_end - loss0
    while gamma / (epsilon * l2_norm_g) > 1.0:
        gamma /= 2.0

    alpha_inv = epsilon * (l2_norm_g / gamma)
    k = int(np.floor(alpha_inv ** 2))
    assert(k > 0)

    #--------------------------------------------------
    # Check behavior of GAAS and of the loss function
    #
    # Note that, if the assumption of local linearity is incorrect,
    # the r_i may fail to increase the loss as anticipated.
    #--------------------------------------------------
    inner_product_test = np.zeros((k,))
    delta_loss = np.zeros((k,))
    delta_loss_test = np.zeros((k,))
    y_hat_test = np.zeros((k,))

    #--------------------------------------------------
    # The notation from the paper is a bit confusing.  The r_i in lemma1 have 
    # unit \ell_2 norm while the r_i in the GAAS construction have \ell_2 norm <= epsilon.
    # To help keep things clear, I will call the vectors from the lemma q_i and the 
    # appropriately rescaled vectors for GAAS will be called r_i.
    #--------------------------------------------------
    Q = gaas(g, k)

    for ii in range(k):
      q_i = np.reshape(Q[:,ii], g.shape)  # the r_i in lemma 1 of [tra17]
      r_i = q_i * epsilon                 # the r_i in GAAS perturbation of [tra17]

      #--------------------------------------------------
      # make sure the lemma is satisfied
      # this should always be true if our GAAS implementation is correct
      #--------------------------------------------------
      inner_product_test[ii] = np.dot(g.flatten(), q_i.flatten()) > (l2_norm_g / alpha_inv)

      #--------------------------------------------------
      # see whether the loss behaves as expected; ie. moving along the r_i 
      # increases the loss by at least gamma.  This assumes the second-order term
      # is sufficiently small that it can be ignored entirely (which may be untrue
      # if the curvature is sufficiently large?).
      #--------------------------------------------------
      #
      loss_i, pred_i = tf_run(sess, [model.loss, model.output], feed_dict={model.x_tf : x0 + r_i, model.y_tf : y0})
      delta_loss[ii] = (loss_i - (gamma + loss0))
      slop = 1e-4 # this should really be tied to the error term in the Taylor series expansion...
      delta_loss_test[ii] = delta_loss[ii] > -slop

      #--------------------------------------------------
      # Check whether r_i was a successful AE.
      # In some sense, this is more a test of the hypothesis that
      # changes in the loss are sufficient to characterize 
      # network predictions (vs. integrity of the code).
      #--------------------------------------------------
      y_hat_test[ii] = np.argmax(pred_i,axis=1)


    #--------------------------------------------------
    # summarize performance on this example
    #--------------------------------------------------
    loss_predicted = loss0 + l2_norm_g * epsilon
    print('   delta_loss: %2.3f, ||g||=%2.3f,  ratio=%2.3f' % (loss_end-loss0, l2_norm_g, l2_norm_g / (loss_end-loss0)))
    print('   loss along gradient direction, predicted/actual: %2.3f / %2.3f  %s' % (loss_predicted, loss_end, '*' if loss_predicted > loss_end else ''))
    print('   k=%d,  #_ip=%d,  #d_loss=%d, #AE=%d' % (k, np.sum(inner_product_test), np.sum(delta_loss_test), np.sum(y_hat_test != y0_scalar)))
    print('')

    if k > 0 and np.sum(delta_loss_test) < 1:
      print(delta_loss)
      overall_result.append(False)
    else:
      overall_result.append(True)

  # all done!
  print('%d (of %d) admissible examples behaved as expected' % (np.sum(overall_result), len(overall_result)))
    



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

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

  for batch_id, (filenames, x0) in enumerate(nets.load_images(input_dir, model.batch_shape)):
    n = len(filenames)
    assert(n==1) # for now, we assume batch size is 1

    # translate relative energy constraint into an ell2 constraint
    epsilon = epsilon_frac * norm(x0.flatten(),2)

    #--------------------------------------------------
    # Use predictions on clean data as truth.
    #--------------------------------------------------
    pred0 = sess.run(model.output, feed_dict={model.x_tf : x0})
    y0_scalar = np.argmax(pred0, axis=1)
    y0 = nets.smooth_one_hot_predictions(y0_scalar, model._num_classes)

    # also compute the loss and its gradient
    feed_dict = {model.x_tf : x0, model.y_tf : y0}
    loss0, g = sess.run([model.loss, model.loss_x], feed_dict=feed_dict)
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

  # otherwise, attack some data
  input_dir = sys.argv[1]
  output_dir = sys.argv[2]
  epsilon_l2 = 0.1   
  epsilon_linf = 0.4

  print('[info]: epsilon_l2=%0.3f, epsilon_linf=%0.3f' % (epsilon_l2, epsilon_linf))

  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  with tf.Graph().as_default(), tf.Session() as sess:
    model = nets.InceptionV3(sess)
    linearity_test(sess, model, input_dir, output_dir)
    #gaas_attack(sess, model, epsilon_l2, input_dir, output_dir) 
    #fgsm_attack(sess, model, epsilon_linf, input_dir, output_dir)

