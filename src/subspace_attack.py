""" SUBSPACE_ATTACK  Code for using GAAS [tra17] to analyze AE.


  References:
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

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception, resnet_v2
slim = tf.contrib.slim


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
# Functions to implement the Gradient aligned adversarial "subspace" (GAAS)
# technique developed by [tra17].
#-------------------------------------------------------------------------------

def _unitary_error(X):
  """ Frobenious norm of (X'X - I)
  """
  err = np.dot(X.T, X) - np.eye(X.shape[1], X.shape[1])
  return np.linalg.norm(err, 'fro')



def null(a, rtol=1e-5):
  """ Returns the nullspace of a.  This is just an application of SVD.
  """
  u,s,v = np.linalg.svd(a)
  rank = (s > rtol*s[0]).sum()
  return rank, v[rank:].T.copy()



def gaas(g, k):
  """ Computes the gradient aligned adversarial "subspace" (GAAS) of [tra17].

   g : the gradient from a single example (presumably an image)
   k : the number of orthogonal vectors

   Q : the matrix S'R ; note the first k rows of Q are the r_i

  """
  d = g.size             # dimension of gradient
  g_vec = g.flatten()    # gradient as a vector

  #------------------------------------------------------------
  # construct the unitary matrix R, which has the property:
  #
  #    Rg = ||g||_2 e_1
  #
  # where e_1 is the first standard basis vector.
  # Note that r_1 (the first row of R) has unit 
  # norm by construction; orthogonality of the
  # r_i then follows from the SVD.
  #------------------------------------------------------------
  r1 = g_vec / norm(g_vec, 2)
  r1 = r1[:,np.newaxis]

  rank, R_null = null(r1.T)
  assert(rank == 1)
  R = np.r_[r1.T, R_null.T]

  #------------------------------------------------------------
  # construct the unitary matrix S, which has the property that
  #
  #   Sz = e_1
  #
  # where z := \sum_{i=1}^k k^{-1/2} e_i
  #------------------------------------------------------------
  z_a = (1. / np.sqrt(k)) * np.ones((k,1)) # the first k elts of z
  rank, Z_null = null(z_a.T)
  assert(rank == 1)

  S_a = np.r_[z_a.T, Z_null.T]
  S_b = np.eye(d-k, d-k)
  S = blkdiag(S_a, S_b)

  Q = np.dot(S.T, R)

  #------------------------------------------------------------
  # Here we check the desired property that
  #
  #  Qg = ||g||_2 z
  #------------------------------------------------------------
  z = np.r_[z_a, np.zeros((d-k,1))]
  err = np.dot(Q,g_vec) - norm(g_vec,2) * z.flatten()
  assert(norm(err,2) < 1e-9)

  return Q



def sample_gaas(g, k, num_samps=1):
  Q = gaas(g,k)
  R = Q[:k,:].T    # the columns of R are the r_i defining the "subspace"

  out = []
  for ii in range(num_samps):
    coeff = np.random.uniform(0,1,size=(k,1))
    coeff /= np.sum(coeff)   # sum-to-one constraint

    putative_ae = np.dot(R,coeff)
    out.append(np.reshape(putative_ae, g.shape))

  return out



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




def gaas_attack(sess, model, epsilon_frac, input_dir, output_dir):
  """ Computes subset of attacks using GAAS method.
  """
  n_images, n_successful = 0, 0

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
    x_adv = np.clip(x0 + epsilon * g_normalized, -1, 1)
    pred_g = sess.run(model.output, feed_dict={model.x_tf : x_adv})
    loss_g = sess.run(model.loss, feed_dict={model.x_tf : x_adv, model.y_tf : y0})

    delta_x = x_adv - x0
    was_ae_successful = np.argmax(pred_g,axis=1) != y0_scalar

    print('[GAAS]: %d successful? %d, ||x||_2 = %0.3f, ||x - x_g||_2 = %0.3f, ||x - x_g||_\inf = %0.3f, delta_loss=%0.3f' % (batch_id, was_ae_successful, norm(x0.flatten(),2), norm(delta_x.flatten(), 2), norm(delta_x.flatten(), np.inf), loss_g - loss0))

    # save images for subsequent analysis
    fn = os.path.join(output_dir, 'grad_%d_' % was_ae_successful + filenames[0])
    imsave(fn,  x_adv[0,...])  

    n_images += 1
    if not was_ae_successful:
      continue
    n_successful += 1

    #--------------------------------------------------
    # Examine the subset of r_i
    #--------------------------------------------------
    for gamma_pct in [.1, .2, .4, .6, .9]:
      gamma = gamma_pct * (loss_g - loss0)

      # Try out the subset of r_i 
      alpha = gamma / (epsilon * norm(g.flatten(),2))
      k = min(g.size, np.floor(1.0/(alpha*alpha)))
      k = int(max(k, 1))

      Q = gaas(g.flatten(),k)

      n_successful = 0
      for ii in range(k):
        r_i = Q[ii,:] * epsilon
        x_adv = np.clip(x0 + r_i, -1, 1)
        pred_ri = sess.run(model.output, feed_dict={model.x_tf : x_adv})
        n_successful += np.argmax(pred_ri,axis=1) != y0_scalar

      print('    gamma_pct=%0.2f, k=%d, ri_succ_rate=%0.3f%%' % (gamma_pct, k, 100.*n_successful / k))

  print('[GAAS]: AE success rate: %0.2f%%' % (100.* n_successful / n_images))


#-------------------------------------------------------------------------------
# Unit tests
#-------------------------------------------------------------------------------

class TestEverything(unittest.TestCase):

  def test_unitary_error(self):
    X = np.eye(3,3);
    self.assertTrue(_unitary_error(X) < 1e-9)

    X = np.random.rand(10,10)
    self.assertTrue(_unitary_error(X) > 1e-3)


  def test_gaas(self):
    k = 2;  alpha = np.sqrt(1.0/k)

    for _ in range(10):
      G_fake = np.random.rand(10,10,3)
      Q = gaas(G_fake, k)
      self.assertTrue(_unitary_error(Q) < 1e-9)

      # next, make sure lemma 1 of [tra17] holds for each r_i
      R = Q[:k,:].T
      self.assertTrue(_unitary_error(R) < 1e-9)
      for ii in range(k):
        gtri = np.dot(G_fake.flatten(), R[:,ii])  # <g, r_i>
        delta = (gtri + 1e-6) - alpha * norm(G_fake.flatten(),2)
        self.assertTrue(delta > 0)


  def test_sample_gaas(self):
    G_fake = np.random.rand(10,10,3)
    n_samps = 5
    ae = sample_gaas(G_fake, 3, n_samps)

    self.assertTrue(len(ae) == n_samps)
    self.assertTrue(np.all(ae[0].shape == G_fake.shape))



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

if __name__ == "__main__":
  if len(sys.argv) < 2:
    # if called without arguments, run some unit tests
    unittest.main()
  else:
    # otherwise, attack some data
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    epsilon_l2 = 0.05
    epsilon_linf = 0.4

    print('[info]: epsilon_l2=%0.3f, epsilon_linf=%0.3f' % (epsilon_l2, epsilon_linf))

    with tf.Graph().as_default(), tf.Session() as sess:
      model = InceptionV3(sess)
      gaas_attack(sess, model, epsilon_l2, input_dir, output_dir)    # TODO: this is under construction
      #fgsm_attack(sess, model, epsilon_linf, input_dir, output_dir)

