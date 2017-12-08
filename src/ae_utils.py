""" Some utilities useful for our AE studies.

  Example usage:

    PYTHONPATH=./cleverhans python study_cifar10.py

"""

__author__ = "mjp"
__date__ = "dec, 2017"


import numpy as np
from numpy.linalg import norm

import pdb, unittest

from gaas import gaas



#def smooth_one_hot_predictions(y, num_classes):
#  """Given a vector (*not* a full matrix) of class labels y, 
#     generates a 'smoothed' one-hot class *matrix*.
#  """
#  out = (1./num_classes) * np.ones((y.size, num_classes), dtype=np.float32)
#  for ii in range(y.size):
#    out[ii,y[ii]] = 0.9
#  return out


def smoothed_one_hot(y):
  """ Given a one-hot encoding of the class for a single example, returns a smoothed variant.

  Note: I'm not sure if the code below is the "standard" way of doing this, but it
        preserves a normalization property for y
  """
  n_classes = y.size
  mag_true = 0.9
  mag_rest = (1. - mag_true) / (n_classes-1)

  y_smooth = mag_rest * np.ones(y.shape)
  y_smooth[np.argmax(y)] = mag_true

  return y_smooth



def gaussian_vector(vec_shape):
  """ Generates a (normalized) vector with iid gaussian entries

    vec_shape : a tuple indicating the shape of the vector/tensor to be created.
  """
  rv = np.random.randn(*vec_shape) 
  return rv / norm(rv.flatten(),2)



def get_info(sess, model, x, y=None):
  """ Queries CNN for basic information about a single example x.
  """
  x_batch = np.zeros(model.batch_shape)
  x_batch[0,...] = x

  if y is not None:
    assert(y.size == model.num_classes)  # y should be one-hot

    # if we have a class label, we can compute a loss
    y_batch = np.zeros((model.batch_shape[0], model.num_classes))
    y_batch[0,...] = y

    pred, loss, grad = sess.run([model.output, model.loss, model.loss_x], 
                                feed_dict={model.x_tf : x_batch, model.y_tf : y_batch})
    return pred[0,...], loss[0], grad[0]
  else:
    # without a class label we can only predict
    pred = sess.run(model.output, feed_dict={model.x_tf : x_batch})
    return pred[0,...]



def distance_to_decision_boundary(sess, model, x, y, direction, d_max, tol=1e-1):
  """ Computes (approximately) the distance one needs to move along
      some direction in order for the CNN to change its decision.  

      x         : a single example/image with shape (rows x cols x channels)
      y         : class label associated with x (scalar)
      direction : the search direction; same shape as x
      d_max     : the maximum distance to move along direction (scalar)
      tol       : the maximum size of the interval around the change
  """

  assert(np.isscalar(y))

  # normalize search direction
  direction = direction / norm(direction.ravel(),2)

  n = model.batch_shape[0]
  if n < 3:
    raise RuntimeError('sorry, I assume a non-trivial batch size')

  x_batch = np.zeros(model.batch_shape, dtype=np.float32)
  a = 0
  b = d_max

  while (b-a) > tol:
    # search over interval [a,b] for changes in label
    epsilon_vals = np.linspace(a, b, n)
    for ii in range(n):
      x_batch[ii,...] = x + epsilon_vals[ii] * direction

    preds = sess.run(model.output, feed_dict={model.x_tf : x_batch})
    y_hat = np.argmax(preds, axis=1)
    if np.all(y_hat == y):
      return d_max, np.Inf, None  # label never changed in given interval

    first_change = np.min(np.where(y_hat != y)[0])
    assert(first_change > 0)

    # refine interval
    a = epsilon_vals[first_change-1]
    b = epsilon_vals[first_change]

  return a, b, y_hat[first_change]




def distance_to_decision_boundary_stats(sess, model, x0, y0, d_max, 
                                        n_samp_d=30, k_vals=[2,5,10]):
  """ Computes distance_to_decision_boundary() for a variety of directions and
      also generates associated statistics.
  """

  # create a simple data structure to hold results
  class Stats:
    def __init__(self):
      self.d_grad = np.nan
      self.d_neg_grad = np.nan
      self.d_gauss = np.nan * np.ones((n_samp_d,))
      self.d_gaas = np.nan * np.ones((len(k_vals), n_samp_d))
  out = Stats()

  #------------------------------
  # get some basic info about x
  #------------------------------
  pred, loss, grad = get_info(sess, model, x0, y0)
  y_hat = np.argmax(pred)
  assert(y_hat == np.argmax(y0))

  #------------------------------
  # distance in gradient direction
  #------------------------------
  a,b,y_new = distance_to_decision_boundary(sess, model, x0, y_hat, grad, d_max)
  if np.isfinite(b):
    print('   label first changes (%d->%d) along gradient direction in [%0.3f, %0.3f]' % (np.argmax(y0),y_new,a,b))
    out.d_grad = a + (b-a)/2.

  a,b,y_new = distance_to_decision_boundary(sess, model, x0, y_hat, -grad, d_max)
  if np.isfinite(b):
    print('   label first changes (%d->%d) along neg. gradient direction in [%0.3f, %0.3f]' % (np.argmax(y0),y_new,a,b))
    out.d_neg_grad = a + (b-a)/2.

  #------------------------------
  # distance in random directions
  #------------------------------
  for jj in range(n_samp_d):
    a, b, y_new = distance_to_decision_boundary(sess, model, x0, y_hat, gaussian_vector(grad.shape), d_max)
    if np.isfinite(b):
      out.d_gauss[jj] = a + (b-a)/2.
  print('   expected first label change along random direction    %0.3f' % (np.nanmean(out.d_gauss)))

  #------------------------------
  # distance in gaas directions
  # Note: instead of picking k=n_samp_d we could use some smaller k and draw convex samples from that...
  #------------------------------
  for k_idx, k in enumerate(k_vals):
    Q = gaas(grad, k)

    for jj in range(n_samp_d):
      # create a convex combo of the q_i
      coeff = np.random.uniform(size=k)
      coeff = coeff / np.sum(coeff)
      q_dir = np.dot(Q,coeff)
      q_dir = np.reshape(q_dir, x0.shape)

      a,b,y_new = distance_to_decision_boundary(sess, model, x0, y_hat, q_dir, d_max)
      if np.isfinite(b):
        out.d_gaas[k_idx,jj] = a + (b-a)/2.

    print('   expected first label change along k=%02d GAAS direction %0.3f' % (k, np.nanmean(out.d_gaas[k_idx,:])))

  return out


#-------------------------------------------------------------------------------

class Tests(unittest.TestCase):
  def test_gaussian_vector(self):
    # we anticipate that vectors generated randomly in this fashion 
    # should be nearly orthogonal with high probability.
    #
    # TODO: look up any analytic result describing the rate 
    # and use this to set tol and dim...
    tol = 1e-2
    dim = (10000,)
    n_trials = 100
    result = np.zeros((n_trials,))

    for ii in range(n_trials):
      a = gaussian_vector(dim)
      b = gaussian_vector(dim)
      result[ii] = np.dot(a,b)

    print(np.sum(result < tol)) # TEMP
    self.assertTrue(np.sum(result < tol) > (.7*n_trials))


if __name__ == "__main__":
  unittest.main()
