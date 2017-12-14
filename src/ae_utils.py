""" Some utilities useful for our AE studies.

  Example usage:

    PYTHONPATH=./cleverhans python study_cifar10.py

"""

__author__ = "mjp"
__date__ = "dec, 2017"


import numpy as np
from scipy.stats import ortho_group
from numpy.linalg import norm
import random
import pandas as pd

import pdb, unittest

from gaas import gaas



def finite_mean(v):
  if not np.any(np.isfinite(v)):
    return np.nan
  else:
    return np.mean(v[np.isfinite(v)])


def to_one_hot(y_vec, n_classes):
  """   
    y_vec     : a numpy array of class labels (non-one-hot, obv.)
    n_classes : the total # of classes 
  """
  out = np.zeros((y_vec.size, n_classes), dtype=np.float32)
  if np.isscalar(y_vec):
    out[0,y_vec] = 1
  else:
    for ii in range(y_vec.size):
      out[ii,y_vec[ii]] = 1

  return out


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




def loss_function_stats(sess, model, x0, y0, d_max, 
                        n_samp_d=30, k_vals=[2,5,10]):
  """ Computes various statistics related to the loss function in the viscinity of (x0,y0).
  returns a dictionary with the different stats
  """

  # create a simple data structure to hold results
  #changed to a list of dictionaries of relevant pieces
  class Direction_Stats:
    def __init__(self):
      self.data = []

    def __str__(self):
      df = self.as_dataframe()
      s = ""

      for dname in ['gradient', 'neg-gradient']:
        tmp = df.loc[df['direction_type'] == dname]
        assert(len(tmp)==1)
        s += '  label first changes (%d->%d) along "%s" direction at distance %0.3f\n' % (tmp['y'], tmp['y_hat'], dname, tmp['boundary_distance'])

      for dname in ['gaussian', 'gaas']:
        tmp = df.loc[df['direction_type'] == dname]
        s += '  expected label change along "%s" direction: %0.3f\n' % (dname, finite_mean(tmp['boundary_distance']))

      return s

    def as_dataframe(self):
      return pd.DataFrame(self.data)

    def append(self, direction_type, y, y_hat, boundary_distance, **kargs):
      if not np.isscalar(y):
        y = np.argmax(y)
      # TODO: could check if boundary_distance is finite to avoid adding Inf to
      #       the table.  However, this is not necessarily an issue.
      entry = {'direction_type' : direction_type, 'y' : y, 'y_hat' : y_hat, 'boundary_distance' : boundary_distance}
      entry.update(kargs)
      self.data.append(entry)

  stats = Direction_Stats()

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
  stats.append('gradient', np.argmax(y0), y_new, (a+b)/2.)

  a,b,y_new = distance_to_decision_boundary(sess, model, x0, y_hat, -grad, d_max)
  stats.append('neg-gradient', np.argmax(y0), y_new, (a+b)/2.)

  #------------------------------
  # distance in random directions
  #------------------------------
  for jj in range(n_samp_d):
    a, b, y_new = distance_to_decision_boundary(sess, model, x0, y_hat, gaussian_vector(grad.shape), d_max)
    stats.append('gaussian', np.argmax(y0), y_new, (a+b)/2.)

  # MJP: this seemed to run very slowly for me...commented out temporarily
  #for did, orth_dir in enumerate(ortho_group.rvs(dim=np.prod(grad.shape), size=n_samp_d)):
  #  a, b, y_new = distance_to_decision_boundary(sess, model, x0, y_hat, orth_dir.reshape(grad.shape), d_max)
  #  stats.append('ortho_group', np.argmax(y0), y_new, (a+b)/2, direction_id=did)

  #------------------------------
  # distance in gaas directions
  # Note: instead of picking k=n_samp_d we could use some smaller k and draw convex samples from that...
  #------------------------------
  for k_idx, k in enumerate(k_vals):
    Q = gaas(grad, k)

    for did, col in enumerate(Q.T): #first check different directions from subspace
      a, b, y_new = distance_to_decision_boundary(sess, model, x0, y_hat, col.reshape(grad.shape), d_max)
      stats.append('gaas', np.argmax(y0), y_new, (a+b)/2., direction_id=did, k=k)

    for jj in range(min(n_samp_d, k)):
      # create a convex combo of the q_i
      coeff = np.random.uniform(size=k)    # coeff : a positive convex combo of q_i
      coeff = coeff / np.sum(coeff)
      q_dir = np.dot(Q,coeff)              # take linear combo
      q_dir = q_dir / norm(q_dir,2)        # back to unit norm
      q_dir = np.reshape(q_dir, x0.shape)

      a,b,y_new = distance_to_decision_boundary(sess, model, x0, y_hat, q_dir, d_max)
      stats.append('gaas_convex_combo', np.argmax(y0), y_new, (a+b)/2., k=k)

  if True:
    print("%s\n" % str(stats))

  return stats.as_dataframe()


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
