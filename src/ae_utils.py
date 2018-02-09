""" This module contains functions to support our AE analyses.
"""

__author__ = "mjp,ef"
__date__ = "dec, 2017"


import math, random
import numpy as np
from numpy.linalg import norm
from scipy.stats import ortho_group
import pandas as pd

import pdb, unittest

from gaas import gaas



#-------------------------------------------------------------------------------
# Generic helper/utility functions
#-------------------------------------------------------------------------------

def splitpath(full_path):
  """
  Splits a path into all possible pieces (vs. just head/tail).
  """
  head, tail = os.path.split(full_path)

  result = [tail]

  while len(head) > 0:
    [head, tail] = os.path.split(head)
    result.append(tail)

  result = [x for x in result if len(x)]
  return result[::-1]



def makedirs_if_needed(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

        

def finite_mean(v):
  "Returns the mean of the finite elements in v."
  if not np.any(np.isfinite(v)):
    return np.nan
  else:
    return np.mean(v[np.isfinite(v)])




#-------------------------------------------------------------------------------
# Functions for dealing with Tensorflow and/or network models
#-------------------------------------------------------------------------------

def to_one_hot(y, n_classes=None):
  """   
    y : One of:
             * a numpy array of class labels (non-one-hot, obv.)
             * a numpy matrix of predictions that should be made one-hot

    n_classes : the total # of classes  (only needed if y is a vector)
  """
  if np.isscalar(y) or y.size == 1:
    # case where y is scalar
    # note that, even in this case, we return a 2d matrix
    out = np.zeros((1,n_classes), dtype=np.float32)
    out[0,y] = 1

  elif y.ndim == 1:
    # Case where y is a vector
    out = np.zeros((y.size, n_classes), dtype=np.float32)
    out[np.arange(y.size),y] = 1

  else:
    # Case where y is a matrix
    out = np.zeros(y.shape, dtype=np.float32)
    for idx,vals in enumerate(y):
      out[idx,np.argmax(vals)] = 1

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

    pred, loss, grad = sess.run([model.logits, model.loss, model.loss_x], 
                                feed_dict={model.x_tf : x_batch, model.y_tf : y_batch})
    return pred[0,...], loss[0], grad[0]
  else:
    # without a class label we can only predict
    pred = sess.run(model.logits, feed_dict={model.x_tf : x_batch})
    return pred[0,...]



def run_in_batches(sess, x_tf, y_tf, output_tf, x_in, y_in, batch_size):
    """ 
     Runs data through a CNN one batch at a time; gathers all results
     together into a single tensor.  This assumes the output of each
     batch is tensor-like.

        sess      : the tensorflow session to use
        x_tf      : placeholder for input x
        y_tf      : placeholder for input y
        output_tf : placeholder for CNN output
        x_in      : data set to process (numpy tensor)
        y_in      : associated labels (numpy, one-hot encoding)
        batch_size : minibatch size (scalar)

    """
    n_examples = x_in.shape[0]  # total num. of objects to feed

    # determine how many mini-batches are required
    nb_batches = int(math.ceil(float(n_examples) / batch_size))
    assert nb_batches * batch_size >= n_examples

    out = []
    with sess.as_default():
        for start in np.arange(0, n_examples, batch_size):
            # the min() stuff here is to handle the last batch, which may be partial
            end = min(n_examples, start + batch_size)
            start_actual = min(start, n_examples - batch_size)

            feed_dict = {x_tf : x_in[start_actual:end], y_tf : y_in[start_actual:end]}
            output_i = sess.run(output_tf, feed_dict=feed_dict)

            # the slice is to avoid any extra stuff in last mini-batch,
            # which might not be entirely "full"
            skip = start - start_actual
            output_i = output_i[skip:]
            out.append(output_i)

    out = np.concatenate(out, axis=0)
    assert(out.shape[0] == n_examples)
    return out



#-------------------------------------------------------------------------------
# Functions for sampling
#-------------------------------------------------------------------------------

class RandomDirections:
  def __init__(self, shape):
    self._shape = tuple(shape)  # shape of a single direction vector
    self.ortho_group = None     # ortogonal group O(\N); we lazily create


  def gaussian_direction(self, n_samps=1):
    if n_samps == 1:
      shape_out = self._shape
    else:
      shape_out = (n_samps,) + self._shape
    return np.random.randn(*shape_out)


  def haar_direction(self, n_samps):
    # creating this group is computationally expensive (for large dimensions)
    # so we defer creating it until we are sure we need it.
    if self.ortho_group is None:
      self.ortho_group = ortho_group.rvs(dim=np.prod(self._shape))

    if n_samps == 1:
      row = np.random.choice(self.ortho_group.shape[0],1)
      return np.reshape(self.ortho_group[row,:], shape=self._shape)
    else:
      m = self.ortho_group.shape[0]
      rows = np.random.choice(m, min(m, n_samps), replace=False)
      out = self.ortho_group[rows,:]
      return np.reshape(out, (n_samps,) + self._shape)



#-------------------------------------------------------------------------------
# Functions related to analysis of AE
#-------------------------------------------------------------------------------


def distance_to_decision_boundary(sess, model, x, y, direction, d_max, tol=1e-1):
  """ Computes (approximately) the distance one needs to move along
      some direction in order for the CNN to change its decision.  

      x         : a single example/image with shape (rows x cols x channels)
      y         : class label associated with x, in one-hot encoding
                  We only require one-hot so that we don't need a separate
                  parameter indicating the number of classes.
      direction : the search direction; same shape as x
      d_max     : the maximum distance to move along direction (scalar)
      tol       : the maximum size of the interval around the change
  """
  assert(not np.isscalar(y))
  y_scalar = np.argmax(y,axis=1)

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

    preds = sess.run(model.logits, feed_dict={model.x_tf : x_batch})
    y_hat = np.argmax(preds, axis=1)
    if np.all(y_hat == y_scalar):
      a,b = d_max, np.Inf  # label never changed in given interval
      break

    first_change = np.min(np.where(y_hat != y_scalar)[0])
    assert(first_change > 0)

    # refine interval
    a = epsilon_vals[first_change-1]
    b = epsilon_vals[first_change]

  if np.isfinite(b):
    # if a point was found, provide some additional info
    y_new = y_hat[first_change]
    pred, loss, _ = get_info(sess, model, x + b*direction, to_one_hot(y_new, y.size))
    return a, b, y_new, loss
  else:
    # no point was found in the interval
    return a, b, np.nan, np.nan



def loss_function_stats(sess, model, x0, y0, d_max, 
                        n_samp_d=200, k_vals=[2,5,10], verbose=True, dir_sampler=None):
  """ Computes various statistics related to the loss function in the viscinity of 
      a single example (x0,y0).  

        y0 : one-hot encoding of class label y

      Returns: a Pandas data frame 
  """

  assert(not np.isscalar(y0))

  # create a simple data structure to hold results
  #changed to a list of dictionaries of relevant pieces
  class Direction_Stats:
    def __init__(self):
      self.data = []

    def __str__(self):
      df = self.as_dataframe()
      s = ""

      if len(df) <= 0:
        return s

      for dname in ['gradient', 'neg-gradient']:
        tmp = df.loc[df['direction_type'] == dname]
        assert(len(tmp)==1)
        if np.all(np.isfinite(tmp['boundary_distance'])):
          s += '  label first changes (%d->%d) along "%s" direction at distance %0.3f\n' % (tmp['y'], tmp['y_hat'], dname, tmp['boundary_distance'])

      for dname in ['gaussian', 'gaas']:
        tmp = df.loc[df['direction_type'] == dname]
        s += '  expected label change along "%s" direction: %0.3f\n' % (dname, finite_mean(tmp['boundary_distance']))

      return s

    def as_dataframe(self):
      return pd.DataFrame(self.data)

    def append(self, direction_type, y, boundary_distance, y_hat, delta_loss, **kargs):
      assert(np.isscalar(y))
      assert(np.isscalar(y_hat))
      # TODO: could check if boundary_distance is finite to avoid adding Inf to
      #       the table.  However, this is not necessarily an issue.
      entry = {'direction_type' : direction_type, 'y' : y, 'y_hat' : y_hat, 'boundary_distance' : boundary_distance, 'delta_loss' : delta_loss}
      entry.update(kargs)
      self.data.append(entry)

  stats = Direction_Stats()
  if dir_sampler is None:
    dir_sampler = RandomDirections(x0.shape)

  #------------------------------
  # get some basic info about x
  #------------------------------
  pred0, loss0, grad0 = get_info(sess, model, x0, y0)
  assert(np.argmax(pred0) == np.argmax(y0))

  #------------------------------
  # distance in gradient direction
  #------------------------------
  a,b,y_new,loss_new = distance_to_decision_boundary(sess, model, x0, y0, grad0, d_max)
  stats.append('gradient', np.argmax(y0), (a+b)/2., y_new, loss_new-loss0)

  #------------------------------
  # distance in -gradient direction
  #------------------------------
  a,b,y_new,loss_new = distance_to_decision_boundary(sess, model, x0, y0, -grad0, d_max)
  stats.append('neg-gradient', np.argmax(y0), (a+b)/2., y_new, loss_new-loss0)

  #------------------------------
  # distance in random Gaussian directions
  #------------------------------
  for gv in dir_sampler.gaussian_direction(n_samp_d):
    a, b, y_new, loss_new  = distance_to_decision_boundary(sess, model, x0, y0, gv, d_max)
    stats.append('gaussian', np.argmax(y0), (a+b)/2., y_new, loss_new-loss0)

  #------------------------------
  # distance in random orthogonal directions
  #------------------------------
  #for idx, ov in enumerate(dir_sampler.haar_direction(n_samp_d)):
  #  a, b, y_new, loss_new = distance_to_decision_boundary(sess, model, x0, y0, ov, d_max)
  #  stats.append('ortho_group', np.argmax(y0), (a+b)/2., y_new, loss_new-loss0, direction_id=idx)

  #------------------------------
  # distance in gaas directions
  # Note: instead of picking k=n_samp_d we could use some smaller k and draw convex samples from that...
  #------------------------------
  for k_idx, k in enumerate(k_vals):
    # Determine the k directions that define the "subspace"
    Q = gaas(grad0, k)

    # calculate approx. distance to decision boundary for each 
    # GAAS 'basis' vector
    for did, col in enumerate(Q.T): 
      a, b, y_new, loss_new = distance_to_decision_boundary(sess, model, x0, y0, col.reshape(x0.shape), d_max)
      stats.append('gaas', np.argmax(y0), (a+b)/2., y_new, loss_new-loss0, direction_id=did, k=k)

    # Here we test sampling from the GAAS "subspace" by taking a convex
    # combination of the q_i \in Q
    for jj in range(min(n_samp_d, k)):
      coeff = np.random.uniform(size=k)    # coeff : a positive convex combo of q_i
      coeff = coeff / np.sum(coeff)
      q_dir = np.dot(Q,coeff)              # take linear combo
      q_dir = q_dir / norm(q_dir,2)        # back to unit norm
      q_dir = np.reshape(q_dir, x0.shape)

      a,b,y_new,loss_new = distance_to_decision_boundary(sess, model, x0, y0, q_dir, d_max)
      stats.append('gaas_convex_combo', np.argmax(y0), (a+b)/2., y_new, loss_new-loss0, k=k)

  if verbose:
    print("%s" % str(stats))

  df = stats.as_dataframe()
  df['ell2_grad'] = norm(grad0.ravel(), 2)
  return df


#-------------------------------------------------------------------------------

class Tests(unittest.TestCase):
  def test_gaussian_vector(self):
    # we anticipate that vectors generated randomly in this fashion 
    # should be nearly orthogonal with high probability.
    #
    # TODO: look up any analytic result describing the rate 
    # and use this to set tol and dim...
    tol = 1e-1
    n_trials = 100
    result = np.zeros((n_trials,))
    rd = RandomDirections((10000,))

    for ii in range(n_trials):
      a = rd.gaussian_direction();  a = a / norm(a.ravel(),2)
      b = rd.gaussian_direction();  b = b / norm(b.ravel(),2)
      result[ii] = np.abs(np.dot(a,b))

    self.assertTrue(np.sum(result < tol) > (.7*n_trials))


  def test_haar_vector(self):
    # these directions should be orthogonal
    n_samps = 100
    rd = RandomDirections((100,))
    samps = rd.haar_direction(n_samps)
    ip = np.dot(samps, samps.T)
    self.assertTrue(norm(ip - np.eye(n_samps,n_samps), 'fro') < 1e-8)


  def test_to_one_hot(self):
    # test scalar form
    y = 3
    y_oh = to_one_hot(y, 10)
    self.assertTrue(y_oh[0,3] == 1)
    self.assertTrue(np.sum(y_oh) == 1)

    # test vector form
    y = np.array([0,1,2,3,4,5])
    y_oh = to_one_hot(y, 6)
    self.assertTrue(np.all(y_oh == np.eye(6,6)))

    # test matrix form
    self.assertTrue(np.all(y_oh == to_one_hot(y_oh + 10)))


if __name__ == "__main__":
  unittest.main()
