""" Some utilities useful for our AE studies.
"""

__author__ = "mjp"
__date__ = "dec, 2017"


import numpy as np
from numpy.linalg import norm

import unittest


def smooth_one_hot_predictions(p, num_classes):
  """Given a vector (*not* a full matrix) of predicted class labels p, 
     generates a 'smoothed' one-hot prediction *matrix*.
  """
  out = (1./num_classes) * np.ones((p.size, num_classes), dtype=np.float32)
  for ii in range(p.size):
    out[ii,p[ii]] = 0.9
  return out



def gaussian_vector(vec_shape):
  """ Generates a vector with iid gaussian entries

    vec_shape : a tuple indicating the shape of the vector/tensor to be created.
  """
  rv = np.random.randn(*vec_shape) 
  return rv / norm(rv.flatten(),2)



def distance_to_decision_boundary(sess, model, x, 
                                   y0=None, direction=None, 
                                   epsilon_max=100, epsilon0=.5):
  """ Computes (approximately) the distance one needs to move along
      some direction in order for the CNN to change its decision.  

      The distance is denoted epsilon; if no direction is specified, the gradient 
      of the loss evaluated will be used by default.
  """

  # compute the initial prediction (if needed)
  if y0 is None:
    pred0 = tf_run(sess, model.output, feed_dict={model.x_tf : x})
    y0_scalar = np.argmax(pred0)
    y0 = nets.smooth_one_hot_predictions(y0_scalar, model._num_classes)

  # use gradient direction by default (if no explicit direction provided)
  if direction is None:
    grad = tf_run(sess, model.loss_x, feed_dict={model.x_tf : x, model.y_tf : y0})
    direction = grad.astype(np.float64) 

  # normalize vector
  direction = direction / norm(direction.flatten(),2)

  # brute force search
  epsilon = epsilon0 
  epsilon_lb = 0
  done = False

  while (epsilon < epsilon_max) and (not done):
    x_step = x + epsilon * direction
    pred_end = tf_run(sess, model.output, feed_dict={model.x_tf : x_step})
    if np.argmax(pred_end) != np.argmax(y0):
      # prediction changed; all done
      done = True
    else:
      # keep searching
      epsilon_lb = epsilon
      epsilon = epsilon * 1.1

  # XXX: could search between lb and epsilon for more precise value

  return epsilon


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
