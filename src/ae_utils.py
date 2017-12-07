""" Some utilities useful for our AE studies.
"""

__author__ = "mjp"
__date__ = "dec, 2017"


import numpy as np
from numpy.linalg import norm

import pdb, unittest


def smooth_one_hot_predictions(p, num_classes):
  """Given a vector (*not* a full matrix) of predicted class labels p, 
     generates a 'smoothed' one-hot prediction *matrix*.
  """
  out = (1./num_classes) * np.ones((p.size, num_classes), dtype=np.float32)
  for ii in range(p.size):
    out[ii,p[ii]] = 0.9
  return out



def gaussian_vector(vec_shape):
  """ Generates a (normalized) vector with iid gaussian entries

    vec_shape : a tuple indicating the shape of the vector/tensor to be created.
  """
  rv = np.random.randn(*vec_shape) 
  return rv / norm(rv.flatten(),2)



def distance_to_decision_boundary(sess, model, x, y, direction, d_max, tol=1e-1):
  """ Computes (approximately) the distance one needs to move along
      some direction in order for the CNN to change its decision.  
  """

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
      return 0, np.Inf  # fail to find place where label changes

    first_change = np.min(np.where(y_hat != y)[0])
    assert(first_change > 0)

    # refine interval
    a = epsilon_vals[first_change-1]
    b = epsilon_vals[first_change]

  return a,b


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
