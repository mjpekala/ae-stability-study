"""
  Implements the Gradient aligned adversarial "subspace" (GAAS) of [tra17].

  REFERENCES:
   [tra17] Tramer et al. "The Space of Transferable Adversarial
           Examples," arXiv 2017.
   [gvl96] Golub and Van Loan "Matrix Computations" 1996.

"""


__author__ = "mjp"
__date__ = 'november, 2017'


import numpy as np
from numpy.linalg import norm

import unittest
import pdb


def gaas(g, k, sanity_check=True):
  """
     g : the gradient of the loss (a tensor)
     k : the GAAS dimensionality (a scalar)

     Returns:
     R : a dxk matrix of k orthogonal vectors satisfying the GAAS conditions

  """
  g = g.flatten() 

  d = g.size
  R = np.zeros((d,k))                         # columns of R are the GAAS r_i
  z = np.zeros((d,));  z[:k] = 1/np.sqrt(k);  # this is z from proof of lemma 1 in [tra17]

  #--------------------------------------------------
  # SPECIAL CASE: if k is 1, just return the trivial result g / ||g||
  #
  #--------------------------------------------------
  #if k == 1:
  #  R[:,0] = g / norm(g,2)
  #  return R


  v_s, beta_s = householder_vec(z)
  v_r, beta_r = householder_vec(g)

  #--------------------------------------------------
  # To calculate the r_i we use:
  #
  #     r_i := Q' e_i
  #          = R' S e_i
  #          = R S e_i
  #
  # where R = R' from the symmetry of Householder matrices
  # (follows from symmetry of I and vv').
  #--------------------------------------------------
  for ii in range(k):
    e_i = np.zeros((d,));  e_i[ii] = 1;
    sei = apply_householder_to_vector(v_s, beta_s, e_i)
    r_i = apply_householder_to_vector(v_r, beta_r, sei)
    R[:,ii] = r_i


  #--------------------------------------------------
  # (optional) check the solution for correctness
  #--------------------------------------------------
  if sanity_check:
    # the r_i should be orthonormal
    RtR = np.dot(R.T, R)
    assert(norm(RtR-np.eye(k,k), 'fro') < 1e-5)
   
    # make sure Qg = ||g||_2 z
    #
    # Note: the transpose on R below is because I stored
    #       the r_i as columns in R.
    #
    err = np.dot(R.T, g) - norm(g,2) * z[:k]
    assert(norm(err,2) < 1e-5)
 
    # make sure <g,r_i> behaves as expected.
    for ii in range(k):
        gtr = np.dot(g, R[:,ii])
        err = np.dot(g, r_i) - norm(g,2) / np.sqrt(k)
        assert(abs(err) < 1e-5)

  return R



def householder_vec(x):
  """ Returns elements needed to construct Householder 
      reflection matrix for the vector x, i.e.

          H = I - \beta v v'

      where H x = ||x||_2 e_1

  See Algorithm 5.1.1 in Golub and Van Loan.
  """ 
  n = x.size
  v = np.ones((n,));  v[1:] = x[1:]

  sigma = np.dot(x[1:], x[1:])
  if sigma == 0:
    beta = 0
  else:
    mu = np.sqrt(x[0] ** 2 + sigma)
    if x[0] <= 0:
        v[0] = x[0] - mu
    else:
        v[0] = -sigma / (x[0] + mu)
    beta = 2 * (v[0] ** 2) / (sigma + v[0]**2)
    v = v / v[0]

  return v, beta



def apply_householder_to_vector(v, beta, x):
  """ Computes Householder reflection of vector x.

    Applying a Householder transformation H to a vector x
    does not require the explict construction of H since
   
        H x = (I - \beta vv') x = x - \beta v (v' x)
   
    In particular, this avoids the deadly outer product vv'.
    See also 5.1.4 in Golub and Van Loan.
  """
  return x - beta * v * np.dot(v, x)



if __name__ == "__main__":
  # example usage
  for n_trials in range(10):
    g = np.random.rand(300,1)
    k = 20
    R = gaas(g,k)

  print('[info]: GAAS calculation looks ok!')
