"""  Here we use CIFAR-10 to explore differences between clean and AE 
     in terms of their nearest distance to a decision boundary.

  Example usage:
     PYTHONPATH=./cleverhans python study_cifar10.py
"""

__author__ = "mjp,ef"
__date__ = "december, 2017"


import numpy as np
import pdb

import tensorflow as tf
import keras

from models import cifar10
import ae_utils





def approx_conf(v):
  'a crude measure of "confidence"'
  values = np.sort(v)
  return values[-1] - values[-2]


def main():
  batch_size = 32       # CNN mini-batch size
  d_max = 100           # maximum distance to move in any one direction

  tf.set_random_seed(1099) 

  #--------------------------------------------------
  # Load clean and AE data.
  # In the future we perhaps might load muliple different AE data sets...
  #--------------------------------------------------
  X_train, Y_train, X_test, Y_test = cifar10.data_cifar10()

  f = np.load('cifar10_fgsm_eps0.30.npz')
  X_adv = f['x']

  print(X_test.shape, X_adv.shape, Y_test.shape)
  assert(np.all(X_test.shape == X_adv.shape))

  #--------------------------------------------------
  # Decision boundary analysis
  #--------------------------------------------------
  with tf.Graph().as_default(), tf.Session() as sess:
    keras.backend.set_image_dim_ordering('tf')
    keras.backend.set_session(sess)
    model = cifar10.Cifar10(sess, num_in_batch=batch_size)

    #for ii in range(X_test.shape[0]):
    for ii in range(100):
      xi = X_test[ii,...]
      yi = Y_test[ii,...]
      xi_adv = X_adv[ii,...]

      # TODO: should we smooth labels prior to the analysis below?

      pred_clean = ae_utils.get_info(sess, model, xi)
      y_hat_clean = np.zeros(pred_clean.shape);  y_hat_clean[np.argmax(pred_clean)] = 1

      pred_ae = ae_utils.get_info(sess, model, xi_adv)
      y_hat_ae = np.zeros(pred_ae.shape);  y_hat_ae[np.argmax(pred_ae)] = 1

      print('\nEXAMPLE %d, y=%d, y_hat=%d, y_hat_ae=%d, conf=%0.3f' % (ii, np.argmax(yi), np.argmax(y_hat_clean), np.argmax(y_hat_ae), approx_conf(pred_clean)))

      if np.argmax(y_hat_clean) == np.argmax(yi): # only study distances for correctly classified examples
        ae_utils.distance_to_decision_boundary_stats(sess, model, xi, yi, d_max)

        if np.argmax(y_hat_ae) != np.argmax(yi): # for AE, we only care about successful attack
          print('   For AE:')
          ae_utils.distance_to_decision_boundary_stats(sess, model, xi_adv, y_hat_ae, d_max)



if __name__ == "__main__":
  main()
