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
from gaas import gaas



def get_info(sess, model, x, y=None):
  """ Queries CNN for information about x.
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
    




if __name__ == "__main__":
  batch_size = 32       # CNN mini-batch size
  d_max = 100           # maximum distance to move in any one direction
  n_samp_d = 30         # number of directions to sample

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

      # TODO: should we smooth labels prior to the following analysis???

      #--------------------------------------------------
      # analysis for clean examples
      #--------------------------------------------------
      pred, loss, grad = get_info(sess, model, xi, yi)
      y_hat = np.argmax(pred)
      print('\nEXAMPLE %d, y=%d, y_hat=%d' % (ii, np.argmax(yi), y_hat))

      if y_hat == np.argmax(yi):  # only study distances for correctly classified examples
        a,b = ae_utils.distance_to_decision_boundary(sess, model, xi, y_hat, grad, d_max)
        print('   label of clean example first changes along gradient direction in [%0.3f, %0.3f]' % (a,b))

        # random directions
        for jj in range(n_samp_d):
          a,b = ae_utils.distance_to_decision_boundary(sess, model, xi, y_hat, ae_utils.gaussian_vector(grad.shape), d_max)
          print('   label of clean example first changes along random direction in [%0.3f, %0.3f]' % (a,b))

        # gaas directions
        # Note: instead of picking k=n_samp_d we could use some smaller k and draw convex samples from that...
        Q = gaas(grad, n_samp_d)
        for jj in range(Q.shape[1]):
          q_j = np.reshape(Q[:,jj], grad.shape)
          a,b = ae_utils.distance_to_decision_boundary(sess, model, xi, y_hat, q_j, d_max)
          print('   label of clean example first changes along GAAS direction %d in [%0.3f, %0.3f]' % (jj,a,b))

      #--------------------------------------------------
      # analysis for AE
      #--------------------------------------------------
      pred = get_info(sess, model, xi_adv)
      y_hat_scalar = np.argmax(pred)
      print('   AE %d, y=%d, y_hat=%d' % (ii, np.argmax(yi), y_hat_scalar))

      # we are not really interested in unsuccessful AE
      if y_hat_scalar != np.argmax(yi):
        y_hat = np.zeros(yi.shape)
        y_hat[y_hat_scalar] = 1

        pred, loss, grad = get_info(sess, model, xi_adv, y_hat)
        assert(np.argmax(pred) == np.argmax(y_hat))

        a,b = ae_utils.distance_to_decision_boundary(sess, model, xi_adv, y_hat_scalar, grad, d_max)
        print('   label of AE first changes along gradient direction in [%0.3f, %0.3f]' % (a,b))

        a,b = ae_utils.distance_to_decision_boundary(sess, model, xi_adv, y_hat_scalar, ae_utils.gaussian_vector(grad.shape), d_max)
        print('   label of AE first changes along random direction in [%0.3f, %0.3f]' % (a,b))

