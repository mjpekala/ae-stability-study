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
    


def distance_to_boundary_analysis(sess, model, x0, y0, d_max, n_samp_d=30):
  pred, loss, grad = get_info(sess, model, x0, y0)
  y_hat = np.argmax(pred)

  assert(y_hat == np.argmax(y0))

  #------------------------------
  # distance in gradient direction
  #------------------------------
  a,b = ae_utils.distance_to_decision_boundary(sess, model, x0, y_hat, grad, d_max)
  print('   label first changes along gradient direction in [%0.3f, %0.3f]' % (a,b))

  a,b = ae_utils.distance_to_decision_boundary(sess, model, x0, y_hat, -grad, d_max)
  print('   label first changes along neg. gradient direction in [%0.3f, %0.3f]' % (a,b))

  #------------------------------
  # distance in random directions
  #------------------------------
  d_min_rand = np.zeros((n_samp_d,))
  d_max_rand = np.zeros((n_samp_d,))
  for jj in range(n_samp_d):
    d_min_rand[jj], d_max_rand[jj] = ae_utils.distance_to_decision_boundary(sess, model, x0, y_hat, ae_utils.gaussian_vector(grad.shape), d_max)

  d_max_rand[d_max_rand==np.Inf] = np.nan
  print('   expected first label change along random direction [%0.3f, %0.3f]' % (np.mean(d_min_rand), np.nanmean(d_max_rand)))

  #------------------------------
  # distance in gaas directions
  # Note: instead of picking k=n_samp_d we could use some smaller k and draw convex samples from that...
  #------------------------------
  for k in [2,10,n_samp_d]:
    d_min_gaas = np.zeros((k,))
    d_max_gaas = np.zeros((k,))
    Q = gaas(grad, k)
    for jj in range(Q.shape[1]):
      q_j = np.reshape(Q[:,jj], grad.shape)
      d_min_gaas[jj], d_max_gaas[jj] = ae_utils.distance_to_decision_boundary(sess, model, x0, y_hat, q_j, d_max)

    d_max_rand[d_max_rand==np.Inf] = np.nan
    print('   expected first label change along k=%d GAAS direction [%0.3f, %0.3f]' % (k, np.mean(d_min_gaas), np.nanmean(d_max_gaas)))



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

      pred_clean = get_info(sess, model, xi)
      y_hat_clean = np.zeros(pred_clean.shape);  y_hat_clean[np.argmax(pred_clean)] = 1

      pred_ae = get_info(sess, model, xi_adv)
      y_hat_ae = np.zeros(pred_ae.shape);  y_hat_ae[np.argmax(pred_ae)] = 1

      print('\nEXAMPLE %d, y=%d, y_hat=%d, y_hat_ae=%d, conf=%0.3f' % (ii, np.argmax(yi), np.argmax(y_hat_clean), np.argmax(y_hat_ae), approx_conf(pred_clean)))

      if np.argmax(y_hat_clean) == np.argmax(yi): # only study distances for correctly classified examples
        distance_to_boundary_analysis(sess, model, xi, yi, d_max)

        if np.argmax(y_hat_ae) != np.argmax(yi): # for AE, we only care about successful attack
          print('   For AE:')
          distance_to_boundary_analysis(sess, model, xi_adv, y_hat_ae, d_max)



if __name__ == "__main__":
  main()
