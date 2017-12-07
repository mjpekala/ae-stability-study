"""
"""

import numpy as np
import pdb

import tensorflow as tf
import keras

from models import cifar10
import ae_utils



def get_info(sess, model, x, y):
    assert(y.size == model.num_classes)  # y should be one-hot

    x_batch = np.zeros(model.batch_shape)
    x_batch[0,...] = x
    y_batch = np.zeros((model.batch_shape[0], model.num_classes))
    y_batch[0,...] = y

    pred, loss, grad = sess.run([model.output, model.loss, model.loss_x], 
                                feed_dict={model.x_tf : x_batch, model.y_tf : y_batch})

    return pred[0,...], loss[0], grad[0]




if __name__ == "__main__":
  batch_size = 32
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

      pred, loss, grad = get_info(sess, model, xi, yi)
      y_hat = np.argmax(pred)
      print('\n example %d, y=%d, y_hat=%d' % (ii, np.argmax(yi), np.argmax(pred)))

      # if the classifier made a mistake, we do not worry about AE.
      if y_hat != np.argmax(yi):
        continue

      # some analysis of distance to boundary
      a,b = ae_utils.distance_to_decision_boundary(sess, model, xi, y_hat, grad, 100)
      print('   label of clean example first changes along gradient direction in [%0.3f, %0.3f]' % (a,b))

      a,b = ae_utils.distance_to_decision_boundary(sess, model, xi, y_hat, ae_utils.gaussian_vector(grad.shape), 100)
      print('   label of clean example first changes along random direction in [%0.3f, %0.3f]' % (a,b))


