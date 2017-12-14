"""  Here we use CIFAR-10 to explore differences between clean and AE 
     in terms of their nearest distance to a decision boundary.

  Example usage:
     PYTHONPATH=./cleverhans python study_cifar10.py
"""

__author__ = "mjp,ef"
__date__ = "december, 2017"


import time

import numpy as np
import pdb

import pandas as pd
from scipy.io import savemat

import tensorflow as tf
#import keras

#from models import cifar10_lite as cifar10
from models.cifar10 import cifar10_wrapper as cifar10
import ae_utils



def approx_conf(v):
  'a crude measure of "confidence"'
  # TODO: normalize?? (e.g. apply softmax?)
  values = np.sort(v)
  return values[-1] - values[-2]


def main():
  batch_size = 32       # CNN mini-batch size
  d_max = 100           # maximum distance to move in any one direction
  eps = 0.05

  tf.set_random_seed(1099) 

  #--------------------------------------------------
  # Load clean and AE data.
  # In the future we perhaps might load muliple different AE data sets...
  #--------------------------------------------------
  #_, _, X_test, Y_test = cifar10.data_cifar10()
  data_file = '/home/pekalmj1/Data/CIFAR10/cifar-10-batches-py/test_batch'
  X_test, Y_test = cifar10.load_cifar10_python(data_file)

  f = np.load('./cifar10_fgsm_eps%0.2f.npz' % eps)  # created by cifar10_wrapper.py
  X_adv = f['x_fgsm']

  print(X_test.shape, X_adv.shape, Y_test.shape)
  assert(np.all(X_test.shape == X_adv.shape))

  #--------------------------------------------------
  # Decision boundary analysis
  #--------------------------------------------------
  config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
  with tf.Graph().as_default(), tf.Session(config=config) as sess:
    #keras.backend.set_image_dim_ordering('tf')
    #keras.backend.set_session(sess)
    model = cifar10.Cifar10(sess)

    dsamp = ae_utils.RandomDirections(X_test[0,...].shape)
    df_list = []  # stores intermediate results

    #for ii in range(X_test.shape[0]):
    for ii in range(300):  # for now we only consider a subset of examples (saves time)
      xi = X_test[ii,...]
      yi_scalar = Y_test[ii]  # NOTE: whether this is one-hot or not depends on data source!
      yi_oh = ae_utils.to_one_hot(yi_scalar, 10)
      xi_adv = X_adv[ii,...]

      # use the CNN to predict label for clean and AE
      pred_clean = ae_utils.get_info(sess, model, xi)
      y_hat_clean = ae_utils.to_one_hot(np.argmax(pred_clean), 10)

      pred_ae = ae_utils.get_info(sess, model, xi_adv)
      y_hat_ae = ae_utils.to_one_hot(np.argmax(pred_ae), 10)

      print('\nEXAMPLE %d, y=%d, y_hat=%d, y_hat_ae=%d, conf=%0.3f' % (ii, yi_scalar, np.argmax(y_hat_clean), np.argmax(y_hat_ae), approx_conf(pred_clean)))

      # OPTIONAL: smoothing one-hot class label vectors
      if 0:
        y_hat_clean = ae_utils.smoothed_one_hot(y_hat_clean)
        y_hat_ae = ae_utils.smoothed_one_hot(y_hat_ae)

      # for now, we only care about examples where:
      #  1.  The network correctly classified the clean example and
      #  2.  The AE attack was successful
      if np.argmax(y_hat_clean) != yi_scalar or np.argmax(y_hat_ae) == yi_scalar:
        continue

      stats = pd.DataFrame(ae_utils.loss_function_stats(sess, model, xi, yi_oh, d_max, dir_sampler=dsamp))
      stats['Dataset'] = 'cifar10'
      stats['Example#'] = ii
      stats['Approx_conf'] = approx_conf(pred_clean)
      df_list.append(stats.copy())

      print(' CORRESPONDING AE :')
      stats_ae = pd.DataFrame(ae_utils.loss_function_stats(sess, model, xi_adv, y_hat_ae, d_max, dir_sampler=dsamp))
      stats_ae['Dataset'] = 'cifar10-Adv-FGM-%0.2f' % eps
      stats_ae['Example#'] = ii
      stats['Approx_conf'] = approx_conf(pred_ae)
      df_list.append(stats_ae.copy())

  #--------------------------------------------------
  # save results
  #--------------------------------------------------
  master_stats = pd.concat(df_list)
  master_stats.to_pickle('cifar10_stats_df.pkl')


if __name__ == "__main__":
  from tensorflow.python.client import device_lib
 
  # Use CUDA_AVAIABLE_DEVICES to restrict this to a given gpu 
  avail = device_lib.list_local_devices()
  print([x.name for x in avail])

  main()
