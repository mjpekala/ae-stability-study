"""  Here we use CIFAR-10 to explore differences between clean and AE 
     in terms of their nearest distance to a decision boundary.

  Example usage:
     PYTHONPATH=./cleverhans python study_cifar10.py
"""

__author__ = "mjp,ef"
__date__ = "december, 2017"


import sys, time, os
import h5py

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
  """ Main processing loop.
  """

  batch_size = 32             # CNN mini-batch size
  d_max = 20                  # maximum distance to move in any one direction
  tf.set_random_seed(1099) 
  k_vals_for_gaas = [2,3,4,5,10,20,50,100]

  # TODO: smoothing one-hot class label vectors???

  with h5py.File('cifar10_AE.h5', 'r') as h5:
    # original/clean data
    x = h5['cifar10']['x'].value
    y = h5['cifar10']['y'].value
    all_ae_datasets = [x for x in h5['cifar10'] if x.startswith('FGM')]  # for now, we only consider FGM

    print('x min/max: %0.2f / %0.2f' % (np.min(x), np.max(x)))
    
    dsamp = ae_utils.RandomDirections(x[0,...].shape)   # sampling strategies
    df_list = []                                        # stores intermediate results


    #config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
      model = cifar10.Cifar10(sess, './Weights')

      for ii in range(1500):  # TEMP: process only a subset for now
        xi = x[ii,...]
        yi_scalar = y[ii] 
        yi_oh = ae_utils.to_one_hot(yi_scalar, 10)

        # Use the CNN to predict label.
        # Presumably these estimates should match the one in the database
        # (assuming the same network...) but we recompute anyway...
        pred_clean = ae_utils.get_info(sess, model, xi)
        y_hat_clean = ae_utils.to_one_hot(np.argmax(pred_clean), 10)

        print('-----------------------------------------------------------------------------------')
        print('EXAMPLE %d, y=%d, y_hat=%d, conf=%0.3f' % (ii, yi_scalar, np.argmax(pred_clean), approx_conf(pred_clean)))
        sys.stdout.flush()

        # If the original example was misclassified, we ignore this example
        # since the notion of AE makes less sense.
        if not yi_scalar == np.argmax(pred_clean):
          continue

        # sample directions
        stats = pd.DataFrame(ae_utils.loss_function_stats(sess, model, xi, yi_oh, d_max, dir_sampler=dsamp, k_vals=k_vals_for_gaas))
        stats['Dataset'] = 'cifar10'
        stats['Example#'] = ii
        stats['Approx_conf'] = approx_conf(pred_clean)
        df_list.append(stats.copy())

        # look at the corresponding AE
        for ae_dataset in all_ae_datasets:
          print(' %s AE:' % ae_dataset)

          xi_adv = h5['cifar10'][ae_dataset]['x'].value[ii,...]

          pred_ae = ae_utils.get_info(sess, model, xi_adv)
          y_hat_ae = ae_utils.to_one_hot(np.argmax(pred_ae), 10)

          if 0:
            y_hat_clean = ae_utils.smoothed_one_hot(y_hat_clean)
            y_hat_ae = ae_utils.smoothed_one_hot(y_hat_ae)

          # for now, we only care about **successful** AE
          if np.argmax(y_hat_ae) == yi_scalar:
            print('   attack unsuccessful, skipping...\n')
            continue

          stats = ae_utils.loss_function_stats(sess, model, xi_adv, y_hat_ae, d_max, dir_sampler=dsamp)
          stats['Dataset'] = ae_dataset
          stats['Example#'] = ii
          stats['Approx_conf'] = approx_conf(pred_ae)
          df_list.append(stats.copy())

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
