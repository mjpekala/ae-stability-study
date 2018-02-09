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



#K_VALS_FOR_GAAS = [2, 5, 10, 20, 50]
#K_VALS_FOR_GAAS = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # see Evan's email 01/22/2018
#K_VALS_FOR_GAAS = [2, 5, 10, 20, 40, 60, 80, 100] # ~ 6min / example
#K_VALS_FOR_GAAS = [2, 5, 10, 20, 50, 75, 100] # ~ 5min / example
K_VALS_FOR_GAAS = [2, 3, 4, 5, 7, 10, 15, 20, 30] #  2/8/2018 - keep low for computational reasons



def approx_conf(v):
  'a crude measure of "confidence"'
  # TODO: normalize?? (e.g. apply softmax?)
  values = np.sort(v)
  return values[-1] - values[-2]


def main():
  """ Main processing loop.
  """

  batch_size = 32             # CNN mini-batch size
  d_max = 40                  # maximum distance to move in any one direction
  tf.set_random_seed(1099) 

  # TODO: smoothing one-hot class label vectors???

  with h5py.File('cifar10_AE_CH.h5', 'r') as h5:
    # original/clean data
    x = h5['cifar10']['x'].value
    y = h5['cifar10']['y'].value
    all_ae_datasets = [x for x in h5['cifar10'] 
                       if x.startswith('FGM') or x.startswith('I-FGM')]  # UPDATE as needed for new AE!

    print('x min/max: %0.2f / %0.2f' % (np.min(x), np.max(x)))
    
    dsamp = ae_utils.RandomDirections(x[0,...].shape)   # sampling strategies
    df_list = []                                        # stores intermediate results
    tic = time.time()


    config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True
    with tf.Graph().as_default(), tf.Session(config=config) as sess:
      model = cifar10.Cifar10(sess, './Weights_n01')  # XXX: update if using multi model!

      for ii in range(200):  # TEMP: process only a subset for now
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
        print('        (net elapsed time: %0.2f min)' % ((time.time()-tic)/60.))
        sys.stdout.flush()

        # If the original example was misclassified, we ignore this example
        # since the notion of AE makes less sense.
        if not yi_scalar == np.argmax(pred_clean):
          continue

        print(' CLEAN EXAMPLE:')

        #----------------------------------------
        # sample directions (clean/original examples)
        #----------------------------------------
        stats = pd.DataFrame(ae_utils.loss_function_stats(sess, model, xi, yi_oh, d_max, dir_sampler=dsamp, k_vals=K_VALS_FOR_GAAS))
        stats['Dataset'] = 'cifar10'
        stats['Example#'] = ii
        stats['Approx_conf'] = approx_conf(pred_clean)
        df_list.append(stats.copy())

        #----------------------------------------
        # look at the corresponding AE
        #----------------------------------------
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

          stats = ae_utils.loss_function_stats(sess, model, xi_adv, y_hat_ae, d_max, dir_sampler=dsamp, k_vals=K_VALS_FOR_GAAS)
          stats['Dataset'] = ae_dataset
          stats['Example#'] = ii
          stats['Approx_conf'] = approx_conf(pred_ae)
          stats['delta_l2'] = h5['cifar10'][ae_dataset]['delta_l2'][ii]
          df_list.append(stats.copy())


  #--------------------------------------------------
  # save results to file for subsequent analysis
  #--------------------------------------------------
  master_stats = pd.concat(df_list)
  master_stats.to_pickle('cifar10_stats_df_CH.pkl')




if __name__ == "__main__":
  from tensorflow.python.client import device_lib
 
  # Use CUDA_AVAIABLE_DEVICES to restrict this to a given gpu 
  # (see the Makefile)
  avail = device_lib.list_local_devices()
  print([x.name for x in avail])

  main()
