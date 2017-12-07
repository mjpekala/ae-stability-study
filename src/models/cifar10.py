""" Implements a simple model for CIFAR-10

  EXAMPLE (command-line usage):
        PYTHONPATH=./cleverhans python ./models/cifar10.py

  The command above will train the model if weights do not yet exist; 
  if the weights do exist, the command will instead run a simple AE
  attack against it.

  Note: we take much of this from cleverhans
        (see cleverhans/examples/ex_cifar10_tf.py)
"""

import os
import pdb

import numpy as np

import tensorflow as tf
slim = tf.contrib.slim


# TODO: replace cleverhans CNN with a pure tensorflow CNN for cifar10
import keras
from keras.datasets import cifar10
from keras import backend
from keras.utils import np_utils

from cleverhans.utils_keras import cnn_model 
from cleverhans.utils_tf import model_loss, model_train, model_eval, batch_eval


class Cifar10:
  """ Our desired model API.
  """

  def __init__(self, sess, num_in_batch=1):
    self.batch_shape = [num_in_batch, 32, 32, 3]
    self._num_classes = 10
    self._weights_file = './Weights/cifar10.ckpt'

    self.x_tf = tf.placeholder(tf.float32, shape=self.batch_shape)
    self.y_tf = tf.placeholder(tf.float32, shape=[self.batch_shape[0], self._num_classes])

    # use cleverhans API to create/access a tensorflow model
    _ = cnn_model(img_rows=32, img_cols=32, channels=3)
    self.output = _(self.x_tf)  # network predictions
    self.loss = model_loss(self.y_tf, self.output, mean=False)
    self.loss_x = tf.gradients(self.loss, self.x_tf)

    saver = tf.train.Saver()
    try:
      saver.restore(sess, self._weights_file)
      print('loaded cifar10 weights successfully')
    except:
      print('WARNING: CIFAR10 model needs to be trained!!')


def data_cifar10():
    """ This function taken directly from cleverhans.
    """

    # These values are specific to CIFAR10
    img_rows = 32
    img_cols = 32
    nb_classes = 10

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    if keras.backend.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, Y_train, X_test, Y_test


def train_cifar10(sess, model):
    """ This function should produce comparable results to 
        cleverhans/examples/ex_cifar10_tf.py
    """
    batch_size=128

    X_train, Y_train, X_test, Y_test = data_cifar10()

    assert(Y_train.shape[1] == 10.)
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Train model
    def evaluate():
        # Evaluate the accuracy of the CIFAR10 model on legitimate test
        # examples
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, model.x_tf, model.y_tf, model.output, 
                              X_test, Y_test, args=eval_params)
        assert X_test.shape[0] == 10000, X_test.shape
        print('Test accuracy on legitimate test examples: ' + str(accuracy))

    train_params = {
        'nb_epochs': 10,
        'batch_size': batch_size,
        'learning_rate': 0.001
    }
    model_train(sess, model.x_tf, model.y_tf, model.output, 
                X_train, Y_train, evaluate=evaluate, args=train_params)

    saver = tf.train.Saver()
    saver.save(sess, model._weights_file)


def _demo_model(sess, model):
    batch_size = 128
    X_train, Y_train, X_test, Y_test = data_cifar10()

    #
    # run on clean data
    #
    eval_params = {'batch_size': batch_size}
    accuracy = model_eval(sess, model.x_tf, model.y_tf, model.output, 
                          X_test, Y_test, args=eval_params)
    assert X_test.shape[0] == 10000, X_test.shape
    print('Test accuracy on legitimate test examples: ' + str(accuracy))

    #
    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    #
    from cleverhans.attacks import fgsm

    adv_x = fgsm(model.x_tf, model.output, eps=0.3)
    eval_params = {'batch_size': batch_size}
    X_test_adv, = batch_eval(sess, [model.x_tf], [adv_x], [X_test], args=eval_params)
    assert X_test_adv.shape[0] == 10000, X_test_adv.shape

    # Evaluate the accuracy of the CIFAR10 model on adversarial examples
    accuracy = model_eval(sess, model.x_tf, model.y_tf, model.output, 
                          X_test_adv, Y_test, args=eval_params)
    print('Test accuracy on adversarial examples: ' + str(accuracy))
 


if __name__ == "__main__":

  with tf.Graph().as_default(), tf.Session() as sess:
    keras.backend.set_image_dim_ordering('tf')
    keras.backend.set_session(sess)
    model = Cifar10(sess, num_in_batch=None)

    if not os.path.exists(model._weights_file + '.index'):
      # use this to train the model
      train_cifar10(sess, model)
    else:
      # make sure weights are loaded ok
      _demo_model(sess, model)
