""" SUBSPACE_ATTACK  Code for using GAAS [tra17] to analyze AE.


  References:
  [tra17] Tramer et al. "The Space of Transferable Adversarial Examples," 2017.

"""

__author__ = "mjp"
__date__ = "november, 2017"


import sys, os
import unittest
import pdb

import numpy as np
from numpy.linalg import norm
from scipy.linalg import block_diag as blkdiag
from scipy.misc import imread, imsave

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception, resnet_v2
slim = tf.contrib.slim
import matplotlib.pyplot as plt
import pandas as pd

#-------------------------------------------------------------------------------
# Functions for dealing with data and tensorflow models
#-------------------------------------------------------------------------------


def _input_filenames(input_dir):
    all_files = tf.gfile.Glob(os.path.join(input_dir, '*.png'))
    all_files.sort()
    return all_files


def _load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:
      filenames: list file names without path of each image
        length of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]

    for filepath in _input_filenames(input_dir):
        with tf.gfile.Open(filepath, mode='rb') as f:
            image = imread(f, mode='RGB').astype(np.float) / 255.0

        # Images for inception classifier are normalized to be in [-1, 1] interval.
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))

        idx += 1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0

    # This is a partial batch left over at end.
    # Note that images will still have the proper size.
    if idx > 0:
        yield filenames, images


def smooth_one_hot_predictions(p, num_classes):
    """Given a vector (*not* a full matrix) of predicted class labels p,
       generates a 'smoothed' one-hot prediction *matrix*.
    """
    out = (1./num_classes) * np.ones((p.size, num_classes), dtype=np.float32)
    for ii in range(p.size):
        out[ii,p[ii]] = 0.9
    return out




class InceptionV3:
    """ Bare-bones interface to the InceptionV3 model.
    """

    def __init__(self, sess):
        self.batch_shape = [1, 299, 299, 3]  # for now, we attack one image at a time
        self._num_classes = 1001
        self._scope = 'InceptionV3'
        self._weights_file = r"C:\Users\fulleed1\Documents\Adversarial Learning\AE_Detection_v2\Weights\inception_v3.ckpt"

        #
        # network inputs
        #
        self.x_tf = tf.placeholder(tf.float32, shape=self.batch_shape)
        self.y_tf = tf.placeholder(tf.float32, shape=[self.batch_shape[0], self._num_classes])

        #
        # network outputs
        #
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(self.x_tf, num_classes=self._num_classes, is_training=False, scope=self._scope)

            # output 1: the raw network predictions (a function of x only)
            self.output = end_points['Predictions']

            # output 2: the gradient of a loss function (a function of both x and y)
            cross_entropy_loss = tf.losses.softmax_cross_entropy(self.y_tf,
                                                                 logits,
                                                                 label_smoothing=0.1,
                                                                 weights=1.0)

            cross_entropy_loss += tf.losses.softmax_cross_entropy(self.y_tf,
                                                                  end_points['AuxLogits'],
                                                                  label_smoothing=0.1,
                                                                  weights=0.4)

            self.loss = cross_entropy_loss
            self.nabla_x_loss = tf.gradients(self.loss, self.x_tf)[0]

            #
        # load weights
        #
        saver = tf.train.Saver(slim.get_model_variables(scope=self._scope))
        saver.restore(sess, self._weights_file)



#-------------------------------------------------------------------------------


def fgsm_attack(sess, model, epsilon, input_dir, output_dir):
    """ Simple implementation of a fast gradient sign attack.
        This is just for us to test our code and is not of particular interest.
        Modified to output an array of x, x_adv, loss_x, loss_adv
    """

    n_changed = 0
    n_total = 0
    working_exs = []
    for batch_id, (filenames, x0) in enumerate(_load_images(input_dir, model.batch_shape)):
        n = len(filenames)

        # for now, take predictions on clean data as truth.
        pred0 = sess.run(model.output, feed_dict={model.x_tf : x0})
        y0 = smooth_one_hot_predictions(np.argmax(pred0, axis=1), model._num_classes)

        # compute the gradient
        feed_dict = {model.x_tf : x0, model.y_tf : y0}
        grad = sess.run(model.nabla_x_loss, feed_dict=feed_dict)
        loss = sess.run(model.loss, feed_dict=feed_dict)
        # construct AE
        x_adv = x0 + epsilon * np.sign(grad)
        x_adv = np.clip(x_adv, -1, 1)  # make sure AE respects box contraints

        # predicted label of AE
        pred1 = sess.run(model.output, feed_dict={model.x_tf : x_adv})
        y1 = smooth_one_hot_predictions(np.argmax(pred1, axis=1), model._num_classes)
        feed_dict = {model.x_tf: x_adv, model.y_tf: y1}
        grad_adv = sess.run(model.nabla_x_loss, feed_dict=feed_dict)
        loss_adv = sess.run(model.loss, feed_dict=feed_dict)
        is_same = np.argmax(y1[:n,:],axis=1) == np.argmax(y0[:n,:],axis=1)
        if is_same.any():
            working_exs.append(((x0, y0, loss, grad, x_adv, y1, loss_adv, grad_adv)))

    return working_exs







#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

input_dir = r"C:\Users\fulleed1\Documents\Adversarial Learning\AE_Detection_v2\NIPS_1000\Test"
output_dir = r"C:\Users\fulleed1\Documents\Adversarial Learning\AE_Detection_v2\NIPS_1000\Altered"
epsilon_l2 = 0.05
epsilon_linf = 0.4

tf.Graph().as_default()
sess = tf.Session()
model = InceptionV3(sess)
ae_examples = fgsm_attack(sess, model, epsilon_linf, input_dir, output_dir)
np.save(output_dir + 'AE1.npy', ae_examples)

ae_examples[1]
plt.imshow(x0.reshape(299,299,3))
plt.figure()
plt.imshow(x_adv.reshape(299,299,3))
np.linalg.norm((x0-x_adv).flatten(), np.inf)

(x0, loss, grad, x_adv, loss_adv, grad_adv) = ae_examples[3]
np.argmax(sess.run(model.output, feed_dict={model.x_tf : x_adv}))
np.argmax(sess.run(model.output, feed_dict={model.x_tf : x0}))


# examine n random perturbations around x of size r (l2 norm)
def check_perturb_effect(x, n, r=200):
    changed = []
    y = np.argmax(sess.run(model.output, feed_dict={model.x_tf: x}))
    for _ in range(n):
        direc = np.random.random_sample(x.shape)
        perturbed = x + n*direc / np.linalg.norm(direc)
        if (y!= np.argmax(sess.run(model.output, feed_dict={model.x_tf: perturbed}))).any():
            changed.append(perturbed)
            #print('Found one.')
    print(len(changed))
    return changed

first_attempt = check_perturb_effect(x0, 100)

adv_changed = check_perturb_effect(x_adv, 100)

plt.imshow(adv_changed[3].reshape(299,299,3))

for row in ae_examples:
    print('Regular:')
    check_perturb_effect(row[0], 100)
    print('Adversarial:')
    check_perturb_effect(row[3], 100)



import numpy as np
rand_list = [np.random.random(1000)-0.5 for _ in range(100)]
unit_rand = [x / np.linalg.norm(x) for x in rand_list]

y = unit_rand[0]

dotprods = []
for x in unit_rand[:49]:
    for y in unit_rand[50:]:
        dotprods.append(np.dot(x,y))

dotprods.sort()
np.max(dotprods)
np.mean(np.abs(dotprods))
plt.figure()
plt.plot(dotprods)

x = np.random.random(400)
x



from scipy.linalg import hadamard
hadamard(9)