"""  Code for working with tensorflow / CNNs.

  Originally developed using tensorflow 1.2.1.
  Updated to 1.3.1 via
     conda install -c anaconda tensorflow-gpu

"""

import os

import numpy as np
from scipy.misc import imread, imsave


import tensorflow as tf
from tensorflow.contrib.slim.nets import inception, resnet_v2
slim = tf.contrib.slim


"""
Example of explicitly computing the hessian, from:
     https://groups.google.com/a/tensorflow.org/forum/#!msg/discuss/Xb_lwMIsNPc/sLk7aNcZCQAJ

It is probably better to use tensorflow.hessians() though...


def replace_none_with_zero(l):
  return [0 if i==None else i for i in l] 

tf.reset_default_graph()

x = tf.Variable(1.)
y = tf.Variable(1.)
loss = tf.square(x) + tf.square(y)
sess = create_session()
grads = tf.gradients([loss], [x, y])
hess0 = replace_none_with_zero(tf.gradients([grads[0]], [x, y]))
hess1 = replace_none_with_zero(tf.gradients([grads[1]], [x, y]))
hessian = tf.pack([tf.pack(hess0), tf.pack(hess1)])
print hessian.eval()
"""



#-------------------------------------------------------------------------------
# Helper functions for data I/O
#-------------------------------------------------------------------------------
def input_filenames(input_dir):
  all_files = tf.gfile.Glob(os.path.join(input_dir, '*.png'))
  all_files.sort()
  return all_files


def load_images(input_dir, batch_shape):
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

  for filepath in input_filenames(input_dir):
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


#-------------------------------------------------------------------------------
# Code for working with classification via CNNs. 
#-------------------------------------------------------------------------------

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

  NOTE: we assume a mini-batch size of 1! 
        This can be changed, but then there may be an issue with the hessian calculation...
  """

  def __init__(self, sess):
    self.batch_shape = [1, 299, 299, 3]  # for now, we attack one image at a time
    self._num_classes = 1001
    self._scope = 'InceptionV3'
    self._weights_file = './Weights/inception_v3.ckpt'

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
      self.loss_x = tf.gradients(self.loss, self.x_tf)[0] 

      #x_flat = tf.reshape(self.x_tf, shape=[-1])
      #self.loss_xx = tf.hessians(self.loss, x_flat) # not working at the moment - some issue with "none values not supported"

    #
    # load weights
    #
    saver = tf.train.Saver(slim.get_model_variables(scope=self._scope))
    saver.restore(sess, self._weights_file)


