
import tensorflow as tf
import cifar10

import pdb


class Cifar10:
  def __init__(self):
    # Note: it seems they crop the images in this model
    self.batch_shape = [128, 24, 24, 3]
    self.num_classes = 10
    #self._weights_file = './Weights/cifar10.ckpt'

    self.x_tf, self.y_tf = cifar10.inputs(eval_data='test')
    self.outputs = cifar10.inference(self.x_tf)

    self.loss = cifar10.loss(self.outputs, self.y_tf)
    self.loss_x = tf.gradients(self.loss, self.x_tf)[0]  # TODO: is this ok?


if __name__ == "__main__":
  model = Cifar10()
  pdb.set_trace() # TEMP
