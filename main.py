import tensorflow as tf
from model import *


def configure():
    flags = tf.app.flags
    flags.DEFINE_integer('max_step', 100, 'How many steps to train')
    flags.DEFINE_float('rate', 1e-4, 'learning rate for training')
    flags.DEFINE_integer('reload_step', 0, 'Reload step to continue training')
    flags.DEFINE_integer('save_interval', 100, 'interval to save model')
    flags.DEFINE_integer('summary_interval', 5, 'interval to save summary')
    flags.DEFINE_integer('n_classes', 2, 'output class number')
    flags.DEFINE_integer('batch_size', 4, 'batch size for one iter')
    flags.DEFINE_boolean('is_training', True, 'training or predict (for batch normalization)')
    flags.DEFINE_string('datadir', './data/train/train.tfrecords', 'directory of data')
    flags.DEFINE_string('logdir', 'logs', 'directory to save logs of accuracy and loss')
    flags.DEFINE_string('modeldir', 'models', 'directory to save models ')
    flags.DEFINE_string('model_name', 'UNet', 'Model file name')

    flags.FLAGS.__dict__['__parsed'] = False
    return flags.FLAGS


if __name__ == '__main__':
    model = UNet(tf.Session(), configure())
    model.predicts()
