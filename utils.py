import tensorflow as tf


# Weights
def new_weights(shape, stddev):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=stddev))


# Biases
def new_biases(length):
    return tf.Variable(tf.constant(0.1, shape=[length]))


# Convolutional layer
def conv(input, shape, stddev, is_training, stride):
    """
    :param stride: stride
    :param is_training: 是否训练
    :param input: 输入
    :param shape: 过滤器尺寸
    :param stddev: 初始化
    :return:
    """
    weights = new_weights(shape, stddev)
    biases = new_biases(shape[-1])
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, stride, stride, 1], padding='SAME')
    layer += biases
    layer = batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    return layer


def deconv(input, shape, stride, stddev):
    in_shape = tf.shape(input)
    output_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3] // 2])
    weights = new_weights(shape, stddev)
    return tf.nn.conv2d_transpose(input, filter=weights, output_shape=output_shape,
                                  strides=[1, stride, stride, 1], padding='SAME')


def max_pool(input, size):
    return tf.nn.max_pool2d(input, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='SAME')


def concat(a, b):
    return tf.concat([a, b], 3)


def batch_normalization(inputs, training):
    bn = tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        momentum=0.997,
        epsilon=1e-5,
        center=True,
        scale=True,
        training=training,
        fused=True)

    return bn
