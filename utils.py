import tensorflow as tf


def conv2d(x,
           out_channel_dimen,
           kernel_size=None,
           stride=None,
           initializer=tf.truncated_normal_initializer(stddev=0.02),
           data_format='NHWC',
           padding='SAME',
           name='conv2d'):
    with tf.variable_scope(name):
        kernel_shape = []

        if kernel_size is None:
            kernel_size = [5, 5]

        if stride is None:
            stride = [2, 2]

        if data_format == 'NCHW':
            stride = [1, 1, stride[0], stride[1]]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], out_channel_dimen]

        elif data_format == 'NHWC':
            stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], out_channel_dimen]

        w = tf.get_variable('weight', kernel_shape, tf.float32, initializer=initializer)
        conv = tf.nn.conv2d(x, w, strides=stride, padding=padding, data_format=data_format)

        b = tf.get_variable('biases', [out_channel_dimen], initializer=tf.constant_initializer(0.0))
        bias_add_out = tf.nn.bias_add(conv, b, data_format=data_format)

        out = tf.reshape(bias_add_out, conv.get_shape())

    return out


def deconv2d(x,
             output_shape,
             kernel_size=None,
             stride=None,
             initializer=tf.random_normal_initializer(stddev=0.02),
             data_format='NHWC',
             name='deconv2d'):
    with tf.variable_scope(name):
        kernel_shape = []
        bias_out_shape = []

        if kernel_size is None:
            kernel_size = [5, 5]

        if stride is None:
            stride = [2, 2]

        if data_format == 'NCHW':
            stride = [1, 1, stride[0], stride[1]]
            kernel_shape = [kernel_size[0], kernel_size[1], output_shape[1], x.get_shape()[1]]
            bias_out_shape = [output_shape[1]]

        elif data_format == 'NHWC':
            stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], output_shape[-1], x.get_shape()[-1]]
            bias_out_shape = [output_shape[-1]]

        w = tf.get_variable('weight', kernel_shape, tf.float32, initializer=initializer)
        deconv = tf.nn.conv2d_transpose(x, w, strides=stride, output_shape=output_shape, data_format=data_format)

        b = tf.get_variable('biases', bias_out_shape, initializer=tf.constant_initializer(0.0))
        bias_add_out = tf.nn.bias_add(deconv, b, data_format=data_format)

        out = tf.reshape(bias_add_out, deconv.get_shape())

    return out


def batch_norm(input, data_format, scope='batch'):
    return tf.contrib.layers.batch_norm(input, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True,
                                        scope=scope, data_format=data_format)


def leaky_relu(x, alpha=0.2):
    return tf.maximum(x, alpha * x)


def encoder_layer(name, input, out_dimen, data_format='NHWC'):
    with tf.variable_scope(name):
        # convolve image and filter
        conv2d_output = conv2d(input, out_dimen, data_format=data_format)
        # batch normalize the output to reduce co-variate shift
        batch_norm_out = batch_norm(conv2d_output, data_format)
    return batch_norm_out


def decoder_layer(name, input, out_shape, data_format='NHWC'):
    with tf.variable_scope(name):
        # upsample image
        deconv2d_out = deconv2d(input, out_shape, data_format=data_format)
        # batch normalize the output to reduce co-variate shift
        batch_norm_out = batch_norm(deconv2d_out, data_format)
    return batch_norm_out


def linear(input, output_size, data_format='NHWC', stddev=0.02, bias_start=0.0, name='linear'):
    shape = input.get_shape().as_list()

    with tf.variable_scope(name):
        weight = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable('bias', [output_size],
                               initializer=tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(input, weight), bias)

        return out
