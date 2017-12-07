# https://github.com/npmenon/DQN-tensorflow/blob/master/config.py
import math


class GANConfig(object):
    # input image details
    input_image_size = 256
    batch_size = 4
    batch_size_sqrt = int(math.sqrt(batch_size))
    input_colors = 1
    input_colors2 = 3

    # output image details
    output_image_size = 256
    output_colors = 3

    lambda_scaling = 100

    max_step = 20000
    test_size = 200

    cnn_format = 'NCHW'

    backend = 'tf'
    logdir = './logs/'


def get_config(FLAGS):
    config = GANConfig

    for k, v in FLAGS.__dict__['__flags'].items():
        if not hasattr(config, k):
            setattr(config, k, v)

    return config
