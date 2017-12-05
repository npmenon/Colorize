import random
import tensorflow as tf

from deep_color import AutoColor
from config import get_config

flags = tf.app.flags

# Etc
flags.DEFINE_boolean('use_gpu', False, 'Whether to use gpu or not')
flags.DEFINE_string('gpu_fraction', '1/1', 'idx / # of gpu fraction e.g. 1/3, 2/3, 3/3')
flags.DEFINE_boolean('is_train', True, 'Whether to do training or testing')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

FLAGS = flags.FLAGS

# Set random seed
# Todo: Check if random is being used
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

if FLAGS.gpu_fraction == '':
    raise ValueError("--gpu_fraction should be defined")


def calc_gpu_fraction(fraction_string):
    idx, num = fraction_string.split('/')
    idx, num = float(idx), float(num)

    fraction = 1 / (num - idx + 1)
    print(" [*] GPU : %.4f" % fraction)
    return fraction


def main(_):
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.7)

    # gpu_options = tf.GPUOptions(
    #     per_process_gpu_memory_fraction=calc_gpu_fraction(FLAGS.gpu_fraction))

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        config = get_config(FLAGS) or FLAGS

        # test gpu
        if not tf.test.is_gpu_available() and FLAGS.use_gpu:
            raise Exception("use_gpu flag is true when no GPUs are available")

        if not FLAGS.use_gpu:
            config.cnn_format = 'NHWC'

        # create an instance of AutoColor
        auto_color = AutoColor(sess, config)

        # train or test
        if FLAGS.is_train:
            auto_color.train()
        else:
            auto_color.test()


if __name__ == '__main__':
    tf.app.run(main=main)
