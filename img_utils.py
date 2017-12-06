import math
import cv2
import numpy as np
import tensorflow as tf
from random import randint


def get_image(image_path):
    image = cv2.imread(image_path, 1)
    cropped_image = cv2.resize(image, (256, 256))
    return np.array(cropped_image)


def concat_img_color(images, proportion, data_format):
    height = width = None
    if data_format == 'NHWC':
        height, width = images.shape[1], images.shape[2]
    elif data_format == 'NCHW':
        height, width = images.shape[2], images.shape[3]

    # create a holder for concatenated_images
    img = np.zeros((height * proportion, width * proportion, 3))

    for index, image in enumerate(images):
        i = index / proportion
        j = index % proportion

        img[i * height:i * height + height, j * width:j * width + width, :] = image

    return img


def concat_img(images, proportion, data_format):
    height = width = None
    if data_format == 'NHWC':
        height, width = images.shape[1], images.shape[2]
    elif data_format == 'NCHW':
        height, width = images.shape[2], images.shape[3]

    # create a holder for concatenated_images
    img = np.zeros((height * proportion, width * proportion, 1))

    for index, image in enumerate(images):
        i = index / proportion
        j = index % proportion

        img[i * height:i * height + height, j * width:j * width + width] = image

    return img[:, :, 0]


def store_image(name, img):
    print "saving img " + name
    cv2.imwrite(name, img * 255)


def imageblur(cimg, sampling=False):
    if sampling:
        cimg = cimg * 0.3 + np.ones_like(cimg) * 0.7 * 255
    else:
        for i in xrange(30):
            randx = randint(0, 205)
            randy = randint(0, 205)
            cimg[randx:randx + 50, randy:randy + 50] = 255
    return cv2.blur(cimg, (100, 100))


def sample_image_processing(data, batch_size, data_format):
    batch_size_sqrt = int(math.sqrt(batch_size))
    base = np.array([get_image(sample_file) for sample_file in data[0:batch_size]])
    base_normalized = base / 255.0

    base_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255,
                                                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2) for
                          ba in base]) / 255.0
    base_edge = np.expand_dims(base_edge, 3)

    base_colors = np.array([imageblur(ba) for ba in base]) / 255.0

    print("Sample Image processing: Result in output folder")
    store_image("output/base.png", concat_img_color(base_normalized, batch_size_sqrt, data_format))
    store_image("output/base_line.jpg", concat_img(base_edge, batch_size_sqrt, data_format))
    store_image("output/base_colors.jpg", concat_img_color(base_colors, batch_size_sqrt, data_format))


def process_batch(batch_files, data_format='NHWC'):
    batch = np.array([get_image(batch_file) for batch_file in batch_files])
    batch_normalized = batch / 255.0

    batch_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255,
                                                 cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9,
                                                 C=2) for ba in batch]) / 255.0
    batch_edge = np.expand_dims(batch_edge, 3)
    batch_colors = np.array([imageblur(ba) for ba in batch]) / 255.0

    if data_format == 'NCHW':
        batch_normalized = tf.transpose(batch_normalized, [0, 3, 1, 2])
        batch_edge = tf.transpose(batch_edge, [0, 3, 1, 2])
        batch_colors = tf.transpose(batch_colors, [0, 3, 1, 2])

    return batch_normalized, batch_edge, batch_colors
