import math
import cv2
import numpy as np
import tensorflow as tf
from random import randint


def get_image(image_path):
    return transform(imread(image_path))


def transform(image, npx=512, is_crop=True):
    cropped_image = cv2.resize(image, (256, 256))

    return np.array(cropped_image)


def imread(path):
    readimage = cv2.imread(path, 1)
    return readimage


def merge_color(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx / size[1]
        img[j * h:j * h + h, i * w:i * w + w] = image

    return img[:, :, 0]


def ims(name, img):
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


def sample_image_processing(data, batch_size):
    batch_size_sqrt = int(math.sqrt(batch_size))

    print data[0]
    base = np.array([get_image(sample_file) for sample_file in data[0:batch_size]])
    base_normalized = base / 255.0

    base_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255,
                                                cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2) for
                          ba in base]) / 255.0
    base_edge = np.expand_dims(base_edge, 3)

    base_colors = np.array([imageblur(ba) for ba in base]) / 255.0

    print("Sample Image processing: Result in results folder")
    ims("results/base.png", merge_color(base_normalized, [batch_size_sqrt, batch_size_sqrt]))
    ims("results/base_line.jpg", merge(base_edge, [batch_size_sqrt, batch_size_sqrt]))
    ims("results/base_colors.jpg", merge_color(base_colors, [batch_size_sqrt, batch_size_sqrt]))


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
