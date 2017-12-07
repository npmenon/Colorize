from __future__ import print_function

import os

from glob import glob
from tqdm import tqdm
from base import BaseModel
from utils import *
from img_utils import *


class AutoColor(BaseModel):
    def __init__(self, sess, config):
        super(AutoColor, self).__init__(config)

        # initialize session
        self.sess = sess

        # Build Graph for cGAN
        # create placeholders
        with tf.variable_scope('graph'):
            self.edge_image, self.noise, self.real_images = self.create_placeholders()

            # create input for gen and disc
            if self.use_gpu:
                final_input_image = tf.concat([self.edge_image, self.noise], 1)
            else:
                final_input_image = tf.concat([self.edge_image, self.noise], 3)

            # create generator and pass input image
            self.generated_images = self.build_generator(final_input_image, out_dimen=64)

            # create input for discriminator
            if self.use_gpu:
                self.positive_example = tf.concat([final_input_image, self.real_images], 1)
                self.negative_example = tf.concat([final_input_image, self.generated_images], 1)
            else:
                self.positive_example = tf.concat([final_input_image, self.real_images], 3)
                self.negative_example = tf.concat([final_input_image, self.generated_images], 3)

            # feed positive and negative examples to the discriminator
            self.positive_out, self.positive_logits = self.build_discriminator(self.positive_example, out_dimen=64)
            self.negative_out, self.negative_logits = self.build_discriminator(self.negative_example, out_dimen=64,
                                                                               reuse_variables=True)

            # Calculate loss for discriminator and generator
            self.dis_loss, self.gen_loss = self.calculate_loss()

        # Get all the training variables of generator and discriminator
        t_vars = tf.trainable_variables()
        self.disc_vars = [var for var in t_vars if 'dis_' in var.name]
        self.gen_vars = [var for var in t_vars if 'gen_' in var.name]

        # Optimize loss function using Adam Optimizer
        self.gen_optimizer, self.dis_optimizer = self.optimize()

        # to keep track of global step and logging purposes
        with tf.variable_scope('step'):
            self.step_op = tf.Variable(0, trainable=False, name='global_step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            self.step_assign_op = self.step_op.assign(self.step_input)

        # save global step
        tf.add_to_collection('global_step', self.step_op)

        # Build summary
        self.merged_summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.config.logdir, self.sess.graph)

        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

        # initialize Saver
        if self.config.is_train:
            self._saver = tf.train.Saver()
        else:
            self._saver = tf.train.Saver(self.gen_vars)

        # Load saved model
        self.load_model()

    def optimize(self):
        gen_optimizer = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.5).minimize(self.gen_loss,
                                                                                        var_list=self.gen_vars)
        dis_optimizer = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.5).minimize(self.dis_loss,
                                                                                        var_list=self.disc_vars)
        return gen_optimizer, dis_optimizer

    def create_placeholders(self):
        edge_images = None
        blurred_images = None
        real_images = None
        if self.cnn_format == 'NHWC':
            edge_images = tf.placeholder(tf.float32,
                                         [self.batch_size, self.input_image_size, self.input_image_size,
                                          self.input_colors])
            blurred_images = tf.placeholder(tf.float32,
                                            [self.batch_size, self.input_image_size, self.input_image_size,
                                             self.input_colors2])
            real_images = tf.placeholder(tf.float32,
                                         [self.batch_size, self.input_image_size, self.input_image_size,
                                          self.output_colors])

        elif self.cnn_format == 'NCHW':  # optimized for GPU usage
            edge_images = tf.placeholder(tf.float32,
                                         [self.batch_size, self.input_colors, self.input_image_size,
                                          self.input_image_size])
            blurred_images = tf.placeholder(tf.float32,
                                            [self.batch_size, self.input_colors2, self.input_image_size,
                                             self.input_image_size])
            real_images = tf.placeholder(tf.float32,
                                         [self.batch_size, self.output_colors, self.input_image_size,
                                          self.input_image_size])

        return edge_images, blurred_images, real_images

    # Method to calculate total loss
    def calculate_loss(self):
        with tf.variable_scope('loss'):
            # Discriminator Loss
            disc_loss_for_positive_examples = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.positive_logits,
                                                        labels=tf.ones_like(self.positive_logits)))

            disc_loss_for_negative_examples = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=self.negative_logits,
                                                        labels=tf.zeros_like(self.negative_logits)))

            # total discriminator loss
            total_disc_loss = disc_loss_for_positive_examples + disc_loss_for_negative_examples

            # Generator Loss
            gen_out_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.negative_logits,
                                                                                  labels=tf.ones_like(
                                                                                      self.negative_logits)))
            # L1 loss for geneator
            gen_L1_loss = self.lambda_scaling * tf.reduce_mean(tf.abs(self.real_images - self.generated_images))

            # Total Generator loss
            gen_loss = gen_out_loss + gen_L1_loss

        tf.summary.scalar("Disc_Loss", total_disc_loss)
        tf.summary.scalar("Gen_Loss", gen_loss)
        return total_disc_loss, gen_loss

    # Input is (256 x 256 x c_channels)
    def build_generator(self, in_image, out_dimen=64):
        # Building encoders

        with tf.variable_scope('gen_layer1'):
            enc_layer1 = conv2d(in_image, out_dimen, data_format=self.cnn_format)  # (128 x 128 x out_dimen)

        enc_layer2 = encoder_layer('gen_layer2', leaky_relu(enc_layer1), out_dimen * 2,
                                   self.cnn_format)  # is (64 x 64 x out_dimen * 2)
        enc_layer3 = encoder_layer('gen_layer3', leaky_relu(enc_layer2), out_dimen * 4,
                                   self.cnn_format)  # is (32 x 32 x out_dimen * 4)
        enc_layer4 = encoder_layer('gen_layer4', leaky_relu(enc_layer3), out_dimen * 8,
                                   self.cnn_format)  # is (16 x 16 x out_dimen * 8)
        enc_layer5 = encoder_layer('gen_layer5', leaky_relu(enc_layer4), out_dimen * 8,
                                   self.cnn_format)  # is (8 x 8 x self.gf_dim*8)

        out_image_size = self.output_image_size

        # Building decoders
        gpu_enabled = False
        if self.cnn_format == 'NCHW':
            gpu_enabled = True

        ###########################################################################################################

        # Decoder 1
        img_size = int(out_image_size / 16)
        out_shape = [self.batch_size, img_size, img_size, out_dimen * 8]
        if gpu_enabled:
            out_shape = [self.batch_size, out_dimen * 8, img_size, img_size]

        dec_layer1 = decoder_layer('gen_layer6', tf.nn.relu(enc_layer5), out_shape, self.cnn_format)

        # skip connection from enc_layer4
        if gpu_enabled:
            dec_layer1 = tf.concat([dec_layer1, enc_layer4], 1)
        else:
            dec_layer1 = tf.concat([dec_layer1, enc_layer4], 3)  # (16 x 16 x out_dimen * 8 * 2)

        ###########################################################################################################

        # Decoder 2
        img_size = int(out_image_size / 8)
        out_shape = [self.batch_size, img_size, img_size, out_dimen * 4]
        if gpu_enabled:
            out_shape = [self.batch_size, out_dimen * 4, img_size, img_size]

        dec_layer2 = decoder_layer('gen_layer7', tf.nn.relu(dec_layer1), out_shape, self.cnn_format)

        # skip connection from enc_layer3
        if gpu_enabled:
            dec_layer2 = tf.concat([dec_layer2, enc_layer3], 1)
        else:
            dec_layer2 = tf.concat([dec_layer2, enc_layer3], 3)  # (32 x 32 x out_dimen * 4 * 2)

        ###########################################################################################################

        # Decoder 3
        img_size = int(out_image_size / 4)
        out_shape = [self.batch_size, img_size, img_size, out_dimen * 2]
        if gpu_enabled:
            out_shape = [self.batch_size, out_dimen * 2, img_size, img_size]

        dec_layer3 = decoder_layer('gen_layer8', tf.nn.relu(dec_layer2), out_shape, self.cnn_format)

        # skip connection from enc_layer2
        if gpu_enabled:
            dec_layer3 = tf.concat([dec_layer3, enc_layer2], 1)
        else:
            dec_layer3 = tf.concat([dec_layer3, enc_layer2], 3)  # (64 x 64 x out_dimen * 2 * 2)

        ###########################################################################################################

        # Decoder 4
        img_size = int(out_image_size / 2)
        out_shape = [self.batch_size, img_size, img_size, out_dimen]
        if gpu_enabled:
            out_shape = [self.batch_size, out_dimen, img_size, img_size]

        dec_layer4 = decoder_layer('gen_layer9', tf.nn.relu(dec_layer3), out_shape, self.cnn_format)

        # skip connection from enc_layer1
        if gpu_enabled:
            dec_layer4 = tf.concat([dec_layer4, enc_layer1], 1)
        else:
            dec_layer4 = tf.concat([dec_layer4, enc_layer1], 3)  # (128 x 128 x out_dimen * 1 * 2)

        ###########################################################################################################

        # Decoder 5
        out_shape = [self.batch_size, out_image_size, out_image_size, self.output_colors]
        if gpu_enabled:
            out_shape = [self.batch_size, self.output_colors, out_image_size, out_image_size]

        dec_layer5 = decoder_layer('gen_layer10', tf.nn.relu(dec_layer4), out_shape, self.cnn_format)

        # (256 x 256 x 3 (RGB))
        return tf.nn.tanh(dec_layer5)

    def build_discriminator(self, input_image, out_dimen, reuse_variables=False):
        # Reuse already created variables from the previous step
        if reuse_variables:
            tf.get_variable_scope().reuse_variables()
        else:  # check whether reuse of variables is disabled
            assert tf.get_variable_scope().reuse == False

        with tf.variable_scope('dis_layer1'):
            l1 = conv2d(input_image, out_dimen, data_format=self.cnn_format)  # (128 x 128 x out_dimen)

        l2 = encoder_layer('dis_layer2', leaky_relu(l1), out_dimen * 2, self.cnn_format)  # (64 x 64 x out_dimen * 2)
        l3 = encoder_layer('dis_layer3', leaky_relu(l2), out_dimen * 4, self.cnn_format)  # (32 x 32 x out_dimen * 4)

        with tf.variable_scope('dis_layer4'):
            l4 = batch_norm(conv2d(leaky_relu(l3), out_dimen * 8, stride=[1, 1], data_format=self.cnn_format),
                            data_format=self.cnn_format)  # (16 x 16 x out_dimen * 8)

        with tf.variable_scope('dis_layer5'):
            l5 = linear(tf.reshape(leaky_relu(l4), [self.batch_size, -1]), 1, self.cnn_format)

        return tf.nn.sigmoid(l5), l5

    def train(self):
        # Todo: Check if this is giving correct start step
        self.start_step = tf.get_default_graph().get_tensor_by_name('step/global_step:0').eval()

        data = glob(os.path.join("imgs", "*.jpg"))

        # sample_image_processing(data, self.batch_size, self.sess, self.cnn_format)
        batch_files = data[0:self.batch_size]
        valid_norm, valid_edge, valid_noise = process_batch(batch_files, self.cnn_format)

        num_iter = len(data) // self.batch_size

        counter = 0
        for self.step in tqdm(range(self.start_step, self.max_step), ncols=70, initial=self.start_step):

            # save current step for future
            self.sess.run(self.step_assign_op, feed_dict={self.step_input: self.step})

            for iter in xrange(num_iter):
                # select next batch of images
                batch_files = data[iter * self.batch_size: (iter + 1) * self.batch_size]

                # process the batch of images
                batch_normalized, batch_edge, batch_noise = process_batch(batch_files, self.sess, self.cnn_format)

                dis_loss, _ = self.sess.run([self.dis_loss, self.dis_optimizer],
                                            feed_dict={self.edge_image: batch_edge, self.noise: batch_noise,
                                                       self.real_images: batch_normalized})
                gen_loss, _ = self.sess.run([self.gen_loss, self.gen_optimizer],
                                            feed_dict={self.edge_image: batch_edge, self.noise: batch_noise,
                                                       self.real_images: batch_normalized})
                summary = self.sess.run(self.merged_summary,
                                        feed_dict={self.edge_image: batch_edge, self.noise: batch_noise,
                                                   self.real_images: batch_normalized})

                counter += 1
                self.summary_writer.add_summary(summary, global_step=counter)

                print('Epoch: {0}, Batch: {1} of {2}, gen_loss: {3}, dis_loss: {4}'.format(self.step, iter, num_iter,
                                                                                           gen_loss, dis_loss))

                # save checkpoint
                if iter % 500 == 0:
                    self.save_model(self.step)

                # Test Run using Generator
                if iter % 100 == 0:
                    gen_test = self.sess.run(self.generated_images,
                                             feed_dict={self.edge_image: valid_edge, self.noise: valid_noise,
                                                        self.real_images: valid_norm})
                    store_image("train_out/" + str(self.step) + '_' + str(iter) + "_gen.jpg",
                                concat_img_color(gen_test, self.batch_size_sqrt, self.cnn_format), self.sess,
                                self.cnn_format)

    def test(self):
        data = glob(os.path.join("imgs", "*.jpg"))
        num_iter = self.test_size // self.batch_size

        for iter in xrange(num_iter):
            # select next batch of images
            batch_files = data[iter * self.batch_size: (iter + 1) * self.batch_size]

            # process the batch of images
            batch_normalized, batch_edge, batch_noise = process_batch(batch_files, self.sess, self.cnn_format)

            gen_test = self.sess.run(self.generated_images,
                                     feed_dict={self.edge_image: batch_edge, self.noise: batch_noise,
                                                self.real_images: batch_normalized})
            store_image("test_out/" + str(iter) + "_gen.jpg",
                        concat_img_color(gen_test, self.batch_size_sqrt, self.cnn_format), self.sess, self.cnn_format)
            store_image("test_out/" + str(iter) + "_real.jpg",
                        concat_img_color(batch_normalized, self.batch_size_sqrt, self.cnn_format), self.sess,
                        self.cnn_format)
            store_image("test_out/" + str(iter) + "_noise.jpg",
                        concat_img_color(batch_noise, self.batch_size_sqrt, self.cnn_format), self.sess,
                        self.cnn_format)
            store_image("test_out/" + str(iter) + "_edge.jpg",
                        concat_img(batch_edge, self.batch_size_sqrt, self.cnn_format), self.sess, self.cnn_format)
