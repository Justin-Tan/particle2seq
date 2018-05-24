#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import glob, time, os, functools

from network import Network
from data import Data
from config import config_test, directories
from adversary import Adversary

class Model():
    def __init__(self, config, features, labels, args, evaluate=False):
        # Build the computational graph

        arch = Network.sequence_deep_conv
        if args.architecture == 'deep_conv':
            arch = Network.sequence_deep_conv
        elif args.architecture == 'recurrent':
            arch = Network.birnn_dynamic
        elif args.architecture == 'simple_conv':
            arch = Network.sequence_conv2d
        elif args.architecture == 'conv_projection':
            arch = Network.conv_projection

        self.global_step = tf.Variable(0, trainable=False)
        self.handle = tf.placeholder(tf.string, shape=[])
        self.training_phase = tf.placeholder(tf.bool)
        self.rnn_keep_prob = tf.placeholder(tf.float32)

        self.features_placeholder = tf.placeholder(tf.float32, [features.shape[0], features.shape[1]])
        self.labels_placeholder = tf.placeholder(tf.int32, labels.shape)
        self.test_features_placeholder = tf.placeholder(tf.float32)
        self.test_labels_placeholder = tf.placeholder(tf.int32)

        steps_per_epoch = int(self.features_placeholder.get_shape()[0])//config.batch_size

        if config.use_adversary:
            self.pivots_placeholder = tf.placeholder(tf.float32)
            self.pivot_labels_placeholder = tf.placeholder(tf.int32)
            self.test_pivots_placeholder = tf.placeholder(tf.float32)
            self.test_pivot_labels_placeholder = tf.placeholder(tf.int32)

            train_dataset = Data.load_dataset_adversary(self.features_placeholder, self.labels_placeholder, 
                self.pivots_placeholder, self.pivot_labels_placeholder, config.batch_size)
            test_dataset = Data.load_dataset_adversary(self.test_features_placeholder, self.test_labels_placeholder, 
                self.test_pivots_placeholder, self.test_pivot_labels_placeholder, config_test.batch_size, test=True)
        else:
            train_dataset = Data.load_dataset(self.features_placeholder, self.labels_placeholder, 
                config.batch_size)
            test_dataset = Data.load_dataset(self.test_features_placeholder, self.test_labels_placeholder, 
                config_test.batch_size, test=True)

        val_dataset = Data.load_dataset(self.features_placeholder, self.labels_placeholder, config.batch_size, evaluate=True)
        self.iterator = tf.contrib.data.Iterator.from_string_handle(self.handle,
                                                                    train_dataset.output_types,
                                                                    train_dataset.output_shapes)

        self.train_iterator = train_dataset.make_initializable_iterator()
        self.test_iterator = test_dataset.make_initializable_iterator()
        self.val_iterator = val_dataset.make_initializable_iterator()

        # embedding_encoder = tf.get_variable('embeddings', [config.features_per_particle, config.embedding_dim])

        self.example, *self.labels = self.iterator.get_next()

        if config.use_adversary:
            self.labels, self.pivots, self.pivot_labels, = self.labels
            print(self.pivots.get_shape())

        if evaluate:
            # embeddings = tf.nn.embedding_lookup(embedding_encoder, ids=self.example)
            self.logits = arch(self.example, config, self.training_phase)
            self.softmax, self.pred = tf.nn.softmax(self.logits)[:,1], tf.argmax(self.logits, 1)
            self.ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.global_step)
            return

        # embeddings = tf.nn.embedding_lookup(embedding_encoder, ids=self.example)

        with tf.variable_scope('classifier') as scope:
            self.logits = arch(self.example, config, self.training_phase)
        self.softmax, self.pred = tf.nn.softmax(self.logits), tf.argmax(self.logits, 1)


        if config.use_adversary:
            adv = Adversary(config,
                classifier_logits=self.logits,
                labels=self.labels,
                auxillary_variables=self.pivots,
                training_phase=self.training_phase,
                args=args)

            self.cross_entropy = adv.predictor_loss
            self.adv_loss = adv.adversary_combined_loss
            self.total_loss = adv.total_loss
            self.predictor_train_op = adv.predictor_train_op
            self.adversary_train_op = adv.adversary_train_op

            self.joint_step, self.ema = adv.joint_step, adv.ema
            self.joint_train_op = adv.joint_train_op
            
        else:

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                labels=self.labels)
            self.cost = tf.reduce_mean(self.cross_entropy)

            epoch_bounds = [64, 128, 256, 420, 512, 720, 1024]
            lr_values = [1e-3, 4e-4, 1e-4, 6e-5, 1e-5, 6e-6, 1e-6, 2e-7]

            learning_rate = tf.train.piecewise_constant(self.global_step, boundaries=[s*steps_per_epoch for s in
                epoch_bounds], values=lr_values)

            with tf.control_dependencies(update_ops):
                # Ensures that we execute the update_ops before performing the train_step
                if args.optimizer=='adam':
                    self.opt_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost, global_step=self.global_step)
                elif args.optimizer=='momentum':
                    self.opt_op = tf.train.MomentumOptimizer(learning_rate, config.momentum,
                        use_nesterov=True).minimize(self.cost, global_step=self.global_step)

            self.ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.global_step)
            maintain_averages_op = self.ema.apply(tf.trainable_variables())

            with tf.control_dependencies(update_ops+[self.opt_op]):
                self.train_op = tf.group(maintain_averages_op)

        self.str_accuracy, self.update_accuracy = tf.metrics.accuracy(self.labels, self.pred)
        correct_prediction = tf.equal(self.labels, tf.cast(self.pred, tf.int32))
        _, self.auc_op = tf.metrics.auc(predictions=self.pred, labels=self.labels, num_thresholds=1024)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('Dcost', self.cost)
        tf.summary.scalar('auc', self.auc_op)
        self.merge_op = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, '{}_train_{}'.format(args.name, time.strftime('%d-%m_%I:%M'))), graph=tf.get_default_graph())
        self.test_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, '{}_test_{}'.format(args.name, time.strftime('%d-%m_%I:%M'))))
