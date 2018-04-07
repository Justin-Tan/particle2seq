#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import glob, time, os

from network import Network
from data import Data

class Model():
    def __init__(self, config, directories, tokens, labels, args, evaluate=False):
        # Build the computational graph

        arch = Network.sequence_deep_conv

        self.global_step = tf.Variable(0, trainable=False)
        self.handle = tf.placeholder(tf.string, shape=[])
        self.training_phase = tf.placeholder(tf.bool)
        self.rnn_keep_prob = tf.placeholder(tf.float32)

        self.tokens_placeholder = tf.placeholder(tf.int32, [tokens.shape[0], config.max_seq_len])
        self.labels_placeholder = tf.placeholder(tf.int32, labels.shape)
        self.test_tokens_placeholder = tf.placeholder(tf.int32)
        self.test_labels_placeholder = tf.placeholder(tf.int32)

        steps_per_epoch = int(self.tokens_placeholder.get_shape()[0])//config.batch_size

        train_dataset = Data.load_dataset(self.tokens_placeholder, self.labels_placeholder, config.batch_size)
        test_dataset = Data.load_dataset(self.test_tokens_placeholder, self.test_labels_placeholder, config.batch_size, test=True)
        self.iterator = tf.contrib.data.Iterator.from_string_handle(self.handle,
                                                                    train_dataset.output_types,
                                                                    train_dataset.output_shapes)

        self.train_iterator = train_dataset.make_initializable_iterator()
        self.test_iterator = test_dataset.make_initializable_iterator()

        embedding_encoder = tf.get_variable('embeddings', [config.vocab_size, config.embedding_dim])

        if evaluate:
            self.example = self.tokens_placeholder
            self.labels = self.labels_placeholder
            word_embeddings = tf.nn.embedding_lookup(embedding_encoder, ids=self.example)
            self.logits = arch(word_embeddings, config, self.training_phase)
            self.softmax, self.pred = tf.nn.softmax(self.logits), tf.argmax(self.logits, 1)
            self.ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.global_step)
            correct_prediction = tf.equal(self.labels, tf.cast(self.pred, tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return
        else:
            self.example, self.labels = self.iterator.get_next()

        word_embeddings = tf.nn.embedding_lookup(embedding_encoder, ids=self.example)

        self.logits = arch(word_embeddings, config, self.training_phase)
        self.softmax, self.pred = tf.nn.softmax(self.logits), tf.argmax(self.logits, 1)

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
        precision, self.update_precision = tf.metrics.precision(self.labels, self.pred)
        recall, self.update_recall = tf.metrics.recall(self.labels, self.pred)
        self.f1 = 2 * precision * recall / (precision + recall)
        self.precision = precision
        self.recall = recall
        correct_prediction = tf.equal(self.labels, tf.cast(self.pred, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('f1 score', self.f1)
        self.merge_op = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, '{}_train_{}'.format(args.name, time.strftime('%d-%m_%I:%M'))), graph=tf.get_default_graph())
        self.test_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, '{}_test_{}'.format(args.name, time.strftime('%d-%m_%I:%M'))))
