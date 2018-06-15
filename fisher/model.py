#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import glob, time, os, functools

from network import Network
from data import Data
from config import config_test, directories
from adversary import Adversary
from utils import Utils

class Model():
    def __init__(self, config, features, labels, args, evaluate=False):
        # Build the computational graph

        arch = Network.sequence_deep_conv
        sequential = True 
        if args.architecture == 'deep_conv':
            arch = Network.sequence_deep_conv
        elif args.architecture == 'recurrent':
            arch = Network.birnn_dynamic
        elif args.architecture == 'simple_conv':
            arch = Network.sequence_conv2d
        elif args.architecture == 'conv_projection':
            arch = Network.conv_projection
        elif args.architecture == 'dense':
            arch = functools.partial(Network.dense_network, num_features=features.shape[1]+len(config.pivots))
            sequential = False

        self.global_step = tf.Variable(0, trainable=False)
        self.handle = tf.placeholder(tf.string, shape=[])
        self.training_phase = tf.placeholder(tf.bool)
        self.rnn_keep_prob = tf.placeholder(tf.float32)

        self.features_placeholder = tf.placeholder(tf.float32, [features.shape[0], features.shape[1]])
        self.labels_placeholder = tf.placeholder(tf.int32, labels.shape)
        self.test_features_placeholder = tf.placeholder(tf.float32)
        self.test_labels_placeholder = tf.placeholder(tf.int32)
        self.pivots_placeholder = tf.placeholder(tf.float32)
        self.test_pivots_placeholder = tf.placeholder(tf.float32)

        steps_per_epoch = int(self.features_placeholder.get_shape()[0])//config.batch_size

        if config.use_adversary and not evaluate:
            self.pivots_placeholder = tf.placeholder(tf.float32)
            self.pivot_labels_placeholder = tf.placeholder(tf.int32)
            self.test_pivots_placeholder = tf.placeholder(tf.float32)
            self.test_pivot_labels_placeholder = tf.placeholder(tf.int32)

            train_dataset = Data.load_dataset_adversary(self.features_placeholder, self.labels_placeholder,
                self.pivots_placeholder, self.pivot_labels_placeholder, config.batch_size, sequential=sequential)
            test_dataset = Data.load_dataset_adversary(self.test_features_placeholder, self.test_labels_placeholder,
                self.test_pivots_placeholder, self.test_pivot_labels_placeholder, config_test.batch_size, test=True,
                sequential=sequential)
        else:
            train_dataset = Data.load_dataset(self.features_placeholder, self.labels_placeholder,
                    self.pivots_placeholder, batch_size=config.batch_size, sequential=sequential)
            test_dataset = Data.load_dataset(self.test_features_placeholder, self.test_labels_placeholder,
                    self.pivots_placeholder, config_test.batch_size, test=True, sequential=sequential)


        val_dataset = Data.load_dataset(self.features_placeholder, self.labels_placeholder, self.pivots_placeholder,
                config.batch_size, evaluate=True, sequential=sequential)
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, train_dataset.output_types, train_dataset.output_shapes)

        self.train_iterator = train_dataset.make_initializable_iterator()
        self.test_iterator = test_dataset.make_initializable_iterator()
        self.val_iterator = val_dataset.make_initializable_iterator()

        # embedding_encoder = tf.get_variable('embeddings', [config.features_per_particle, config.embedding_dim])

        if config.use_adversary and not evaluate:
            self.example, self.labels, self.pivots, self.pivot_labels = self.iterator.get_next()
            if len(config.pivots) == 1:
                # self.pivots = tf.expand_dims(self.pivots, axis=1)
                self.pivot_labels = tf.expand_dims(self.pivot_labels, axis=1)
        else:
            self.example, self.labels, self.pivots = self.iterator.get_next()
        self.example.set_shape([None, features.shape[1]])

        # if len(config.pivots) == 1:
        #    self.pivots = tf.expand_dims(self.pivots, axis=1)
        self.pivots.set_shape([None, len(config.pivots)])
        # Fisher information check
        self.example = tf.concat([self.example, self.pivots], axis=1)
        print('Shape of combined:', self.example.get_shape().as_list())

        if evaluate:
            # embeddings = tf.nn.embedding_lookup(embedding_encoder, ids=self.example)
            with tf.variable_scope('classifier') as scope:
                self.logits = arch(self.example, config, self.training_phase)
            self.softmax = tf.nn.sigmoid(self.logits)
            self.pred = tf.cast(tf.greater(self.softmax, 0.5), tf.int32)
            self.ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.global_step)
            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                labels=(1-tf.one_hot(self.labels, depth=1)))

            log_likelihood = -tf.reduce_sum(self.cross_entropy)

            dldTheta = tf.gradients(log_likelihood, self.pivots)[0]
            self.observed_fisher_information = tf.square(tf.squeeze(dldTheta))
            return

        # embeddings = tf.nn.embedding_lookup(embedding_encoder, ids=self.example)

        with tf.variable_scope('classifier') as scope:
            self.logits = arch(self.example, config, self.training_phase)

        # self.softmax, self.pred = tf.nn.softmax(self.logits), tf.argmax(self.logits, 1)
        self.softmax = tf.nn.sigmoid(self.logits)[:,0]
        self.pred = tf.cast(tf.greater(self.softmax, 0.5), tf.int32)
        true_background_pivots = tf.boolean_mask(self.pivots, tf.cast((1-self.labels), tf.bool))
        pred_background_pivots = tf.boolean_mask(self.pivots, tf.cast((1-self.pred), tf.bool))


        epoch_bounds = [64, 128, 256, 420, 512, 720, 1024]
        lr_values = [1e-3, 4e-4, 1e-4, 6e-5, 1e-5, 6e-6, 1e-6, 2e-7]
        learning_rate = tf.train.piecewise_constant(self.global_step, boundaries=[s*steps_per_epoch for s in
            epoch_bounds], values=lr_values)

        if config.use_adversary:
            adv = Adversary(config,
                classifier_logits=self.logits,
                labels=self.labels,
                pivots=self.pivots,
                pivot_labels=self.pivot_labels,
                training_phase=self.training_phase,
                predictor_learning_rate=learning_rate,
                args=args)

            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                labels=(1-tf.one_hot(self.labels, depth=1)))
            self.cost = tf.reduce_mean(self.cross_entropy)
            log_likelihood = -tf.reduce_sum(self.cross_entropy)
            dldTheta = tf.gradients(log_likelihood, self.pivots)[0]
            self.observed_fisher_information = tf.reduce_mean(tf.square(tf.squeeze(dldTheta)))
            bkg_dldTheta = tf.boolean_mask(dldTheta, tf.cast((1-self.labels), tf.bool))
            self.observed_bkg_fisher_information = tf.reduce_mean(tf.square(tf.squeeze(bkg_dldTheta)))

            self.adv_loss = adv.adversary_combined_loss
            self.total_loss = adv.total_loss
            self.predictor_train_op = adv.predictor_train_op
            self.adversary_train_op = adv.adversary_train_op

            self.joint_step, self.ema = adv.joint_step, adv.ema
            self.joint_train_op = adv.joint_train_op

        else:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
            #    labels=self.labels)
            # self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
            #    labels=tf.one_hot(self.labels, depth=1))
            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                labels=(1-tf.one_hot(self.labels, depth=1)))
            self.cost = tf.reduce_mean(self.cross_entropy)

            # Calculate Fisher Information matrix
            bkg_xentropy = tf.boolean_mask(self.cross_entropy, tf.logical_not(tf.cast(self.labels, tf.bool)))
            log_likelihood = -tf.reduce_sum(self.cross_entropy)

            dldTheta = tf.gradients(log_likelihood, self.pivots)[0]
            self.observed_fisher_information = tf.reduce_mean(tf.square(tf.squeeze(dldTheta)))

            bkg_dldTheta = tf.boolean_mask(dldTheta, tf.cast((1-self.labels), tf.bool))
            self.observed_bkg_fisher_information = tf.reduce_mean(tf.square(tf.squeeze(bkg_dldTheta)))

            dfdTheta = tf.gradients(self.logits, self.pivots)[0]
            bkg_dfdTheta = tf.boolean_mask(dldTheta, tf.cast((1-self.labels), tf.bool))

            self.output_gradients = tf.reduce_mean(tf.square(tf.squeeze(dfdTheta)))
            bkg_output_gradients = tf.reduce_mean(tf.square(tf.squeeze(bkg_dfdTheta)))

            # Calculate mutual information
            self.MI_logits_theta = tf.py_func(Utils.mutual_information_1D_kraskov, inp=[tf.squeeze(self.logits),
                tf.squeeze(self.pivots[:,0])], Tout=tf.float64)
            self.MI_xent_theta = tf.py_func(Utils.mutual_information_1D_kraskov, inp=[tf.squeeze(self.cross_entropy),
                tf.squeeze(self.pivots[:,0])], Tout=tf.float64)

            if args.fisher_penalty:
                self.cost += config.fisher_penalty * self.observed_fisher_information
            
            if args.mutual_information_penalty:
                self.cost += config.MI_penalty * self.MI_logits_theta

            # self.cost += config.fisher_penalty * self.output_gradients

            # Alternatively, calculate the observed Fisher Information as the negative expected Hessian(ll)
            #hessian_ll = tf.hessians(log_likelihood, self.pivots)
            #FIM = -tf.squeeze(hessian_ll)
            #self.observed_fisher_diagonal_from_hessian = tf.diag_part(FIM)
            #self.observed_bkg_fisher_diagonal_from_hessian = tf.boolean_mask(self.observed_fisher_diagonal_from_hessian, 
            #    tf.cast((1-self.labels), tf.bool))
            #self.observed_fisher_information_from_hessian = tf.trace(FIM)
            #self.observed_bkg_fisher_information_from_hessian = tf.reduce_sum(self.observed_bkg_fisher_diagonal_from_hessian)

            
            with tf.control_dependencies(update_ops):
                # Ensures that we execute the update_ops before performing the train_step
                if args.optimizer=='adam':
                    self.opt_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.cost, global_step=self.global_step)
                elif args.optimizer=='momentum':
                    self.opt_op = tf.train.MomentumOptimizer(config.learning_rate, config.momentum,
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
        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('auc', self.auc_op)
        tf.summary.scalar('fisher_information', self.observed_fisher_information)
        tf.summary.scalar('bkg_fisher_information', self.observed_bkg_fisher_information)
        tf.summary.scalar('logits_theta_MI', self.MI_logits_theta)
        tf.summary.scalar('xent_theta_MI', self.MI_xent_theta)    

        pivot = 'Mbc'
        tf.summary.histogram('true_{}_background_distribution'.format(pivot), true_background_pivots[:,0])
        tf.summary.histogram('pred_{}_background_distribution'.format(pivot), pred_background_pivots[:,0])

        self.merge_op = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, '{}_train_{}'.format(args.name, time.strftime('%d-%m_%I:%M'))), graph=tf.get_default_graph())
        self.test_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, '{}_test_{}'.format(args.name, time.strftime('%d-%m_%I:%M'))))
