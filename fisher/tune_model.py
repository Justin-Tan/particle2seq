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
    def __init__(self, config, tune_config, features, labels, args, evaluate=False):
        """
        Build the computational graph
        tune_config: Dictionary of hyperparameters to be tuned
        """

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
            arch = functools.partial(Network.dense_network, num_features=features.shape[1]+len(config.pivots), actv=tf.nn.elu)
            sequential = False

        self.global_step = tf.Variable(0, trainable=False)
        self.MINE_step = tf.Variable(0, trainable=False)
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
        self.pivots.set_shape([None, 2*len(config.pivots)])

        if evaluate:
            # embeddings = tf.nn.embedding_lookup(embedding_encoder, ids=self.example)
            with tf.variable_scope('classifier') as scope:
                self.logits, *hreps = arch(self.example, config, self.training_phase)
            self.softmax = tf.nn.sigmoid(self.logits)
            self.pred = tf.cast(tf.greater(self.softmax, 0.5), tf.int32)
            self.ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.global_step)
            print('Y shape:', self.labels.shape)
            print('Logits shape:', self.logits.shape)
            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                labels=(1-tf.one_hot(self.labels, depth=1)))

            log_likelihood = -tf.reduce_sum(self.cross_entropy)

        # embeddings = tf.nn.embedding_lookup(embedding_encoder, ids=self.example)

        with tf.variable_scope('classifier') as scope:
            self.logits, self.hrep = arch(self.example, config, self.training_phase)

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
                
            self.adv_loss = adv.adversary_combined_loss
            self.total_loss = adv.total_loss
            self.predictor_train_op = adv.predictor_train_op
            self.adversary_train_op = adv.adversary_train_op

            self.joint_step, self.ema = adv.joint_step, adv.ema
            self.joint_train_op = adv.joint_train_op

            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                labels=(1-tf.one_hot(self.labels, depth=1)))
            self.cost = tf.reduce_mean(self.cross_entropy)

            self.MI_logits_theta_kraskov = tf.py_func(Utils.mutual_information_1D_kraskov, inp=[tf.squeeze(self.logits),
                tf.squeeze(self.pivots[:,0])], Tout=tf.float64)
            self.MI_xent_theta_kraskov = tf.py_func(Utils.mutual_information_1D_kraskov, inp=[tf.squeeze(self.cross_entropy),
                tf.squeeze(self.pivots[:,0])], Tout=tf.float64)

            self.MI_logits_labels_kraskov = tf.py_func(Utils.mutual_information_1D_kraskov, inp=[tf.squeeze(self.logits),
                tf.squeeze(self.labels)], Tout=tf.float64)

            X = self.logits
            # Calculate mutual information
            with tf.variable_scope('MINE') as scope:
                # X = tf.stop_gradient(self.logits)
                Z = tf.squeeze(self.pivots[:,0])
                # Z_prime = tf.random_shuffle(Z) # tf.squeeze(self.pivots[:,1])
                Z_prime = tf.random_shuffle(tf.squeeze(self.pivots[:,1]))

                *reg_terms, self.marginal_f, self.MI_logits_theta = Network.MINE(x=X, y=Z, y_prime=Z_prime, batch_size=config.batch_size, dimension=2, training=True, actv=tf.nn.elu)

            with tf.variable_scope('LABEL_MINE') as scope:
                Y = tf.cast(self.labels, tf.float32)
                Y_prime = tf.random_shuffle(Y)

                *reg_terms, self.MI_logits_labels_MINE = Network.MINE(x=X, y=Y, y_prime=Y_prime, batch_size=config.batch_size,
                        dimension=2, training=self.training_phase, actv=tf.nn.elu)


        else:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
            #    labels=self.labels)
            # self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
            #    labels=tf.one_hot(self.labels, depth=1))
            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                labels=(1-tf.one_hot(self.labels, depth=1)))
            self.cost = tf.reduce_mean(self.cross_entropy)

            X = self.logits
            X_bkg = tf.boolean_mask(X, tf.cast((1-self.labels), tf.bool))
            # Calculate mutual information
            with tf.variable_scope('MINE') as scope:
                # X = tf.stop_gradient(self.logits)
                Z = tf.squeeze(self.pivots[:,0])
                Z_bkg = tf.boolean_mask(Z, tf.cast((1-self.labels), tf.bool))
                # Z_prime = tf.random_shuffle(Z) # tf.squeeze(self.pivots[:,1])
                Z_prime = tf.random_shuffle(tf.squeeze(self.pivots[:,1]))
                Z_prime_bkg = tf.random_shuffle(Z_bkg) 

                #self.MI_logits_theta = Network.MINE(x=X, y=Z, y_prime=Z_prime, batch_size=config.batch_size,
                #        dimension=2, training=True, actv=tf.nn.elu, jensen_shannon=args.JSD)
                 
                (x_joint, x_marginal), (joint_f, marginal_f), self.MI_logits_theta = Network.MINE(x=X_bkg, y=Z_bkg, y_prime=Z_prime_bkg,
                        batch_size=config.batch_size, dimension=2, training=True, actv=tf.nn.elu, labels=self.labels, bkg_only=True, jensen_shannon=tune_config['jsd'])

            self.MI_logits_theta_kraskov = tf.py_func(Utils.mutual_information_1D_kraskov, inp=[tf.squeeze(self.logits),
                tf.squeeze(self.pivots[:,0])], Tout=tf.float64)
            self.MI_xent_theta_kraskov = tf.py_func(Utils.mutual_information_1D_kraskov, inp=[tf.squeeze(self.cross_entropy),
                tf.squeeze(self.pivots[:,0])], Tout=tf.float64)
            self.MI_logits_labels_kraskov = tf.py_func(Utils.mutual_information_1D_kraskov, inp=[tf.squeeze(self.logits),
                tf.squeeze(self.labels)], Tout=tf.float64)

            with tf.variable_scope('LABEL_MINE') as scope:
                Y = tf.cast(self.labels, tf.float32)
                Y_prime = tf.random_shuffle(Y)

                *reg_terms, self.MI_logits_labels_MINE = Network.MINE(x=X, y=Y, y_prime=Y_prime, batch_size=config.batch_size,
                        dimension=2, training=self.training_phase, actv=tf.nn.elu)

            theta_f = Utils.scope_variables('classifier')
            theta_MINE = Utils.scope_variables('MINE')
            theta_MINE_NY = Utils.scope_variables('LABEL_MINE')
            # print('Classifier parameters:', theta_f)
            # print('mine parameters', theta_MINE)
            # print('Label mine parameters', theta_MINE_NY)
            

            with tf.control_dependencies(update_ops):
                # Ensures that we execute the update_ops before performing the train_step
                self.MINE_lower_bound = self.MI_logits_theta
                self.MINE_labels_lower_bound = self.MI_logits_labels_MINE

                
            if args.mutual_information_penalty:
                print('Penalizing mutual information')
                    # 'minmax' loss
                    # heuristic 'non-saturating loss'
                    # heuristic_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=marginal_f, labels=tf.ones_like(marginal_f))) 
                    # heuristic_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=joint_f, labels=tf.zeros_like(joint_f)))
                    # self.cost += args.MI_lambda * heuristic_loss
                self.cost += tune_config['MI_lambda'] * tf.nn.relu(self.MINE_lower_bound)
                    # self.cost += args.MI_lambda * tf.square(self.MINE_lower_bound)
                # self.cost += -config.MI_label_lagrange_multiplier * tf.nn.relu(self.MI_logits_labels_MINE)

            with tf.control_dependencies(update_ops):

                args.optimizer = args.optimizer.lower()
                if args.optimizer=='adam':
                    self.opt = tf.train.AdamOptimizer(tune_config['learning_rate'])
                elif args.optimizer=='momentum':
                    self.opt = tf.train.MomentumOptimizer(config.learning_rate, config.momentum,
                        use_nesterov=True)
                elif args.optimizer == 'rmsprop':
                    self.opt = tf.train.RMSPropOptimizer(config_learning_rate)
                elif args.optimizer == 'sgd':
                    self.opt =  tf.train.GradientDescentOptimizer(config.learning_rate)

                self.grad_loss = tf.get_variable(name='grad_loss', shape=[], trainable=False)
                self.grads = self.opt.compute_gradients(-self.MI_logits_theta, grad_loss=self.grad_loss)

                self.opt_op = self.opt.minimize(self.cost, global_step=self.global_step, var_list=theta_f)
            
            # MINE_opt = tf.train.MomentumOptimizer(1e-4, momentum=0.9, use_nesterov=True)
            # MINE_opt = tf.train.GradientDescentOptimizer(1e-2)
            MINE_opt = tf.train.AdamOptimizer(tune_config['MINE_learning_rate'])

            self.MINE_opt_op = MINE_opt.minimize(-self.MINE_lower_bound, var_list=theta_MINE, global_step=self.MINE_step)

            self.MINE_labels_opt_op = MINE_opt.minimize(-self.MINE_labels_lower_bound, var_list=theta_MINE_NY)
            
            self.ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.global_step)
            self.MINE_ema = tf.train.ExponentialMovingAverage(decay=0.95, num_updates=self.MINE_step)
            maintain_averages_clf_op = self.ema.apply(theta_f)
            maintain_averages_MINE_op = self.MINE_ema.apply(theta_MINE)
            maintain_averages_MINE_labels_op = self.ema.apply(theta_MINE_NY)

            with tf.control_dependencies(update_ops+[self.opt_op]):
                self.train_op = tf.group(maintain_averages_clf_op)

            with tf.control_dependencies(update_ops+[self.MINE_opt_op]):
                self.MINE_train_op = tf.group(maintain_averages_MINE_op)

            with tf.control_dependencies(update_ops+[self.MINE_labels_opt_op]):
                self.MINE_labels_train_op = tf.group(maintain_averages_MINE_labels_op)

        self.str_accuracy, self.update_accuracy = tf.metrics.accuracy(self.labels, self.pred)
        correct_prediction = tf.equal(self.labels, tf.cast(self.pred, tf.int32))
        _, self.auc_op = tf.metrics.auc(predictions=self.pred, labels=self.labels, num_thresholds=2048)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('auc', self.auc_op)
        tf.summary.scalar('logits_theta_MI', self.MI_logits_theta_kraskov)
        tf.summary.scalar('xent_theta_MI', self.MI_xent_theta_kraskov)    
        tf.summary.scalar('logits_labels_MI', self.MI_logits_labels_kraskov)

        pivot = 'Mbc'
        tf.summary.histogram('true_{}_background_distribution'.format(pivot), true_background_pivots[:,0])
        tf.summary.histogram('pred_{}_background_distribution'.format(pivot), pred_background_pivots[:,0])

        self.merge_op = tf.summary.merge_all()

        self.train_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, '{}_train_{}'.format(args.name, time.strftime('%d-%m_%I:%M'))), graph=tf.get_default_graph())
        self.test_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, '{}_test_{}'.format(args.name, time.strftime('%d-%m_%I:%M'))))
