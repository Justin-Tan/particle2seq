""" Adversarial training for robustness against systematic error """

import tensorflow as tf
import numpy as np
import glob, time, os
from utils import Utils
import functools
from config import directories

class Adversary(object):
    def __init__(self, config, classifier_logits, labels, pivots, pivot_labels, args, training_phase, evaluate=False):
        # Add ops to the graph for adversarial training
        
        adversary_losses_dict = {}
        adversary_logits_dict = {}

        for i, pivot in enumerate(config.pivots):
            # Introduce separate adversary for each pivotal variable
            mode = 'background'
            print('Building adversarial network for {} - {} events'.format(pivot, mode))

            with tf.variable_scope('adversary') as scope:
                adversary_logits = Utils.dense_network(
                    classifier_logits, 
                    layer=Utils.dense_layer,
                    n_layers=config.adv_n_layers,
                    hidden_nodes=config.adv_hidden_nodes,
                    keep_prob=config.adv_keep_prob,
                    training=training_phase,
                    n_input=2,
                    n_classes=config.adv_n_classes,
                    actv=tf.nn.elu,  # try tanh, relu, selu
                    scope='adversary_{}_{}'.format(pivot, mode))

            # Mask loss for signal events
            adversary_loss = tf.reduce_mean(tf.cast((1-labels), 
                tf.float32)*tf.nn.sparse_softmax_cross_entropy_with_logits(logits=adversary_logits,
                    labels=tf.cast(pivot_labels[:,i], tf.int32)))

            adversary_losses_dict[pivot] = adversary_loss
            adversary_logits_dict[pivot] = adversary_logits

            tf.add_to_collection('adversary_losses', adversary_loss)

        self.adversary_combined_loss = tf.add_n(tf.get_collection('adversary_losses'), name='total_adversary_loss')
        self.predictor_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=classifier_logits, labels=labels))
        self.total_loss = self.predictor_loss - config.adv_lambda*self.adversary_combined_loss
    
        theta_f = Utils.scope_variables('classifier')
        theta_r = Utils.scope_variables('adversary')
        print('Classifier parameters:', theta_f)
        print('Adversary parameters', theta_r)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            predictor_opt = tf.train.AdamOptimizer(config.learning_rate)
            self.joint_step = tf.Variable(0, name='predictor_global_step', trainable=False)
            self.predictor_train_op = predictor_opt.minimize(self.total_loss, name='predictor_opt', 
                global_step=joint_step, var_list=theta_f)
            # self.joint_train_op = predictor_optimizer.minimize(self.total_loss, name='joint_opt', 
            # global_step=predictor_gs, var_list=theta_f)

            adversary_opt = tf.train.AdamOptimizer(config.adv_learning_rate)
            adversary_gs = tf.Variable(0, name='adversary_global_step', trainable=False)
            self.adversary_train_op = adversary_opt.minimize(self.adversary_combined_loss, name='adversary_opt', 
                global_step=adversary_gs, var_list=theta_r)

        self.ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.joint_step, name='predictor_ema')
        maintain_predictor_averages_op = self.ema.apply(theta_f)
        with tf.control_dependencies([self.predictor_train_op]):
            self.joint_train_op = tf.group(maintain_predictor_averages_op)

        classifier_pred = tf.argmax(self.logits, 1)
        true_background = tf.boolean_mask(pivots, (1-labels))
        pred_background = tf.boolean_mask(pivots, (1-classifier_pred))

        tf.summary.scalar('adversary_loss', self.adversary_combined_loss)
        tf.summary.scalar('total_loss', self.total_loss)

        for i, pivot in enumerate(config.pivots):
            adv_correct_prediction = tf.equal(tf.cast(tf.argmax(adversary_logits_dict[pivot],1), tf.int32), 
                tf.cast(pivot_labels[:,i], tf.int32))
            adv_accuracy = tf.reduce_mean(tf.cast(adv_correct_prediction, tf.float32))
            tf.summary.scalar('adversary_acc_{}'.format(pivot), adv_accuracy)
            tf.summary.histogram('true_{}_background_distribution'.format(pivot), true_background[:,i])
            tf.summary.histogram('pred_{}_background_distribution'.format(pivot), pred_background[:,i])

