""" Adversarial training for robustness against systematic error """

import tensorflow as tf
import numpy as np
import glob, time, os
from utils import Utils
import functools
from config import directories

class Adversary(object):
    def __init__(self, config, classifier_logits, labels, auxillary_variables, args, training_phase, evaluate=False):
        # Add ops to the graph for adversarial training
        
        adversary_losses_dict = {}
        adversary_logits_dict = {}
        for i, pivot in enumerate(config.pivots):
            # Introduce separate adversary for each pivotal variable
            mode = 'background'
            print('Building adversarial network for {} - {}'.format(pivot, mode))

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
                    actv=tf.nn.elu,
                    scope='adversary_{}_{}'.format(pivot, mode))

            adversary_loss = tf.reduce_mean(tf.cast((1-labels), 
                tf.float32)*tf.nn.sparse_softmax_cross_entropy_with_logits(logits=adversary_logits,
                    labels=tf.cast(self.ancillary[:,i+2], tf.int32)))

            adversary_losses_dict[pivot] = adversary_loss
            adversary_logits_dict[pivot] = adversary_logits

            tf.add_to_collection('adversary_losses', adversary_loss)

        self.adversary_loss = tf.add_n(tf.get_collection('adversary_losses'), name='total_adversary_loss')
        self.predictor_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=classifier_logits, labels=labels))
        self.total_loss = self.predictor_loss - config.adv_lambda*self.adversary_loss
    
        theta_f = Utils.scope_variables('classifier')
        theta_r = Utils.scope_variables('adversary')
        self.theta = theta_f, theta_r

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            predictor_opt = tf.train.AdamOptimizer(config.learning_rate)
            predictor_gs = tf.Variable(0, name='predictor_global_step', trainable=False)
            predictor_optimize = predictor_optimizer.minimize(self.total_loss, name='predictor_opt', 
                global_step=predictor_gs, var_list=theta_f)
            # self.joint_train_op = predictor_optimizer.minimize(self.total_loss, name='joint_opt', 
            # global_step=predictor_gs, var_list=theta_f)

            adversary_opt = tf.train.AdamOptimizer(config.adv_learning_rate)
            adversary_gs = tf.Variable(0, name='adversary_global_step', trainable=False)
            self.adversary_train_op = adversary_opt.minimize(self.adversary_loss, name='adversary_opt', 
                global_step=adversary_gs, var_list=theta_r)

        self.ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=predictor_gs, name='predictor_ema')
        maintain_predictor_averages_op = self.ema.apply(theta_f)
        with tf.control_dependencies([predictor_optimize]):
            self.joint_train_op = tf.group(maintain_predictor_averages_op)

        tf.summary.scalar('adversary_loss', self.adversary_loss)
        tf.summary.scalar('total_loss', self.total_loss)
        for pivot in adversary_logits_dict.keys():
            adv_correct_prediction = tf.equal(tf.cast(tf.argmax(adversary_logits_dict[pivot],1), tf.int32), tf.cast(self.ancillary[:,3], tf.int32))
            adv_accuracy = tf.reduce_mean(tf.cast(adv_correct_prediction, tf.float32))
            tf.summary.scalar('adversary_acc_{}'.format(pivot), adv_accuracy)


        




    @staticmethod
    def conv_projection(x, config, training, reuse=False, actv=tf.nn.relu):
        print('Using convolutional architecture')
        init = tf.contrib.layers.xavier_initializer()
        kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':True}

        # reshape outputs to [batch_size, max_time_steps, config.embedding_dim, 1]
        max_time = config.max_seq_len  # tf.shape(x)[1]
        cnn_inputs = tf.expand_dims(tf.reshape(x, [-1, max_time, config.embedding_dim]), -1)

        # Convolution + max-pooling over n-particle windows
        filter_sizes = [3,4,5]
        n_filters = 128  # output dimensionality
        feature_maps = list()

        for filter_size in filter_sizes:
            # Each kernel extracts a specific n-gram of particles
            with tf.variable_scope('conv_proj2D-{}'.format(filter_size)) as scope:

                fs = [filter_size, config.embedding_dim, 1, n_filters]
                K = tf.get_variable('filter-{}'.format(filter_size), shape=fs, initializer=init)
                b = tf.get_variable('bias-{}'.format(filter_size), shape=[n_filters], initializer=tf.constant_initializer(0.01))
                W = tf.get_variable('proj-{}'.format(filter_size), shape=[max_time-filter_size+1, config.proj_dim], initializer=init)

                conv_i = tf.nn.conv2d(cnn_inputs, filter=K, strides=[1,1,1,1], padding='VALID')
                conv_i = actv(tf.nn.bias_add(conv_i, b))
                conv_i = tf.layers.batch_normalization(conv_i, **kwargs)
                print(conv_i.get_shape().as_list())

                # Project 2nd dimension into embedding space
                # [batch_size, J, 1, n_filters] (x) [J, embedding_dim] -> [batch_size, embedding_dim, 1, n_filters]
                proj_i = tf.einsum('ijkl,jm->imkl', conv_i, W)

                feature_maps.append(proj_i)

        # Combine feature maps
        print([fm.get_shape().as_list() for fm in feature_maps])
        convs = tf.concat(feature_maps, axis=1)
        print('before aggregated convolution:', convs.get_shape().as_list())

        agg_conv_filters = [256,128]
        convs = tf.layers.conv2d(convs, filters=agg_conv_filters[0], kernel_size=[3,1], kernel_initializer=init, activation=actv)
        convs = tf.layers.batch_normalization(convs, **kwargs)
        convs = tf.layers.conv2d(convs, filters=agg_conv_filters[1], kernel_size=[3,1], kernel_initializer=init, activation=actv)

        print('after aggregated convolution:', convs.get_shape().as_list())
        feature_vector = tf.contrib.layers.flatten(convs)
        feature_vector = tf.layers.dropout(feature_vector, rate=1-config.conv_keep_prob, training=training)

        # Fully connected layer for classification
        with tf.variable_scope("fc"):
            logits_CNN = tf.layers.dense(feature_vector, units=config.n_classes, kernel_initializer=init)

        return logits_CNN

    @staticmethod
    def birnn_dynamic(x, config, training):

        print('Using recurrent architecture')
         # reshape outputs to [batch_size, max_time_steps, n_features]
        max_time = config.max_seq_len
        rnn_inputs = tf.reshape(x, [-1, max_time, config.embedding_dim])
        sequence_lengths = Utils.length(rnn_inputs)
        init = tf.contrib.layers.xavier_initializer()

         # Choose rnn cell type
        if config.rnn_cell == 'lstm':
            base_cell = functools.partial(tf.nn.rnn_cell.LSTMCell, num_units=config.rnn_hidden_units)
        elif config.rnn_cell == 'gru':
            base_cell = functools.partial(tf.nn.rnn_cell.GRUCell, num_units=config.rnn_hidden_units)
        elif config.rnn_cell == 'layer_norm':
            base_cell = functools.partial(tf.contrib.rnn.LayerNormBasicLSTMCell, num_units=config.rnn_hidden_units,
                dropout_keep_prob=config.recurrent_keep_prob)
        else:
            raise Exception('Invalid RNN cell specified.')
     
        if config.add_skip_connections:
            print('Using skip connections in stacked rnn')
            skip_connection = lambda x: tf.nn.rnn_cell.ResidualWrapper(x)
        else:
            skip_connection = lambda x: x

        if (config.output_keep_prob * config.input_keep_prob < 1) and config.rnn_cell is not 'layer_norm':
            # rnn_inputs = tf.nn.dropout(rnn_inputs, self.keep_prob)
            fwd_cells = [tf.nn.rnn_cell.DropoutWrapper(
                skip_connection(base_cell()) if i != 0 else base_cell(),
                input_size=config.embedding_dim if i == 0 else config.rnn_hidden_units,
                input_keep_prob=config.input_keep_prob,
                output_keep_prob=config.output_keep_prob,
                variational_recurrent=True, dtype=tf.float32) for i in range(config.rnn_layers)]
            bwd_cells = [tf.nn.rnn_cell.DropoutWrapper(
                skip_connection(base_cell()) if i != 0 else base_cell(),
                input_size=config.embedding_dim if i == 0 else config.rnn_hidden_units,
                input_keep_prob=config.input_keep_prob,
                output_keep_prob=config.output_keep_prob,
                variational_recurrent=True, dtype=tf.float32) for i in range(config.rnn_layers)]
        else:
            fwd_cells = [base_cell() for _ in range(config.rnn_layers)]
            bwd_cells = [base_cell() for _ in range(config.rnn_layers)]

        fwd_cells = tf.contrib.rnn.MultiRNNCell(fwd_cells)
        bwd_cells = tf.contrib.rnn.MultiRNNCell(bwd_cells)
        
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fwd_cells,
            cell_bw=bwd_cells,
            inputs=rnn_inputs,
            sequence_length=sequence_lengths,
            dtype=tf.float32,
            parallel_iterations=128)

        birnn_output = tf.concat(outputs,2)

        if config.attention:  # invoke soft attention mechanism - attend to different particles
            summary_vector = Utils.soft_attention(birnn_output, config.attention_dim)
        else:  # Select last relevant output
            summary_vector = Utils.last_relevant(birnn_output, sequence_lengths)
        
        print(summary_vector.get_shape().as_list())
        # Fully connected layer for classification
        with tf.variable_scope("fc"):
            logits_RNN = tf.layers.dense(summary_vector, units=config.n_classes, kernel_initializer=init)
        
        return logits_RNN

    @staticmethod
    def birnn(x, config, training):

        print('Using recurrent architecture')
        # reshape outputs to [batch_size, max_time_steps, n_features]
        max_time = config.max_seq_len
        rnn_inputs = tf.reshape(x, [-1, max_time, config.embedding_dim])
        sequence_lengths = Utils.length(rnn_inputs)
        init = tf.contrib.layers.xavier_initializer()

         # Choose rnn cell type
        if config.rnn_cell == 'lstm':
            args = {'num_units': config.hidden_units, 'forget_bias': 1.0, 'state_is_tuple': True}
            base_cell = tf.nn.rnn_cell.LSTMCell
        elif config.rnn_cell == 'gru':
            args = {'num_units': config.hidden_units}
            base_cell = tf.nn.rnn_cell.GRUCell
        elif config.rnn_cell == 'layer_norm':
            args = {'num_units': config.hidden_units, 'forget_bias': 1.0, 'dropout_keep_prob': config.recurrent_keep_prob}
            base_cell = tf.contrib.rnn.LayerNormBasicLSTMCell
     
        cell = base_cell

        if config.output_keep_prob < 1:
            # rnn_inputs = tf.nn.dropout(rnn_inputs, self.keep_prob)
            fwd_cells = [tf.nn.rnn_cell.DropoutWrapper(
                cell(**args), 
                output_keep_prob=config.output_keep_prob,
                variational_recurrent=True, dtype=tf.float32) for _ in range(config.rnn_layers)]
            bwd_cells = [tf.nn.rnn_cell.DropoutWrapper(
                cell(**args),
                output_keep_prob=config.output_keep_prob,
                variational_recurrent=True, dtype=tf.float32) for _ in range(config.rnn_layers)]
        else:
            fwd_cells = [cell(**args) for _ in range(config.rnn_layers)]
            bwd_cells = [cell(**args) for _ in range(config.rnn_layers)]
        
        birnn_output, fwd_state, bwd_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=fwd_cells,
            cells_bw=bwd_cells,
            inputs=rnn_inputs,
            sequence_length=sequence_lengths,
            dtype=tf.float32,
            parallel_iterations=128)

        if config.attention:  # invoke soft attention mechanism - attend to different particles
            summary_vector = Utils.soft_attention(birnn_output, config.attention_dim)
        else:  # Select last relevant output
            summary_vector = Utils.last_relevant(birnn_output, sequence_lengths)
            print('Summarizing vector shape:', summary_vector.get_shape().as_list())

        # Fully connected layer for classification
        with tf.variable_scope("fc"):
            W_fc = tf.get_variable('W_fc', shape=[2*config.rnn_hidden_units, 2], initializer=init)
            b_fc = tf.get_variable('b_fc', shape=[2], initializer=tf.constant_initializer(0.01))
            logits_RNN = tf.matmul(summary_vector, W_fc) + b_fc

        # with tf.variable_scope("fc"):
        #    logits_RNN = tf.layers.dense(summary_vector, units=config.n_classes, kernel_initializer=init)
        
        return logits_RNN    

    @staticmethod
    def sequence_conv2d(x, config, training, reuse=False, actv=tf.nn.relu):
        init = tf.contrib.layers.xavier_initializer()
        kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':True}
        
        # reshape outputs to [batch_size, max_time_steps, config.embedding_dim, 1]
        # max_time = tf.shape(x)[1]
        max_time = config.max_seq_len
        cnn_inputs = tf.expand_dims(tf.reshape(x, [-1, max_time, config.embedding_dim]), -1)

        # Convolution + max-pooling over n-particle windows
        filter_sizes = [2,3,4,5]
        n_filters = 128  # output dimensionality
        feature_maps = list()

        for filter_size in filter_sizes:
            # Each kernel extracts a specific n-gram
            with tf.variable_scope('conv_pool2D-{}'.format(filter_size)) as scope:

                fs = [filter_size, config.embedding_dim, 1, n_filters]
                K = tf.get_variable('filter-{}'.format(filter_size), shape=fs, initializer=init)
                b = tf.get_variable('bias-{}'.format(filter_size), shape=[n_filters], initializer=tf.constant_initializer(0.01))
                conv_i = tf.nn.conv2d(cnn_inputs, filter=K, strides=[1,1,1,1], padding='VALID')
                conv_i = actv(tf.nn.bias_add(conv_i, b))
                conv_i = tf.layers.batch_normalization(conv_i, **kwargs)

                # Max over-time pooling - final size [batch_size, 1, 1, n_filters]
                # pool_i = tf.nn.max_pool(conv_i, ksize=[1,max_time-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID')
                pool_i = tf.nn.max_pool(conv_i, ksize=[1,max_time-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID')

                # conv_i = tf.layers.conv2d(cnn_inputs, filters=n_filters, kernel_size=[filter_size, config.embedding_dim], 
                #     padding='valid', use_bias=True, activation=actv, kernel_initializer=init)
                # pool_i = tf.layers.max_pooling2d(conv_i, pool_size=[max_time-filter_size+1], strides=[1,1,1,1], padding='valid')

                feature_maps.append(pool_i)
        
        # Combine feature maps
        print([fm.get_shape().as_list() for fm in feature_maps])
        total_filters = n_filters * len(filter_sizes)
        feature_vector = tf.concat(feature_maps, axis=3)
        feature_vector = tf.reshape(feature_vector, [-1, total_filters])
        feature_vector = tf.layers.dropout(feature_vector, rate=1-config.conv_keep_prob, training=training)

        # Fully connected layer for classification
        with tf.variable_scope("fc"):
            logits_CNN = tf.layers.dense(feature_vector, units=config.n_classes, kernel_initializer=init)
        
        return logits_CNN

    @staticmethod
    def sequence_deep_conv(x, config, training, reuse=False, actv=tf.nn.relu):
        print('Using convolutional architecture')
        init = tf.contrib.layers.xavier_initializer()
        kwargs = {'center':True, 'scale':True, 'training':training, 'fused':True, 'renorm':True}
        
        # reshape outputs to [batch_size, max_time_steps, config.embedding_dim, 1]
        # max_time = tf.shape(x)[1]
        max_time = config.max_seq_len
        cnn_inputs = tf.expand_dims(tf.reshape(x, [-1, max_time, config.embedding_dim]), -1)

        # Convolution + max-pooling over n-word windows
        filter_sizes = [3,4,5]
        n_filters = 128  # output dimensionality
        feature_maps = list()

        for filter_size in filter_sizes:
            # Each kernel extracts a specific n-gram of particles
            with tf.variable_scope('conv_pool2D-{}'.format(filter_size)) as scope:

                fs = [filter_size, config.embedding_dim, 1, n_filters]
                K = tf.get_variable('filter-{}'.format(filter_size), shape=fs, initializer=init)
                b = tf.get_variable('bias-{}'.format(filter_size), shape=[n_filters], initializer=tf.constant_initializer(0.01))
                conv_i = tf.nn.conv2d(cnn_inputs, filter=K, strides=[1,1,1,1], padding='VALID')
                conv_i = actv(tf.nn.bias_add(conv_i, b))
                conv_i = tf.layers.batch_normalization(conv_i, **kwargs)

                # Max over-time pooling - final size [batch_size, 1, 1, n_filters]
                # pool_i = tf.nn.max_pool(conv_i, ksize=[1,max_time-filter_size+1,1,1], strides=[1,1,1,1], padding='VALID')
                pool_i = tf.nn.max_pool(conv_i, ksize=[1,filter_size,1,1], strides=[1,1,1,1], padding='VALID')

                # conv_i = tf.layers.conv2d(cnn_inputs, filters=n_filters, kernel_size=[filter_size, config.embedding_dim], 
                #     padding='valid', use_bias=True, activation=actv, kernel_initializer=init)
                # pool_i = tf.layers.max_pooling2d(conv_i, pool_size=[max_time-filter_size+1], strides=[1,1,1,1], padding='valid')

                feature_maps.append(pool_i)
        
        # Combine feature maps
        print([fm.get_shape().as_list() for fm in feature_maps])
        convs = tf.concat(feature_maps, axis=1)
        print('before aggregated convolution:', convs.get_shape().as_list())

        agg_conv_filters = [256,128]
        convs = tf.layers.conv2d(convs, filters=agg_conv_filters[0], kernel_size=[3,1], kernel_initializer=init, activation=actv)
        convs = tf.layers.batch_normalization(convs, **kwargs)
        convs = tf.layers.conv2d(convs, filters=agg_conv_filters[1], kernel_size=[3,1], kernel_initializer=init, activation=actv)

        print('after aggregated convolution:', convs.get_shape().as_list())
        feature_vector = tf.contrib.layers.flatten(convs)
        feature_vector = tf.layers.dropout(feature_vector, rate=1-config.conv_keep_prob, training=training)

        # Fully connected layer for classification
        with tf.variable_scope("fc"):
            logits_CNN = tf.layers.dense(feature_vector, units=config.n_classes, kernel_initializer=init)
        
        return logits_CNN

