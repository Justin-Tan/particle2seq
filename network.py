""" Network wiring """

import tensorflow as tf
import numpy as np
import glob, time, os
from diagnostics import Diagnostics

class Network(object):

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
    def birnn_dynamic(x, config, training, attention=False):

        print('Using recurrent architecture')
         # reshape outputs to [batch_size, max_time_steps, n_features]
        max_time = config.max_seq_len
        rnn_inputs = tf.reshape(x, [-1, max_time, config.embedding_dim])
        sequence_lengths = Diagnostics.length(rnn_inputs)
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

        if attention:  # invoke soft attention mechanism - attend to different particles
            summary_vector = attention(birnn_output, config.attention_dim, custom=False)
        else:  # Select last relevant output
            summary_vector = Diagnostics.last_relevant(birnn_output, sequence_lengths)
        
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
        sequence_lengths = Diagnostics.length(rnn_inputs)
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
            summary_vector = Diagnostics.soft_attention(birnn_output, config.attention_dim)
        else:  # Select last relevant output
            summary_vector = Diagnostics.last_relevant(birnn_output, sequence_lengths)
            print('Summarizing vector shape:', summary_vector.get_shape().as_list())

        # Fully connected layer for classification
        with tf.variable_scope("fc"):
            W_fc = tf.get_variable('W_fc', shape=[2*config.hidden_units, 2], initializer=init)
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

