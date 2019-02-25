""" Network wiring """

import tensorflow as tf
import numpy as np
import glob, time, os
from utils import Utils
import functools

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

    @staticmethod
    def dense_network(x, config, training, name='fully_connected', actv=tf.nn.relu, **kwargs):
        # Toy dense network for binary classification
        
        init = tf.contrib.layers.xavier_initializer()
        shape = [512,512,512,512,512]
        kwargs = {'center': True, 'scale': True, 'training': training, 'fused': True, 'renorm': True}
        # x = tf.reshape(x, [-1, num_features])
        # x = x[:,:-1]
        print('Input X shape', x.get_shape())

        with tf.variable_scope(name, initializer=init, reuse=tf.AUTO_REUSE) as scope:
            h0 = tf.layers.dense(x, units=shape[0], activation=actv)
            h0 = tf.layers.batch_normalization(h0, **kwargs)

            h1 = tf.layers.dense(h0, units=shape[1], activation=actv)
            h1 = tf.layers.batch_normalization(h1, **kwargs)

            h2 = tf.layers.dense(h1, units=shape[2], activation=actv)
            h2 = tf.layers.batch_normalization(h2, **kwargs)

            h3 = tf.layers.dense(h2, units=shape[3], activation=actv)
            h3 = tf.layers.batch_normalization(h3, **kwargs)

            h4 = tf.layers.dense(h3, units=shape[3], activation=actv)
            h4 = tf.layers.batch_normalization(h4, **kwargs)

        out = tf.layers.dense(h4, units=1, kernel_initializer=init)
        
        return out, h4

    @staticmethod
    def MINE(x, y, y_prime, training, batch_size, name='MINE', actv=tf.nn.elu, dimension=2, labels=None, bkg_only=False, jensen_shannon=False):
        """
        Mutual Information Neural Estimator
        (x,y):      Drawn from joint
        y_prime:    Drawn from marginal

        returns
        MI:         Lower bound on mutual information between x,y
        """

        init = tf.contrib.layers.xavier_initializer()
        drop_rate = 0.0
        shape = [64,64,64,64,64,64]
        shape = [128,128,128,128,128]
        # shape = [512,512,512,512,512,512]
        shape = [128,128,64,32,16,8,4]  # or[256,256]
        shape = [128,128]
        kwargs = {'center': True, 'scale': True, 'training': training, 'fused': True, 'renorm': False}

        if bkg_only:
            batch_size = tf.cast(batch_size - tf.reduce_sum(labels), tf.int32)

        if dimension == 2:
            y, y_prime = tf.expand_dims(y, axis=1), tf.expand_dims(y_prime, axis=1)
        if len(x.get_shape().as_list()) < 2:
            x = tf.expand_dims(x, axis=1)
        # y_prime = tf.random_shuffle(y)

        z = tf.concat([x,y], axis=1)
        z_prime = tf.concat([x,y_prime], axis=1)
        z.set_shape([None, dimension])
        z_prime.set_shape([None, dimension])
        print('X SHAPE:', x.get_shape().as_list())
        print('Z SHAPE:', z.get_shape().as_list())
        print('Z PRIME SHAPE:', z_prime.get_shape().as_list())

        def statistic_network(t, name='MINE', reuse=False):
            with tf.variable_scope(name, initializer=init, reuse=reuse) as scope:

                # h0 = tf.layers.dense(t, units=shape[0], activation=actv)
                #h0 = tf.layers.dropout(h0, rate=drop_rate, training=training)
                #h0 = tf.layers.batch_normalization(h0, **kwargs)
                # h0 = tf.contrib.layers.layer_norm(h0, center=True, scale=True, activation_fn=None)
                h0 = tf.layers.dense(t, units=shape[0], activation=None)
                h0 = tf.contrib.layers.layer_norm(h0, center=True, scale=True, activation_fn=actv)

                # h1 = tf.layers.dense(h0, units=shape[1], activation=actv)
                #h1 = tf.layers.dropout(h1, rate=drop_rate, training=training)
                #h1 = tf.layers.batch_normalization(h1, **kwargs)
                # h1 = tf.contrib.layers.layer_norm(h1, center=True, scale=True, activation_fn=None)
                h1 = tf.layers.dense(h0, units=shape[1], activation=None)
                h1 = tf.contrib.layers.layer_norm(h1, center=True, scale=True, activation_fn=actv)

                #h2 = tf.layers.dense(h1, units=shape[2], activation=actv)
                #h2 = tf.layers.dropout(h2, rate=drop_rate, training=training)
                #h2 = tf.layers.batch_normalization(h2, **kwargs)

                #h3 = tf.layers.dense(h2, units=shape[3], activation=actv)
                #h3 = tf.layers.dropout(h3, rate=drop_rate, training=training)
                #h3 = tf.layers.batch_normalization(h3, **kwargs)

                out = tf.layers.dense(h1, units=1, kernel_initializer=init)

            return out

        def log_sum_exp_trick(x, batch_size, axis=1):
            # Compute along batch dimension
            x = tf.squeeze(x)
            x_max = tf.reduce_max(x)
            # lse = x_max + tf.log(tf.reduce_mean(tf.exp(x-x_max)))
            lse = x_max + tf.log(tf.reduce_sum(tf.exp(x-x_max))) - tf.log(batch_size)
            return lse

        joint_f = statistic_network(z)
        marginal_f = statistic_network(z_prime, reuse=True)
        print('Joint shape', joint_f.shape)
        print('marginal shape', marginal_f.shape)

        # MI_lower_bound = tf.reduce_mean(joint_f) - tf.log(tf.reduce_mean(tf.exp(marginal_f)) + 1e-5)
        MI_lower_bound = tf.squeeze(tf.reduce_mean(joint_f)) - tf.squeeze(log_sum_exp_trick(marginal_f,
            tf.cast(batch_size, tf.float32)))

        joint_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=joint_f,
            labels=tf.ones_like(joint_f)))
        marginal_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=marginal_f,
            labels=tf.zeros_like(marginal_f)))

        JSD_lower_bound = -(marginal_loss + joint_loss) + tf.log(4.0)
        # JSD_lower_bound = tf.reduce_mean(tf.log(tf.nn.sigmoid(joint_f))) + tf.reduce_mean(tf.log(1.0 -
        #    tf.nn.sigmoid(marginal_f)))

        # JSD_lower_bound = tf.squeeze(tf.reduce_mean(-tf.nn.softplus(-tf.squeeze(joint_f)))) - tf.squeeze(tf.reduce_mean(tf.nn.softplus(tf.squeeze(marginal_f))))

        if jensen_shannon:
            lower_bound = JSD_lower_bound
        else:
            lower_bound = MI_lower_bound

        return (z, z_prime), (joint_f, marginal_f), lower_bound


    @staticmethod
    def wasserstein_distance(x, y, y_prime, training, batch_size, name='wasserstein', actv=tf.nn.elu, dimension=2, 
            labels=None, bkg_only=True, gradient_penalty=True, lambda_gp=10.):
        """
        Input
            (x,y):      Drawn from joint
            y_prime:    Drawn from marginal
        Output
            WGAN loss:  Measure of distance between joint and product of marginals
        """

        init = tf.contrib.layers.xavier_initializer()
        shape = [128,128,64,32,16,8,4]  # or[256,256]
        shape = [128,128,128]
        kwargs = {'center': True, 'scale': True, 'training': training, 'fused': True, 'renorm': False}

        if bkg_only:
            batch_size = tf.cast(batch_size - tf.reduce_sum(labels), tf.int32)

        if dimension == 2:
            y, y_prime = tf.expand_dims(y, axis=1), tf.expand_dims(y_prime, axis=1)
        if len(x.get_shape().as_list()) < 2:
            x = tf.expand_dims(x, axis=1)
        # y_prime = tf.random_shuffle(y)

        z = tf.concat([x,y], axis=1)
        z_prime = tf.concat([x,y_prime], axis=1)
        #z.set_shape([None, dimension])
        #z_prime.set_shape([None, dimension])
        drop_rate = 0.0
        print('X SHAPE:', x.get_shape().as_list())
        print('Z SHAPE:', z.get_shape().as_list())
        print('Z PRIME SHAPE:', z_prime.get_shape().as_list())

        def statistic_network(t, name='f', reuse=False):
            with tf.variable_scope(name, initializer=init, reuse=reuse) as scope:
                # h0 = tf.layers.dense(t, units=shape[0], activation=actv)
                # h0 = tf.contrib.layers.layer_norm(h0, center=True, scale=True, activation_fn=None)
                h0 = tf.layers.dense(t, units=shape[0], activation=None)
                h0 = tf.contrib.layers.layer_norm(h0, center=True, scale=True, activation_fn=actv)

                # h1 = tf.layers.dense(h0, units=shape[1], activation=actv)
                #h1 = tf.layers.dropout(h1, rate=drop_rate, training=training)
                #h1 = tf.layers.batch_normalization(h1, **kwargs)
                # h1 = tf.contrib.layers.layer_norm(h1, center=True, scale=True, activation_fn=None)
                h1 = tf.layers.dense(h0, units=shape[1], activation=None)
                h1 = tf.contrib.layers.layer_norm(h1, center=True, scale=True, activation_fn=actv)

                #h2 = tf.layers.dense(h1, units=shape[2], activation=actv)
                #h2 = tf.layers.dropout(h2, rate=drop_rate, training=training)
                #h2 = tf.layers.batch_normalization(h2, **kwargs)

                out = tf.layers.dense(h1, units=1, kernel_initializer=init)

            return out

        joint_f = statistic_network(z)
        marginal_f = statistic_network(z_prime, reuse=True)
        print('Joint shape', joint_f.shape)
        print('marginal shape', marginal_f.shape)

        # IPM between joint and marginal - to be maximized
        # $ W(P,Q) = \sup_{f_L} \{ E_P[f] - E_Q[f]\} $
        wasserstein_metric = -tf.reduce_mean(joint_f) + tf.reduce_mean(marginal_f)
        disc_loss = -wasserstein_metric # -wasserstein_metric

        if gradient_penalty:
            # Enforce unit gradient norm as proxy for bounded Lipschitz constant
            uniform_dist = tf.contrib.distributions.Uniform(0.,1.)
            epsilon = uniform_dist.sample([batch_size, tf.shape(z)[1]])
            # Sample uniformly between linear interpolations
            z_interpolated = epsilon * z + (1. - epsilon) * z_prime
            interpolated_f = statistic_network(z_interpolated, reuse=True)
            grads = tf.gradients(interpolated_f, [z_interpolated])[0]
            grads_l2 = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=-1))
            gradient_penalty = tf.reduce_mean(tf.square(grads_l2-1.0))

            tf.summary.scalar("grad_penalty", gradient_penalty)
            tf.summary.scalar("grad_l2_norm", tf.nn.l2_loss(grads))

            disc_loss += lambda_gp * gradient_penalty

        return wasserstein_metric, disc_loss

    @staticmethod
    def kernel_MMD(x, y, y_prime, training, batch_size, name='kernel_MMD', actv=tf.nn.elu, dimension=2, labels=None, bkg_only=True):
        """
        Kernel MMD 
        (x,y):      Drawn from joint
        y_prime:    Drawn from marginal

        returns
        k_mmd:         MMD distance between two distributions
        """

        def gaussian_kernel_mmd2(X, Y, gamma):
            """
            Parameters
            ____
            X: Matrix, shape: (n_samples, features)
            Y: Matrix, shape: (m_samples, features)

            Returns
            ____
            mmd: MMD under Gaussian kernel
            """

            XX = tf.matmul(X, X, transpose_b=True)
            XY = tf.matmul(X, Y, transpose_b=True)
            YY = tf.matmul(Y, Y, transpose_b=True)

            M, N = tf.cast(XX.get_shape()[0], tf.float64), tf.cast(YY.get_shape()[0], tf.float64)

            X_sqnorm = tf.reduce_sum(X**2, axis=-1)
            Y_sqnorm = tf.reduce_sum(Y**2, axis=-1)

            row_bc = lambda x: tf.expand_dims(x,0)
            col_bc = lambda x: tf.expand_dims(x,1)

            K_XX = tf.exp( -gamma * (col_bc(X_sqnorm) - 2 * XX + row_bc(X_sqnorm)))
            K_XY = tf.exp( -gamma * (col_bc(X_sqnorm) - 2 * XY + row_bc(Y_sqnorm)))
            K_YY = tf.exp( -gamma * (col_bc(Y_sqnorm) - 2 * YY + row_bc(Y_sqnorm)))

            mmd2 = tf.reduce_sum(K_XX) / M**2 - 2 * tf.reduce_sum(K_XY) / (M*N) + tf.reduce_sum(K_YY) / N**2

            return mmd2

        def rbf_mixed_mmd2(X, Y, sigmas=[2.0, 5.0, 10.0, 20.0, 40.0, 80.0]):
            """
            Parameters
            ____
            X:      Matrix, shape: (n_samples, features)
            Y:      Matrix, shape: (m_samples, features)
            sigmas: RBF parameter

            Returns
            ____
            mmd2:   MMD under Gaussian mixed kernel
            """

            XX = tf.matmul(X, X, transpose_b=True)
            XY = tf.matmul(X, Y, transpose_b=True)
            YY = tf.matmul(Y, Y, transpose_b=True)

            M, N = tf.cast(XX.get_shape()[0], tf.float32), tf.cast(YY.get_shape()[0], tf.float32)

            X_sqnorm = tf.reduce_sum(X**2, axis=-1)
            Y_sqnorm = tf.reduce_sum(Y**2, axis=-1)

            row_bc = lambda x: tf.expand_dims(x,0)
            col_bc = lambda x: tf.expand_dims(x,1)

            K_XX, K_XY, K_YY = 0,0,0

            for sigma in sigmas:
                gamma = 1 / (2 * sigma**2)
                K_XX += tf.exp( -gamma * (col_bc(X_sqnorm) - 2 * XX + row_bc(X_sqnorm)))
                K_XY += tf.exp( -gamma * (col_bc(X_sqnorm) - 2 * XY + row_bc(Y_sqnorm)))
                K_YY += tf.exp( -gamma * (col_bc(Y_sqnorm) - 2 * YY + row_bc(Y_sqnorm)))

            mmd2 = tf.reduce_sum(K_XX) / M**2 - 2 * tf.reduce_sum(K_XY) / (M*N) + tf.reduce_sum(K_YY) / N**2

            return mmd2

        init = tf.contrib.layers.xavier_initializer()

        if bkg_only:
            batch_size = tf.cast(batch_size - tf.reduce_sum(labels), tf.int32)

        if dimension == 2:
            y, y_prime = tf.expand_dims(y, axis=1), tf.expand_dims(y_prime, axis=1)
        if len(x.get_shape().as_list()) < 2:
            x = tf.expand_dims(x, axis=1)

        z = tf.concat([x,y], axis=1)
        z_prime = tf.concat([x,y_prime], axis=1)
        z.set_shape([None, dimension])
        z_prime.set_shape([None, dimension])
        print('X SHAPE:', x.get_shape().as_list())
        print('Z SHAPE:', z.get_shape().as_list())
        print('Z PRIME SHAPE:', z_prime.get_shape().as_list())

        mmd2 = rbf_mixed_mmd2(z, z_prime)

        return mmd2
