#!/usr/bin/env python3

class config_train(object):
    mode = 'beta'
    n_layers = 5
    num_epochs = 512
    batch_size = 128
    ema_decay = 0.999
    learning_rate = 1e-4
    momentum = 0.9
    vocab_size = 1200
    rnn_layers = 2
    embedding_dim = 128
    rnn_cell = 'gru'
    hidden_units = 256
    output_keep_prob = 0.75
    max_seq_len = 15
    recurrent_keep_prob = 0.8
    conv_keep_prob = 0.5
    n_classes = 2
    features_per_particle = 13
    embedding_dim = 13

class config_test(object):
    mode = 'alpha'
    n_layers = 5
    num_epochs = 512
    batch_size = 256
    ema_decay = 0.999
    learning_rate = 1e-4
    momentum = 0.9
    vocab_size = 1200
    rnn_layers = 2
    embedding_dim = 128
    rnn_cell = 'gru'
    hidden_units = 256
    output_keep_prob = 0.75
    max_seq_len = 15
    recurrent_keep_prob = 1.0
    conv_keep_prob = 1.0
    n_classes = 2
    features_per_particle = 13
    embedding_dim = 13

class directories(object):
    train = 'data/small_train.h5'
    test = 'data/small_test.h5'
    eval = 'data/new_b2sy.h5'
    tensorboard = 'tensorboard'
    checkpoints = 'checkpoints'
    checkpoints_best = 'checkpoints/best'
