#!/usr/bin/env python3

class config_train(object):
    mode = 'beta'
    n_layers = 5
    num_epochs = 512
    batch_size = 128
    ema_decay = 0.999
    learning_rate = 1e-4
    momentum = 0.9
    rnn_layers = 3
    embedding_dim = 128
    rnn_cell = 'gru'
    rnn_hidden_units = 128
    input_keep_prob = 0.75
    output_keep_prob = 0.75
    max_seq_len = 14
    conv_keep_prob = 0.5
    n_classes = 2
    features_per_particle = 14 # 23
    embedding_dim = features_per_particle  # temporary?
    attention = False
    attention_dim = 256
    proj_dim = 32
    add_skip_connections = True
    fisher_penalty = 10000

    # Adversary hyperparameters
    use_adversary = False
    # pivots = ['Mbc']  # or ['Mbc', 'deltaE']
    pivots = ['B_Mbc'] # , 'B_cms_p', 'B_cms_pt'] # 'q_sq', 'B_cms_q2Bh']
    # pivots += ['B_ell0_cms_E', 'B_ell0_cms_eRecoil', 'B_ell0_cms_m2Recoil', 'B_ell0_cms_p', 'B_ell0_cms_pt']
    # pivots += ['B_ell1_cms_E', 'B_ell1_cms_eRecoil', 'B_ell1_cms_m2Recoil', 'B_ell1_cms_p', 'B_ell1_cms_pt']
    adv_n_layers = 3
    adv_keep_prob = 1.0
    adv_hidden_nodes = [128,128,128]
    adv_learning_rate = 5e-3
    adv_lambda = 100
    adv_iterations = 1
    K = adv_iterations
    adv_n_classes = 10  # number of bins for discretized predictions
    n_epochs_initial = 4


class config_test(object):
    mode = 'alpha'
    n_layers = 5
    num_epochs = 512
    batch_size = 10000
    ema_decay = 0.999
    learning_rate = 1e-4
    momentum = 0.9
    rnn_layers = 3
    embedding_dim = 128
    rnn_cell = 'gru'
    rnn_hidden_units = 256
    output_keep_prob = 1.0
    max_seq_len = 14
    recurrent_keep_prob = 1.0
    conv_keep_prob = 1.0
    n_classes = 2
    features_per_particle = 14 # 23
    embedding_dim = features_per_particle  # temporary?
    attention = False
    attention_dim = 256
    proj_dim = 32

    use_adversary = False
    fisher_penalty = 1000
    #pivots = ['Mbc']  # or ['Mbc', 'deltaE']
    pivots = ['B_Mbc'] #, 'B_cms_p', 'B_cms_pt']#, 'q_sq', 'B_cms_q2Bh']
    # pivots += ['B_ell0_cms_E', 'B_ell0_cms_eRecoil', 'B_ell0_cms_m2Recoil', 'B_ell0_cms_p', 'B_ell0_cms_pt']
    # pivots += ['B_ell1_cms_E', 'B_ell1_cms_eRecoil', 'B_ell1_cms_m2Recoil', 'B_ell1_cms_p', 'B_ell1_cms_pt']
    adv_n_layers = 3
    adv_keep_prob = 1.0
    adv_hidden_nodes = [256,256,256]
    adv_learning_rate = 1e-3
    adv_lambda = 100
    adv_iterations = 1
    K = adv_iterations
    adv_n_classes = 10  # number of bins for discretized predictions
    n_epochs_initial = 4

class directories(object):
    train = '/data/projects/punim0011/jtan/data/b2sll_dense_train.h5'
    test = '/data/projects/punim0011/jtan/data/b2sll_dense_test.h5'
    val = '/data/projects/punim0011/jtan/data/b2sll_dense_val.h5'
    tensorboard = 'tensorboard'
    checkpoints = 'checkpoints'
    checkpoints_best = 'checkpoints/best'
    results = 'results'
