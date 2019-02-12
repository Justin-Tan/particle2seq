#!/usr/bin/python3

# Script for adversarial training procedure
# See arXiv 1611.01046

import tensorflow as tf
import numpy as np
import pandas as pd
import time, os, sys
import argparse

# User-defined
from network import Network
from utils import Utils
from data import Data
from model import Model
from config import config_train, directories

tf.logging.set_verbosity(tf.logging.ERROR)

def train(config, args):
    
    config.use_adversary = True
    assert(config.use_adversary), 'use_adversary must be set to True in the configuration file'
    start_time = time.time()
    pretrain_step, joint_step, n_checkpoints, v_auc_best = 0, 0, 0, 0.
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)

    print('Reading data ...')
    features, labels, pivots, pivot_labels = Data.load_data(args.input, adversary=True, parquet=args.parquet)
    print('FSHAPE', features.shape)
    test_features, test_labels, test_pivots, test_pivot_labels = Data.load_data(args.test, adversary=True, parquet=args.parquet)
    config.max_seq_len = int(features.shape[1]/config.features_per_particle)

    # Build graph
    cnn = Model(config, features=features, labels=labels, args=args)
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train_handle = sess.run(cnn.train_iterator.string_handle())
        test_handle = sess.run(cnn.test_iterator.string_handle())

        if args.restore_last and ckpt.model_checkpoint_path:
            # Continue training saved model
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('{} restored.'.format(ckpt.model_checkpoint_path))
        else:   
            if args.restore_path:
                new_saver = tf.train.import_meta_graph('{}.meta'.format(args.restore_path))
                new_saver.restore(sess, args.restore_path)
                print('{} restored.'.format(args.restore_path))

        sess.run(cnn.test_iterator.initializer, feed_dict={
            cnn.test_features_placeholder:test_features,
            cnn.test_labels_placeholder:test_labels,
            cnn.test_pivots_placeholder:test_pivots,
            cnn.test_pivot_labels_placeholder:test_pivot_labels})

        mutual_info_kraskov = list()

        # Pretrain classifier
        if not args.skip_pretrain:
            print('Pretraining classifer for {} epochs'.format(config.n_epochs_initial))
            for epoch in range(config.n_epochs_initial):
                sess.run(cnn.train_iterator.initializer, feed_dict={
                    cnn.features_placeholder:features,
                    cnn.labels_placeholder:labels,
                    cnn.pivots_placeholder:pivots,
                    cnn.pivot_labels_placeholder:pivot_labels})

                # Run utils
                if epoch > 0:
                    save_path = saver.save(sess, os.path.join(directories.checkpoints,
                        'conv_{}_epoch{}_step{}.ckpt'.format(args.name, epoch, pretrain_step)), global_step=epoch)
                    print('Starting epoch {}, Weights saved to file: {}'.format(epoch, save_path))

                v_auc_best = Utils.run_diagnostics(cnn, config_train, directories, sess, saver, train_handle,
                    test_handle, start_time, v_auc_best, epoch, pretrain_step, args.name, args.fisher_penalty)
                train_feed = {cnn.training_phase: True, cnn.handle: train_handle}

                while True:
                    try:
                        pretrain_step += 1
                        # Update weights
                        sess.run([cnn.predictor_train_op, cnn.update_accuracy], 
                            feed_dict={cnn.training_phase: True, cnn.handle: train_handle})

                        if pretrain_step % 100 == 0:
                            # Run utils
                            v_MI_kraskov = sess.run([cnn.MI_logits_theta_kraskov],
                                    feed_dict={cnn.training_phase: True, cnn.handle: train_handle})
                            mutual_info_kraskov.append(v_MI_kraskov)

                        if pretrain_step % 1000 == 0:
                            save_path = saver.save(sess, os.path.join(directories.checkpoints, 'conv_{}_epoch{}_step{}.ckpt'.format(args.name, epoch, pretrain_step)), global_step=epoch)
                            print('Weights saved to file: {}'.format(save_path))
                        
                    except tf.errors.OutOfRangeError:
                        print('End of epoch!')
                        break

                    except KeyboardInterrupt:
                        save_path = saver.save(sess, os.path.join(directories.checkpoints,
                            'p2seq_{}_last.ckpt'.format(args.name)), global_step=epoch)
                        print('Interrupted, model saved to: ', save_path)
                        sys.exit()

            save_path = saver.save(sess, os.path.join(directories.checkpoints,
                                   'p2seq_{}_end.ckpt'.format(args.name)),
                                   global_step=epoch)

            print("Initial training Complete. Model saved to file: {} Time elapsed: {:.3f} s".format(save_path, time.time()-start_time))
        
        # Begin adversarial training
        print('<<<============================ Pretraining complete. Beginning adversarial training ============================>>>')
        for epoch in range(config.num_epochs):
            sess.run(cnn.train_iterator.initializer, feed_dict={
                cnn.features_placeholder:features,
                cnn.labels_placeholder:labels,
                cnn.pivots_placeholder:pivots,
                cnn.pivot_labels_placeholder:pivot_labels})

            if epoch % 10 == 0:
                # Run utils
                v_auc_best = Utils.run_adv_diagnostics(cnn, config_train, directories, sess, saver, train_handle,
                    test_handle, start_time, v_auc_best, epoch, joint_step, args.name)
                train_feed = {cnn.training_phase: True, cnn.handle: train_handle}

            while True:
                try:
                    # Train adversary for K iterations relative to predictive model
                    if joint_step % config.K == 0:
                        joint_step, *ops = sess.run([cnn.joint_step, cnn.joint_train_op, cnn.update_accuracy], train_feed)
                    else:
                        sess.run([cnn.adversary_train_op], train_feed)

                    if joint_step % 5000 == 0:  # Run diagnostics
                        v_MI_kraskov = sess.run([cnn.MI_logits_theta_kraskov],
                                feed_dict={cnn.training_phase: True, cnn.handle: train_handle})
                        v_auc_best = Utils.run_adv_diagnostics(cnn, config_train, directories, sess, saver, train_handle,
                            test_handle, start_time, v_auc_best, epoch, joint_step, args.name)
                        mutual_info_kraskov.append(v_MI_kraskov)

                except tf.errors.OutOfRangeError:
                    print('End of epoch!')
                    break

                except KeyboardInterrupt:
                    save_path = saver.save(sess, os.path.join(directories.checkpoints,
                        'p2seq_adv_{}_last.ckpt'.format(args.name)), global_step=epoch)
                    mi_k = np.array(mutual_info_kraskov)
                    np.save('mi_kraskov_{}.npy'.format(args.name), mi_k)
                    print('Interrupted, model saved to: ', save_path)
                    sys.exit()

        save_path = saver.save(sess, os.path.join(directories.checkpoints,
                               'p2seq_adv_{}_end.ckpt'.format(args.name)),
                               global_step=epoch)

    mi_k = np.array(mutual_info_kraskov)
    np.save('mi_kraskov_{}.npy'.format(args.name), mi_k)
    print("Training Complete. Model saved to file: {} Time elapsed: {:.3f} s".format(save_path, time.time()-start_time))

def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument("-r", "--restore_path", help="path to model to be restored", type=str)
    parser.add_argument("-opt", "--optimizer", default="adam", help="Selected optimizer", type=str)
    parser.add_argument("-n", "--name", default="p2seq_adv", help="Checkpoint/Tensorboard label")
    parser.add_argument("-arch", "--architecture", default="deep_conv", help="Neural architecture",
        choices=set(('deep_conv', 'recurrent', 'simple_conv', 'conv_projection', 'dense')))
    parser.add_argument("-i", "--input", default=None, help="Path to training file", type=str)
    parser.add_argument("-test", "--test", default=None, help="Path to test file", type=str)
    parser.add_argument("-pq", "--parquet", help="Use if dataset is in parquet format", action="store_true")
    parser.add_argument("-skip_pt", "--skip_pretrain", help="skip pretraining of classifier", action="store_true")
    parser.add_argument("-fisher", "--fisher_penalty", help="Penalize Fisher Information of pivots",
        action="store_true")
    parser.add_argument("-lambda", "--adv_lambda", default=0.0, help="Adversary-classification tradeoff control",
        type=float)

    args = parser.parse_args()
    config = config_train

    # Launch training
    train(config, args)

if __name__ == '__main__':
    main()
