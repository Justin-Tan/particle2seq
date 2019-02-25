#!/usr/bin/python3
# python3 train.py -i /home/jtan/gpu/jtan/spark/spark2tf/df_dnn_By_train_v0_train.parquet -test /home/jtan/gpu/jtan/spark/spark2tf/df_dnn_By_train_v0_val.parquet -arch dense -pq -n penalty_free
import tensorflow as tf
import numpy as np
import pandas as pd
import time, os, sys
import argparse

# User-defined
from network import Network
from utils import Utils
from data import Data
from mmd_model import Model
from config import config_train, directories

tf.logging.set_verbosity(tf.logging.ERROR)

def train(config, args):

    assert(config.use_adversary is False), 'To use adversarial training, run `adv_train.py`'
    start_time = time.time()
    global_step, n_checkpoints, v_auc_best = 0, 0, 0.
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)

    print('Reading data ...')
    if args.input is None:
        input_file = directories.train
        test_file = directories.test
    else:
        input_file = args.input
        test_file = args.test

    features, labels, pivots = Data.load_data(input_file, parquet=args.parquet)
    test_features, test_labels, test_pivots = Data.load_data(test_file, parquet=args.parquet)
    config.max_seq_len = int(features.shape[1]/config.features_per_particle)

    # Build graph
    model = Model(config, features=features, labels=labels, args=args)
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train_handle = sess.run(model.train_iterator.string_handle())
        test_handle = sess.run(model.test_iterator.string_handle())

        if args.restore_last and ckpt.model_checkpoint_path:
            # Continue training saved model
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('{} restored.'.format(ckpt.model_checkpoint_path))
            start_epoch = args.restart_epoch
        else:   
            if args.restore_path:
                new_saver = tf.train.import_meta_graph('{}.meta'.format(args.restore_path))
                new_saver.restore(sess, args.restore_path)
                print('{} restored.'.format(args.restore_path))
                start_epoch = args.restart_epoch
            else:
                start_epoch = 0
                
        sess.run(model.test_iterator.initializer, feed_dict={
            model.test_features_placeholder:test_features,
            model.test_pivots_placeholder:test_pivots,
            model.pivots_placeholder:test_pivots,
            model.test_labels_placeholder:test_labels})

        mutual_info_kraskov = list()

        for epoch in range(start_epoch, config.num_epochs):
            sess.run(model.train_iterator.initializer, feed_dict={model.features_placeholder:features, 
                model.labels_placeholder:labels, model.pivots_placeholder:pivots})

            # Run utils
            v_auc_best = Utils.run_diagnostics(model, config_train, directories, sess, saver, train_handle,
                test_handle, start_time, v_auc_best, epoch, global_step, args.name, args.fisher_penalty)

            if epoch > 0:
                save_path = saver.save(sess, os.path.join(directories.checkpoints, 'mmd2_{}_epoch{}_step{}.ckpt'.format(args.name, epoch, global_step)), global_step=epoch)
                print('Starting epoch {}, Weights saved to file: {}'.format(epoch, save_path))

            while True:
                try:
                    # Update weights
                    global_step, *ops = sess.run([model.global_step, model.opt_op, model.MINE_labels_train_op, 
                        model.auc_op, model.update_accuracy], feed_dict={model.training_phase: True, model.handle: train_handle})

                    if global_step % 500 == 0:
                        # Run utils
                        v_MI_kraskov = sess.run([model.MI_logits_theta_kraskov], feed_dict={model.training_phase: True, model.handle: train_handle})
                        v_auc_best, *_ = Utils.run_mmd_diagnostics(model, config_train, directories, sess, saver, train_handle,
                            test_handle, start_time, v_auc_best, epoch, global_step, args.name)
                        mutual_info_kraskov.append(v_MI_kraskov)


                    if global_step % 2500 == 0:
                        save_path = saver.save(sess, os.path.join(directories.checkpoints, 'mmd2_{}_epoch{}_step{}.ckpt'.format(args.name, epoch, global_step)), global_step=epoch)
                        print('Weights saved to file: {}'.format(save_path))

                except tf.errors.OutOfRangeError:
                    print('End of epoch!')
                    break

                except KeyboardInterrupt:
                    save_path = saver.save(sess, os.path.join(directories.checkpoints,
                        'mmd2_{}_last.ckpt'.format(args.name)), global_step=epoch)
                    mi_k = np.array(mutual_info_kraskov)
                    np.save('mi_kraskov_{}.npy'.format(args.name), mi_k)
                    print('Interrupted, model saved to: ', save_path)
                    sys.exit()

        save_path = saver.save(sess, os.path.join(directories.checkpoints,
                               'mmd2_{}_end.ckpt'.format(args.name)),
                               global_step=epoch)

    mi_k = np.array(mutual_info_kraskov)
    np.save('mi_kraskov_{}.npy'.format(args.name), mi_k)
    print("Training Complete. Model saved to file: {} Time elapsed: {:.3f} s".format(save_path, time.time()-start_time))

def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument("-r", "--restore_path", help="path to model to be restored", type=str)
    parser.add_argument("-i", "--input", default=None, help="Path to training file", type=str)
    parser.add_argument("-test", "--test", default=None, help="Path to test file", type=str)
    parser.add_argument("-opt", "--optimizer", default="adam", help="Selected optimizer", type=str)
    parser.add_argument("-n", "--name", default="mmd2", help="Checkpoint/Tensorboard label")
    parser.add_argument("-arch", "--architecture", default="deep_conv", help="Neural architecture",
        choices=set(('deep_conv', 'recurrent', 'simple_conv', 'conv_projection', 'dense')))
    parser.add_argument("-pq", "--parquet", help="Use if dataset is in parquet format", action="store_true")
    parser.add_argument("-mmd2", "--mmd2", help="Penalize magnitude of kernel MMD between pivots and logits", action="store_true")
    parser.add_argument("-lambda", "--mmd_lambda", default=0.0, help="Control tradeoff between xentropy and penalization", type=float)
    parser.add_argument("-re", "--restart_epoch", default=0, help="Epoch to restart from", type=int)

    args = parser.parse_args()
    config = config_train

    # Launch training
    train(config, args)

if __name__ == '__main__':
    main()
