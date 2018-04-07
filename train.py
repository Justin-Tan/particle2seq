#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
import time, os, sys
import argparse

# User-defined
from network import Network
from diagnostics import Diagnostics
from data import Data
from model import Model
from config import config_train, directories

tf.logging.set_verbosity(tf.logging.ERROR)

def train(config, args):

    start_time = time.time()
    global_step, n_checkpoints, v_f1_best = 0, 0, 0.
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)

    tokens, labels = Data.load_data(directories.train)
    test_tokens, test_labels = Data.load_data(directories.test)

    # Build graph
    cnn = Model(config, directories, tokens=tokens, labels=labels, args=args)
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
            cnn.test_tokens_placeholder:test_tokens,
            cnn.test_labels_placeholder:test_labels})

        for epoch in range(config.num_epochs):
            sess.run(cnn.train_iterator.initializer, feed_dict={cnn.tokens_placeholder:tokens, cnn.labels_placeholder:labels})

            # Run diagnostics
            v_f1_best = Diagnostics.run_diagnostics(cnn, config_train, directories, sess, saver, train_handle,
                test_handle, start_time, v_f1_best, epoch, args.name)
            while True:
                try:
                    # Update weights
                    sess.run([cnn.train_op, cnn.update_accuracy], feed_dict={cnn.training_phase: True,
                        cnn.handle: train_handle})

                except tf.errors.OutOfRangeError:
                    print('End of epoch!')
                    break

                except KeyboardInterrupt:
                    save_path = saver.save(sess, os.path.join(directories.checkpoints,
                        'bcmp_{}_last.ckpt'.format(args.name)), global_step=epoch)
                    print('Interrupted, model saved to: ', save_path)
                    sys.exit()

        save_path = saver.save(sess, os.path.join(directories.checkpoints,
                               'bcmp_{}_end.ckpt'.format(args.name)),
                               global_step=epoch)

    print("Training Complete. Model saved to file: {} Time elapsed: {:.3f} s".format(save_path, time.time()-start_time))

def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument("-r", "--restore_path", help="path to model to be restored", type=str)
    parser.add_argument("-opt", "--optimizer", default="adam", help="Selected optimizer", type=str)
    parser.add_argument("-n", "--name", default="text-clf", help="Checkpoint/Tensorboard label")
    args = parser.parse_args()
    config = config_train

    # Launch training
    train(config, args)

if __name__ == '__main__':
    main()
