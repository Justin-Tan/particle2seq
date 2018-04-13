#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import time, os
import argparse
from sklearn.metrics import f1_score

# User-defined
from network import Network
from diagnostics import Diagnostics
from data import Data
from model import Model
from config import config_test, directories

tf.logging.set_verbosity(tf.logging.ERROR)

def evaluate(config, directories, ckpt, args):
    pin_cpu = tf.ConfigProto(allow_soft_placement=True, device_count = {'GPU':0})
    start = time.time()
    eval_df, eval_features, eval_labels = Data.load_data(directories.eval, evaluate=True)

    # Build graph
    cnn = Model(config, directories, features=eval_features, labels=eval_labels, args=args, evaluate=True)

    # Restore the moving average version of the learned variables for eval.
    variables_to_restore = cnn.ema.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with tf.Session(config=pin_cpu) as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        assert (ckpt.model_checkpoint_path), 'Missing checkpoint file!'

        if args.restore_last and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Most recent {} restored.'.format(ckpt.model_checkpoint_path))
        else:
            if args.restore_path:
                new_saver = tf.train.import_meta_graph('{}.meta'.format(args.restore_path))
                new_saver.restore(sess, args.restore_path)
                print('Previous checkpoint {} restored.'.format(args.restore_path))

        eval_dict = {cnn.training_phase: False, cnn.example: eval_features, cnn.labels: eval_labels}

        y_prob, y_pred, v_acc, v_auc = sess.run([cnn.softmax, cnn.pred, cnn.accuracy, cnn.auc_op], feed_dict=eval_dict)
        eval_df['y_pred'] = y_pred
        eval_df['y_prob'] = y_prob

        eval_df.to_hdf('df_sequence_val_{}.h5'.format(args.architecture), key='df')

        print("Validation accuracy: {:.3f}".format(v_acc))
        print("Validation AUC: {:.3f}".format(v_auc))
        print("Eval complete. Duration: %g s" %(time.time()-start))

        return v_acc


def main(**kwargs):
    parser = argparse.ArgumentParser()
#   parser.add_argument("-i", "--input", help="path to test dataset in h5 format")
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument("-r", "--restore_path", help="path to model to be restored", type=str)
    parser.add_argument("-arch", "--architecture", default="deep_conv", help="Neural architecture")
    args = parser.parse_args()

    # Load training, test data
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)

    # Evaluate
    val_accuracy = evaluate(config_test, directories, ckpt, args)

if __name__ == '__main__':
    main()
