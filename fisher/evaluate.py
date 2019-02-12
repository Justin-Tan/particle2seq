#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import time, os
import argparse
from sklearn.metrics import roc_auc_score

# User-defined
from network import Network
from utils import Utils
from data import Data
from model import Model
from config import config_test, directories

tf.logging.set_verbosity(tf.logging.ERROR)

def evaluate(config, args):

    start = time.time()
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)
    assert (ckpt.model_checkpoint_path), 'Missing checkpoint file!'
    
    print('Reading data...')
    eval_df, eval_features, eval_labels, eval_pivots = Data.load_data(args.input, evaluate=True,
            adversary=config.use_adversary, parquet=args.parquet)
    config.max_seq_len = int(eval_features.shape[1]/config.features_per_particle)

    # Build graph
    cnn = Model(config, features=eval_features, labels=eval_labels, args=args, evaluate=True)

    # Restore the moving average version of the learned variables for eval - rework this for adversary
    # variables_to_restore = cnn.ema.variables_to_restore()
    # saver = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        val_handle = sess.run(cnn.val_iterator.string_handle())

        if args.restore_last and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Most recent {} restored.'.format(ckpt.model_checkpoint_path))
        else:
            if args.restore_path:
                new_saver = tf.train.import_meta_graph('{}.meta'.format(args.restore_path))
                new_saver.restore(sess, args.restore_path)
                print('Previous checkpoint {} restored.'.format(args.restore_path))

        sess.run(cnn.val_iterator.initializer, feed_dict={cnn.features_placeholder: eval_features,
            cnn.labels_placeholder: eval_labels, cnn.pivots_placeholder: eval_pivots})

        labels, probs, preds, raw_output, xentropy, fisher, pivots = list(), list(), list(), list(), list(), list(), list()

        while True:
            try:
                eval_dict = {cnn.training_phase: False, cnn.handle: val_handle}
                y_true, y_prob, y_pred, logits, xent, z = sess.run([cnn.labels, cnn.softmax, cnn.pred,
                    cnn.logits, cnn.cross_entropy, cnn.pivots], feed_dict=eval_dict)
                probs.append(np.squeeze(y_prob))
                preds.append(np.squeeze(y_pred))
                labels.append(y_true)
                xentropy.append(np.squeeze(xent))
                raw_output.append(np.squeeze(logits))
                pivots.append(np.squeeze(z))
        #         fisher.append(observed_fisher)

            except tf.errors.OutOfRangeError:
                print('End of evaluation. Elapsed time: {:.2f} s'.format(time.time()-start))
                break

        y_prob = np.hstack(probs)
        y_pred = np.hstack(preds)
        y_true = np.hstack(labels)
        xentropy = np.hstack(xentropy)
        logits = np.hstack(raw_output)
        pivots = np.vstack(pivots)
        print('pivots', pivots.shape)
        # observed_fisher = np.hstack(fisher)
        max_length = y_prob.shape[0]
        eval_df = eval_df[:max_length]
        print('eval_df shape', eval_df.shape)

        eval_df['y_pred'] = y_pred
        eval_df['y_prob'] = y_prob
        eval_df['y_true'] = y_true
        eval_df['xentropy'] = xentropy
        eval_df['logits'] = logits
        eval_df['pivots'] = pivots[:,0]
        eval_df['pivots_marginal'] = pivots[:,1]
        # eval_df['fisher'] = observed_fisher

        v_acc = np.equal(y_true, y_pred).mean()
        v_auc = roc_auc_score(y_true, y_prob)

        out = args.out
        print('Running over {} with signal/background ratio {:.3f}'.format(out, y_true.mean()))
        h5_out = os.path.join(directories.results, '{}_{}_results.h5'.format(out, args.architecture))
        eval_df.to_hdf(h5_out, key='df', mode='w')
        print('Saved to', h5_out)
        # Utils.plot_ROC_curve(eval_df['y_true'], eval_df['y_prob'], 
        #        meta=r'$b \rightarrow s \gamma$' + ' Channel {}'.format(int(eval_df['B_ewp_channel'].head().mean())), out=out)

        print("Validation accuracy: {:.3f}".format(v_acc))
        print("Validation AUC: {:.3f}".format(v_auc))

        return v_acc


def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="path to evaluation dataset in h5 format")
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument("-r", "--restore_path", help="path to model to be restored", type=str)
    parser.add_argument("-arch", "--architecture", default="deep_conv", help="Neural architecture",
        choices=set(('deep_conv', 'recurrent', 'simple_conv', 'conv_projection', 'dense')))
    parser.add_argument("-pq", "--parquet", help="Use if dataset is in parquet format", action="store_true")
    parser.add_argument("-fisher", "--fisher_penalty", help="Penalize Fisher Information of pivots", action="store_true")
    parser.add_argument("-o", "--out", default='mein_model', help="Output filename")
    args = parser.parse_args()

    # Evaluate
    val_accuracy = evaluate(config_test, args)

if __name__ == '__main__':
    main()
