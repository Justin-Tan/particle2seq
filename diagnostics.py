# -*- coding: utf-8 -*-
# Diagnostic helper functions for Tensorflow session
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import os, time
from sklearn.metrics import f1_score

class Diagnostics(object):
    
    @staticmethod
    def length(sequence):
        used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.cast(length, tf.int32)
        return length

    @staticmethod
    def last_relevant(output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant

    @staticmethod
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        #return local_device_protos
        print('Available GPUs:')
        print([x.name for x in local_device_protos if x.device_type == 'GPU'])

    @staticmethod
    def run_diagnostics(model, config, directories, sess, saver, train_handle,
            test_handle, start_time, v_auc_best, epoch, name):
        t0 = time.time()
        improved = ''
        sess.run(tf.local_variables_initializer())
        feed_dict_train = {model.training_phase: False, model.handle: train_handle}
        feed_dict_test = {model.training_phase: False, model.handle: test_handle}

        try:
            t_auc, t_acc, t_loss, t_summary = sess.run([model.auc_op, model.accuracy, model.cost, model.merge_op], feed_dict=feed_dict_train)
            model.train_writer.add_summary(t_summary)
        except tf.errors.OutOfRangeError:
            t_auc, t_loss, t_acc = float('nan'), float('nan'), float('nan')

        v_auc, v_acc, v_loss, v_summary, y_true, y_pred = sess.run([model.auc_op, model.accuracy, model.cost, model.merge_op, model.labels, model.pred], feed_dict=feed_dict_test)
        model.test_writer.add_summary(v_summary)
        v_f1 = f1_score(y_true, y_pred, average='macro', labels=np.unique(y_pred))

        if v_auc > v_auc_best:
            v_auc_best = v_auc
            improved = '[*]'
            if epoch>5:
                save_path = saver.save(sess,
                            os.path.join(directories.checkpoints_best, 'conv_{}_epoch{}.ckpt'.format(name, epoch)),
                            global_step=epoch)
                print('Graph saved to file: {}'.format(save_path))

        if epoch % 10 == 0 and epoch>10:
            save_path = saver.save(sess, os.path.join(directories.checkpoints, 'conv_{}_epoch{}.ckpt'.format(name, epoch)), global_step=epoch)
            print('Graph saved to file: {}'.format(save_path))

        print('Epoch {} | Training Acc: {:.3f} | Test Acc: {:.3f} | Test auc: {:.3f} | Test F1: {:.3f} | Train Loss: {:.3f} | Test Loss: {:.3f} | Rate: {} examples/s ({:.2f} s) {}'.format(epoch, t_acc, v_acc, v_auc, v_f1, t_loss, v_loss, int(config.batch_size/(time.time()-t0)), time.time() - start_time, improved))

        return v_auc_best
