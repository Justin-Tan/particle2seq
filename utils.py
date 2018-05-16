# -*- coding: utf-8 -*-
# Diagnostic helper functions for Tensorflow session
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import os, time
from sklearn.metrics import f1_score

class Utils(object):
    
    @staticmethod
    def soft_attention(inputs, attention_dim, feedforward=True):
        print('Using attention mechanism')
        hidden_units = inputs.shape[2].value  # D = dim of RNN hidden state
        init = tf.contrib.layers.xavier_initializer()

        W_ff = tf.get_variable('W_ff', shape=[hidden_units, attention_dim], initializer=init)
        b_ff = tf.get_variable('b_ff', shape=[attention_dim], initializer=init)
        u_query = tf.get_variable('context', shape=[attention_dim], initializer=init)

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_dim
            v = tf.tanh(tf.tensordot(inputs, W_ff, axes=1) + b_ff)

        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        energy = tf.tensordot(v, u_query, axes=1, name='attention_energy')  # (B,T) shape
        alphas = tf.nn.softmax(energy)         # (B,T) shape

        # Output: (B,D)
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas,-1), 1)

        return output

    @staticmethod
    def attention(summaries, attention_dim, feedforward=True, custom=True):

        # init = tf.random_normal_initializer(stddev=0.512)
        init = tf.contrib.layers.xavier_initializer()

        sequence_length = summaries.get_shape()[1].value
        hidden_units = summaries.get_shape()[2].value
        # Flatten to apply same weights at each time step
        A_re = tf.reshape(summaries, [-1, hidden_units])

        W_ff = tf.get_variable('W_ff', shape=[hidden_units, attention_dim])
        b_ff = tf.get_variable('b_ff', shape=[attention_dim])
        u_context = tf.get_variable('context', shape=[attention_dim], initializer=init)

        input_embedding = tf.tanh(tf.add(tf.matmul(A_re, W_ff), tf.reshape(b_ff, [1,-1])))
        energy = tf.matmul(input_embedding, tf.expand_dims(u_context,1))
        attention_energy = tf.reshape(energy, [-1, sequence_length])
        p = tf.nn.softmax(attention_energy)
        D = tf.matrix_diag(p)

        # Compute weighted sum of summaries
        if custom:
            output = tf.reduce_sum(tf.matmul(D, summaries), 1)
        else:
            output = tf.reduce_sum(summaries * tf.reshape(p, [-1, sequence_length, 1]), 1)

        return output

    @staticmethod
    def achtung(summaries, attention_dim, feedforward=True, custom=True):

        sequence_length = summaries.get_shape()[1].value
        hidden_units = summaries.get_shape()[2].value
        B_re = tf.reshape(summaries, [hidden_units, -1])

        W_ff = tf.get_variable('W_ff', shape = [attention_dim, hidden_units])
        b_ff = tf.get_variable('b_ff', shape = [attention_dim])
        u_context = tf.get_variable('context', shape = [attention_dim], initializer = tf.random_normal_initializer(stddev = 0.512))

        prod = tf.matmul(W_ff, B_re)
        b_ff_tiled = tf.tile(tf.expand_dims(b_ff,1), [1,prod.shape[1].value])

        input_embedding = tf.tanh(tf.add(prod, b_ff_tiled))
        energy = tf.matmul(tf.transpose(tf.expand_dims(u_context,1)), input_embedding)
        energy = tf.reshape(energy, [-1, sequence_length])

        p = tf.nn.softmax(energy)
        D = tf.matrix_diag(p)

        # Compute weighted sum of summaries
        if custom:
            output = tf.reduce_sum(tf.matmul(D, summaries), 1)
        else:
            output = tf.reduce_sum(summaries * tf.reshape(p, [-1, sequence_length, 1]), 1)

        return output

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

    @staticmethod
    def top_k_pool(x, k, axis, batch_size=None):
        # Input: tensor x with shape 'NHWC'
        
        # Swap axis-to-pool with last
        perm = np.arange(len(x.get_shape().as_list()))
        perm[-1], perm[axis] = axis, perm[-1]
        x = tf.transpose(x, perm)

        in_shape = tf.shape(x)
        last_dim = x.get_shape().as_list()[-1]
        x_re = tf.reshape(x, [-1,last_dim])

        values, indices = tf.nn.top_k(x_re, k=k, sorted=False)
        out = []
        vals = tf.unstack(values, axis=0)
        inds = tf.unstack(indices-(last_dim-k), axis=0)
        for i, idx in enumerate(inds):
            out.append(tf.sparse_tensor_to_dense(tf.SparseTensor(tf.reshape(tf.cast(idx,tf.int64),[-1,1]), vals[i], [k]), validate_indices=False))
        
        x_out = tf.stack(out)
        # shaped_out = tf.reshape(tf.stack(out), in_shape)
        # x_out = tf.transpose(x_out, perm)
        
        return x_out