# -*- coding: utf-8 -*-
# Diagnostic helper functions for Tensorflow session
import tensorflow as tf
import os, time
from tensorflow.python.client import device_lib
from tfrecords import dl_cifar10, dl_cifar100

class Utils(object):

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
    def length(sequence):
	used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
	length = tf.reduce_sum(used, 1)
	length = tf.cast(length, tf.int32)
	return length

    @staticmethod
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        #return local_device_protos
        print('Available GPUs:')
        print([x.name for x in local_device_protos if x.device_type == 'GPU'])

class Diagnostics(object):
    
    @staticmethod
    def setup_dataset(name):
        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')
            os.mkdir('checkpoints/best')
        
        assert name in ['cifar10', 'cifar100'], 'Dataset name should be one of (cifar10, cifar100).'
        train_dataset = 'tfrecords/{}/{}_test.tfrecord'.format(name, name)
        test_dataset = 'tfrecords/{}/{}_test.tfrecord'.format(name, name)

        if not (os.path.isfile(train_dataset) and os.path.isfile(test_dataset)):
            if name=='cifar10':
                dl_cifar10.run('tfrecords/cifar10')
            elif name=='cifar100':
                dl_cifar100.run('tfrecords/cifar100')


    @staticmethod
    def run_diagnostics(model, config, directories, sess, saver, train_handle,
            test_handle, start_time, v_acc_best, epoch, name):
        t0 = time.time()
        improved = ''
        sess.run(tf.local_variables_initializer())
        feed_dict_train = {model.training_phase: False, model.handle: train_handle}
        feed_dict_test = {model.training_phase: False, model.handle: test_handle}

        try:
            t_acc, t_loss, t_summary = sess.run([model.accuracy, model.cost, model.merge_op], feed_dict=feed_dict_train)
            model.train_writer.add_summary(t_summary)
        except tf.errors.OutOfRangeError:
            t_loss, t_acc = float('nan'), float('nan')

        v_acc, v_loss, v_summary = sess.run([model.accuracy, model.cost, model.merge_op], feed_dict=feed_dict_test)
        model.test_writer.add_summary(v_summary)

        if v_acc > v_acc_best:
            v_acc_best = v_acc
            improved = '[*]'
            if epoch>5:
                save_path = saver.save(sess,
                            os.path.join(directories.checkpoints_best, 'crnn_{}_epoch{}.ckpt'.format(name, epoch)),
                            global_step=epoch)
                print('Graph saved to file: {}'.format(save_path))

        if epoch % 10 == 0 and epoch>10:
            save_path = saver.save(sess, os.path.join(directories.checkpoints, 'crnn_{}_epoch{}.ckpt'.format(name, epoch)), global_step=epoch)
            print('Graph saved to file: {}'.format(save_path))

        print('Epoch {} | Training Acc: {:.3f} | Test Acc: {:.3f} | Train Loss: {:.3f} | Test Loss: {:.3f} | Rate: {} examples/s ({:.2f} s) {}'.format(epoch, t_acc, v_acc, t_loss, v_loss, int(config.batch_size/(time.time()-t0)), time.time() - start_time, improved))

        return v_acc_best
