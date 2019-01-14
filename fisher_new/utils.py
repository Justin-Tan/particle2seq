# -*- coding: utf-8 -*-
# Diagnostic helper functions for Tensorflow session
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import os, time
import selu
# from sklearn.metrics import f1_score

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
    def selu_layer(x, shape, name, keep_prob, training=True):
        init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')

        with tf.variable_scope(name) as scope:
            W = tf.get_variable("weights", shape = shape, initializer=init)
            b = tf.get_variable("biases", shape = [shape[1]], initializer=tf.random_normal_initializer(stddev=0.01))
            actv = selu.selu(tf.add(tf.matmul(x, W), b))
            out = selu.dropout_selu(actv, rate=1-keep_prob, training=training)

        return out

    @staticmethod
    def dense_layer(x, shape, name, keep_prob, training=True, actv=tf.nn.elu):
        init=tf.contrib.layers.xavier_initializer()
        kwargs = {'center': True, 'scale': True, 'training': training, 'fused': True, 'renorm': True}

        with tf.variable_scope(name, initializer=init) as scope:
            layer = tf.layers.dense(x, units=shape[1], activation=actv)
            bn = tf.layers.batch_normalization(layer, **kwargs)
            out = tf.layers.dropout(bn, 1-keep_prob, training=training)

        return out

    @staticmethod
    def dense_network(x, n_layers, hidden_nodes, keep_prob, n_input, n_classes, scope, layer, actv=tf.nn.elu, reuse=False, training=True):

        SELU_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN')
        init = SELU_initializer if layer is Utils.selu_layer else tf.contrib.layers.xavier_initializer()
        assert n_layers == len(hidden_nodes), 'Specified layer nodes and number of layers do not correspond.'
        layers = [x]

        with tf.variable_scope(scope, reuse=reuse):
            hidden_0 = layer(x, shape=[n_input, hidden_nodes[0]], name='hidden0',
                                    keep_prob = keep_prob, training=training)
            layers.append(hidden_0)
            for n in range(0,n_layers-1):
                hidden_n = layer(layers[-1], shape=[hidden_nodes[n], hidden_nodes[n+1]], name='hidden{}'.format(n+1),
                                    keep_prob=keep_prob, training=training, actv=actv)
                layers.append(hidden_n)

            logits = tf.layers.dense(hidden_n, units=n_classes, kernel_initializer=init)

        return logits


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
    def scope_variables(name):
        with tf.variable_scope(name):
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)

    @staticmethod
    def plot_distributions(model, name, epoch, step, sess, handle, nbins=64, notebook=False, multiple_pivots=False):

        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        plt.style.use('seaborn-darkgrid')
        plt.style.use('seaborn-talk')
        feed_dict = {model.training_phase: False, model.handle: handle}

        pivots, labels, predictions, softmax = sess.run([model.pivots, model.labels, model.pred, model.softmax], feed_dict=feed_dict)
        assert pivots.shape[0] == labels.shape[0], 'Shape mismatch along batch dimension.'
        df_z = pd.DataFrame({'mbc': pivots[:,0]})# , 'l_pt': pivots[:,1]})
        df_z = df_z.assign(labels=labels, predictions=predictions, signal_prob=softmax) # softmax[:,1]

        def normPlot(variable, pdf, epoch, signal, nbins=42, bkg_rejection=0.99, plot_hist=True):
            titles={'mbc': r'$M_{bc}$ (GeV)', 'deltae': r'$\Delta E$ (GeV)', 'daughterInvM': r'$M_{X_q}$ (GeV)', 
                    'q_sq': r'$M_{\ell\ell}^2$ (GeV$^2$)', 'l_pt': r'$(\ell\ell)_{p_t}$'}
            bkg = pdf[pdf['labels']<0.5]
            post_bkg = bkg.nlargest(int(bkg.shape[0]*(1-bkg_rejection)), columns=['signal_prob'])
            threshold = post_bkg['signal_prob'].min()
            if signal:
                sig = pdf[pdf['labels']>0.5]
                post_sig = pdf[(pdf['labels']==1) & (pdf['predictions']>threshold)]
                print('Post-selection (signal):', post_sig.shape[0])
                if post_sig.shape[0] < 10:
                   pass 
                else:
                    sns.distplot(post_sig[variable], hist=plot_hist, kde=True, label='Signal post-cut', bins=nbins)
                    sns.distplot(sig[variable], hist=plot_hist, kde=True, label='Signal', bins=nbins)
            else:
                print('Post-selection (background):', post_bkg.shape[0])
                sns.distplot(post_bkg[variable], hist=plot_hist, kde=True, label='Background - {} rejection'.format(bkg_rejection), bins=nbins)
                sns.distplot(bkg[variable], hist=plot_hist, kde=True, label='Background, step {}'.format(step), bins=nbins)

            plt.xlabel(r'{}'.format(titles[variable]))
            plt.ylabel(r'Normalized events/bin')
            plt.legend(loc = "upper left")

            if variable == 'mbc':
                plt.xlim([5.22, 5.30])
                plt.ylim([0, 30])
            # elif variable == 'q_sq':
            #    plt.xlim([1.0,6.0])
            else:
                plt.autoscale(enable=True, axis='x', tight=False)

            if notebook:
                plt.show()
            else:
                disttype = 'signal' if signal else 'bkg'
                plt.savefig('graphs/{}_{}_dist_adv-ep{}-{}_{}.pdf'.format(variable, disttype, epoch, step, name), bbox_inches='tight',format='pdf', dpi=512)
                plt.savefig('graphs/{}_{}_dist_adv-ep{}-{}_{}.png'.format(variable, disttype, epoch, step, name),
                        bbox_inches='tight',format='png', dpi=128)
            plt.gcf().clear()

        def binscatter(variable, x, y, nbins=69):
            titles={'mbc': r'$M_{bc}$ (GeV)', 'deltae': r'$\Delta E$ (GeV)', 'daughterInvM': r'$M_{X_q}$ (GeV)'}
            n,_ = np.histogram(x, nbins)
            sy, _ = np.histogram(x, bins=nbins, weights=y)
            sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
            mean = sy / n
            std = np.sqrt(np.square(sy2/n - mean*mean))/4
            bins = (_[1:] + _[:-1])/2
            plt.errorbar(bins, mean, yerr=std, fmt='ro', label='Traditional NN', markersize=6, alpha=0.8)
            plt.xlabel(r'{}'.format(titles[variable]))
            plt.ylabel('NN Posterior')
            plt.legend(loc='best')
            # sns.regplot(bins,mean,order=1, marker='.',color='r')
            if notebook:
                plt.show()
            else:
                plt.savefig('graphs/{}_adv-ep{}-{}_scatter_{}.pdf'.format(variable, epoch, step, name),
                        bbox_inches='tight',format='pdf', dpi=512)
                plt.savefig('graphs/{}_adv-ep{}-{}_scatter_{}.pdf'.format(variable, epoch, step, name),
                        bbox_inches='tight',format='pdf', dpi=128)
            plt.gcf().clear()

        normPlot('mbc', df_z, epoch=epoch, signal=False, plot_hist=False)
        # normPlot('l_pt', df_z, epoch=epoch, signal=False, plot_hist=False)
        # normPlot('mbc', df_z, epoch=epoch, signal=True)

        # normPlot('deltae', df_z, epoch=epoch, signal=False)
        # df_sig = df_z[df_z['labels']>0.5]
        # df_bkg = df_z[df_z['labels']<0.5]
        # binscatter('mbc', df_bkg['mbc'], df_bkg['predictions'])
        # binscatter('deltae', df_bkg['deltae'], df_bkg['predictions'])

    @staticmethod
    def run_diagnostics(model, config, directories, sess, saver, train_handle,
            test_handle, start_time, v_auc_best, epoch, step, name):
        t0 = time.time()
        improved = ''
        sess.run(tf.local_variables_initializer())
        feed_dict_train = {model.training_phase: False, model.handle: train_handle}
        feed_dict_test = {model.training_phase: False, model.handle: test_handle}

        try:
            t_auc, t_acc, t_loss, t_summary = sess.run([model.auc_op, model.accuracy, model.cost, model.merge_op], 
                feed_dict=feed_dict_train)
            model.train_writer.add_summary(t_summary)
        except tf.errors.OutOfRangeError:
            t_auc, t_loss, t_acc = float('nan'), float('nan'), float('nan')

        v_FI, v_MI, v_auc, v_acc, v_loss, v_summary, y_true, y_pred = sess.run([model.observed_fisher_information, model.MI_logits_theta, model.auc_op, model.accuracy, model.cost, model.merge_op, model.labels, model.pred], feed_dict=feed_dict_test)
        model.test_writer.add_summary(v_summary)

        if v_auc > v_auc_best:
            v_auc_best = v_auc
            improved = '[*]'
            if epoch>5:
                save_path = saver.save(sess,
                            os.path.join(directories.checkpoints_best, 'conv_{}_epoch{}.ckpt'.format(name, epoch)),
                            global_step=epoch)
                print('Weights saved to file: {}'.format(save_path))

        # if epoch % 10 == 0 and epoch>10:
        if step % 10000 == 0:
            save_path = saver.save(sess, os.path.join(directories.checkpoints, 'conv_{}_epoch{}.ckpt'.format(name, epoch)), global_step=epoch)
            print('Weights saved to file: {}'.format(save_path))

        print('Epoch {} | Training Acc: {:.3f} | Test Acc: {:.3f} | Test auc: {:.3f} | Fisher Info: {:.3f} | MI: {:.3f} | Train Loss: {:.3f} | Test Loss: {:.3f} | Rate: {} examples/s ({:.2f} s) {}'.format(epoch, t_acc, v_acc, v_auc, v_FI, v_MI, t_loss, v_loss, int(config.batch_size/(time.time()-t0)), time.time() - start_time, improved))
        Utils.plot_distributions(model, name, epoch, step, sess, handle=test_handle)

        return v_auc_best

    @staticmethod
    def run_adv_diagnostics(model, config, directories, sess, saver, train_handle,
            test_handle, start_time, v_auc_best, epoch, step, name):
        t0 = time.time()
        improved = ''
        sess.run(tf.local_variables_initializer())
        feed_dict_train = {model.training_phase: False, model.handle: train_handle}
        feed_dict_test = {model.training_phase: False, model.handle: test_handle}

        t_acc, t_loss, t_auc, t_summary = sess.run([model.accuracy, model.cost, model.auc_op, model.merge_op],
                                            feed_dict = feed_dict_train)
        v_ops = [model.accuracy, model.cost, model.adv_loss, model.auc_op, model.total_loss, model.merge_op]
        v_acc, v_loss, v_adv_loss, v_auc, v_total, v_summary = sess.run(v_ops, feed_dict=feed_dict_test)
        model.train_writer.add_summary(t_summary)
        model.test_writer.add_summary(v_summary)

        if v_auc > v_auc_best:
            v_auc_best = v_auc
            improved = '[*]'
            if epoch>0:
                save_path = saver.save(sess,
                            os.path.join(directories.checkpoints_best, 'conv_{}_epoch{}.ckpt'.format(name, epoch)),
                            global_step=epoch)
                print('Weights saved to file: {}'.format(save_path))

        # if epoch % 2 == 0 and epoch > 1:
        if step % 10000 == 0:
            save_path = saver.save(sess, os.path.join(directories.checkpoints, 'conv_{}_epoch{}.ckpt'.format(name, epoch)), global_step=epoch)
            print('Weights saved to file: {}'.format(save_path))

        print('Epoch {} | Training Acc: {:.3f} | Test Acc: {:.3f} | Test Loss: {:.3f} | Test AUC: {:.3f} | Adv. loss: {:.3f} | Total loss: {:.3f} | Rate: {} examples/s ({:.2f} s) {}'.format(epoch, t_acc, v_acc, v_loss, v_auc, v_adv_loss, v_total, int(config.batch_size/(time.time()-t0)), time.time() - start_time, improved))
        Utils.plot_distributions(model, name, epoch, step, sess, handle=test_handle)


        return v_auc_best

    @staticmethod
    def plot_ROC_curve(y_true, y_pred, out, meta = ''):

        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_curve, auc

        plt.style.use('seaborn-darkgrid')
        plt.style.use('seaborn-talk')
        plt.style.use('seaborn-pastel')

        # Compute ROC curve, integrate
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        print('Val AUC:', roc_auc)

        plt.figure()
        plt.axes([.1,.1,.8,.7])
        plt.figtext(.5,.9, r'$\mathrm{Receiver \;Operating \;Characteristic}$', fontsize=15, ha='center')
        plt.figtext(.5,.85, meta, fontsize=10,ha='center')
        plt.plot(fpr, tpr, # color='darkorange',
                         lw=2, label='ROC (area = %0.4f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=1.0, linestyle='--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel(r'$\mathrm{False \;Positive \;Rate}$')
        plt.ylabel(r'$\mathrm{True \;Positive \;Rate}$')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join('results', '{}_ROC.pdf'.format(out)), format='pdf', dpi=1000)
        plt.gcf().clear()

    @staticmethod
    def mutual_information_1D_naive(x, y, bins=30):
        c_xy = np.histogram2d(x, y, bins)[0]
        c_xy = c_xy / np.sum(c_xy)
        c_y = np.sum(c_xy, axis=0)
        c_x = np.sum(c_xy, axis=1)
        marginal_product = np.expand_dims(c_x, axis=1) * np.expand_dims(c_y, axis=0)
        MI = np.sum(c_xy[c_xy>0] * np.log(c_xy[c_xy>0]/marginal_product[c_xy>0]) + 1e-6)
        
        return MI

    @staticmethod
    def mutual_information_1D_kraskov(x, y):
        # k-NN based estimate of mutual information
        from lnc import MI

        mi = MI.mi_LNC([x,y],k=5,base=np.exp(1),alpha=0.2)
        return mi

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

