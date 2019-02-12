#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
import time, os, sys
import argparse
from lnc import MI

def MINE(x, y, y_prime, training, batch_size, name='MINE', actv=tf.nn.relu, dimension=2):
    """
    Mutual Information Neural Estimator
    (x,y):      Drawn from joint
    y_prime:    Drawn from marginal

    returns
    MI:         Lower bound on mutual information between x,y
    """

    init = tf.contrib.layers.xavier_initializer()
    drop_rate = 0.0
    shape = [512,512,512,512,512,512]
    shape = [128,128,128,128,128,128]
    shape = [64,64,64,64,64,64,64]
    shape = [32,32,32,32,32,32,32]
    kwargs = {'center': True, 'scale': True, 'training': training, 'fused': True, 'renorm': False}

    if dimension == 2:
        x,y,y_prime = tf.expand_dims(x, axis=1), tf.expand_dims(y, axis=1), tf.expand_dims(y_prime, axis=1)
    # y_prime = tf.random_shuffle(y)

    z = tf.concat([x,y], axis=1)
    z_prime = tf.concat([x,y_prime], axis=1)
    print('Z SHAPE', z.get_shape().as_list())
    print('Z_PRIME SHAPE', z_prime.get_shape().as_list())

    lambda_l2 = 0.001
    regularization =  tf.contrib.layers.l2_regularizer(scale=lambda_l2)
    regularization = None

    def statistic_network(t, name='stat_net', reuse=False):
        with tf.variable_scope(name, initializer=init, reuse=reuse) as scope:

            h0 = tf.layers.dense(t, units=shape[0], activation=actv, kernel_regularizer=regularization) 
            h0 = tf.layers.dropout(h0, rate=drop_rate, training=training)
            # h0 = tf.layers.batch_normalization(h0, **kwargs)

            h1 = tf.layers.dense(h0, units=shape[1], activation=actv, kernel_regularizer=regularization) 
            h1 = tf.layers.dropout(h1, rate=drop_rate, training=training)
            # h1 = tf.layers.batch_normalization(h1, **kwargs)

            h2 = tf.layers.dense(h1, units=shape[2], activation=actv, kernel_regularizer=regularization) 
            h2 = tf.layers.dropout(h2, rate=drop_rate, training=training)
            # h2 = tf.layers.batch_normalization(h2, **kwargs)

            h3 = tf.layers.dense(h2, units=shape[3], activation=actv, kernel_regularizer=regularization) 
            h3 = tf.layers.dropout(h3, rate=drop_rate, training=training)
            #h3 = tf.layers.batch_normalization(h3, **kwargs)

            #h4 = tf.layers.dense(h3, units=shape[4], activation=actv, kernel_regularizer=regularization) 
            #h4 = tf.layers.dropout(h4, rate=drop_rate, training=training)

            #h5 = tf.layers.dense(h4, units=shape[5], activation=actv, kernel_regularizer=regularization) 
            #h5 = tf.layers.dropout(h5, rate=drop_rate, training=training)

            out = tf.layers.dense(h2, units=1, kernel_initializer=init, kernel_regularizer=regularization) 

        return out

    def log_sum_exp_trick(x, batch_size, axis=1):
        # Compute along batch dimension
        x = tf.squeeze(x)
        x_max = tf.reduce_max(x)
        # lse = x_max + tf.log(tf.reduce_mean(tf.exp(x-x_max)))
        lse = x_max + tf.log(tf.reduce_sum(tf.exp(x-x_max))) - tf.log(batch_size)
        return lse

    joint_f = statistic_network(z)
    marginal_f = statistic_network(z_prime, reuse=True)
    print('joint shape', joint_f.shape)
    print('marginal shape', marginal_f.shape)
    
    # MI_lower_bound = tf.reduce_mean(joint_f) - tf.log(tf.reduce_mean(tf.exp(marginal_f)) + 1e-5)
    MI_lower_bound = tf.squeeze(tf.reduce_mean(joint_f)) - tf.squeeze(log_sum_exp_trick(marginal_f, 1.*batch_size))

    #joint_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=joint_f,
    #    labels=tf.ones(batch_size, tf.int32)))
    #marginal_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=marginal_f, 
    #    labels=tf.zeros(batch_size, tf.int32)))

    joint_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=joint_f,
        labels=tf.ones_like(joint_f)))
    marginal_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=marginal_f,
        labels=tf.zeros_like(marginal_f)))

    JSD_lower_bound = -(joint_loss + marginal_loss) + tf.log(4.0)
    # JSD_lower_bound = tf.reduce_mean(tf.log(tf.nn.sigmoid(joint_f))) + tf.reduce_mean(tf.log(1.0 - tf.nn.sigmoid(marginal_f)))

    # JSD_lower_bound = tf.reduce_mean(-tf.nn.softplus(-tf.squeeze(joint_f) + 1e-5)) - tf.reduce_mean(tf.nn.softplus(tf.squeeze(marginal_f + 1e-5)))

    # lower_bound = MI_lower_bound
    lower_bound = JSD_lower_bound

    return lower_bound

def mutual_information_1D_kraskov(x, y):
    # k-NN based estimate of mutual information
    from lnc import MI
    mi = MI.mi_LNC([x,y],k=5,base=np.exp(1),alpha=0.5)

    return mi

def scope_variables(name):
    with tf.variable_scope(name):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)

class Model():
    def __init__(self, args, X, Z, Z_prime):

        tfd = tf.contrib.distributions
        self.global_step = tf.Variable(0, trainable=False)

        if args.feed_input:
            self.X_placeholder = tf.placeholder(X.dtype, X.shape)
            self.Z_placeholder = tf.placeholder(Z.dtype, Z.shape)
            self.Z_prime_placeholder = tf.placeholder(Z_prime.dtype, Z_prime.shape)
            dataset = tf.data.Dataset.from_tensor_slices((self.X_placeholder, self.Z_placeholder,
                self.Z_prime_placeholder))
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(args.sample_size)
            dataset = dataset.repeat(int(1e8))
            self.iterator = dataset.make_initializable_iterator()
            x, y, y_prime = self.iterator.get_next()
            self.X, self.Z, self.Z_prime = x, y, y_prime
            self.MI_true = np.nan
            dim = 2
            self.MI_estimate = MINE(x, y, y_prime, batch_size=args.sample_size, dimension=dim, training=True,
                    actv=tf.nn.elu)
        else:
            if args.dims == 2:
                dim = 2
                mu = [5.,5.]
                cov = [[2.,2.*args.rho],[2.*args.rho,2.]]
                mvn = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov)
                self.mvn_sample = mvn.sample(args.sample_size)
                self.mvn_sample_2 = mvn.sample(args.sample_size)

                x = self.mvn_sample[:,0]
                y = self.mvn_sample[:,1]
                y_prime = self.mvn_sample_2[:,1]
                self.MI_true = - 0.5 * np.log(1-args.rho**2)
                self.X, self.Z, self.Z_prime = x, y, y_prime

            else:
                from sklearn import datasets 
                dim = args.dims
                m_dim = int(dim/2)
                mu = np.zeros(dim).astype(np.float32)
                cov = datasets.make_spd_matrix(n_dim=dim).astype(np.float32)
                cov_x, cov_y = cov[:m_dim,:m_dim], cov[m_dim:,m_dim:]
                mvn = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov)
                self.mvn_sample = mvn.sample(args.sample_size)
                self.mvn_sample_2 = mvn.sample(args.sample_size)
                mvn_x = tfd.MultivariateNormalFullCovariance(loc=mu[:m_dim], covariance_matrix=cov_x)
                mvn_y = tfd.MultivariateNormalFullCovariance(loc=mu[:m_dim], covariance_matrix=cov_y)

                x = self.mvn_sample[:,:m_dim]
                y = self.mvn_sample[:,m_dim:]
                y_prime = self.mvn_sample_2[:,m_dim:]

                self.MI_true = 1/2 * np.log(np.linalg.det(cov_x) * np.linalg.det(cov_y) / np.linalg.det(cov))
                self.tf_kl = tf.distributions.kl_divergence(mvn_x,mvn_y)

            control_deps = [self.mvn_sample, self.mvn_sample_2]
            with tf.variable_scope('miner') as scope:
                with tf.control_dependencies(control_deps):
                    self.MI_estimate = MINE(x, y, y_prime, batch_size=args.sample_size, dimension=dim, training=True,
                            actv=tf.nn.elu)

        self.MI_estimate_kNN = tf.py_func(mutual_information_1D_kraskov, inp=[tf.squeeze(x), tf.squeeze(y)], Tout=tf.float64)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        lr = 1e-4
        theta_MINE = scope_variables('miner')
        print('Parameters', theta_MINE)
        args.optimizer = args.optimizer.lower()
        if args.optimizer == 'adam':
            self.opt = tf.train.AdamOptimizer(lr)
        elif args.optimizer == 'momentum':
            self.opt = tf.train.MomentumOptimizer(lr, momentum=0.9, use_nesterov=True)
        elif args.optimizer == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(lr)
        elif args.optimizer == 'sgd':
            self.opt =  tf.train.GradientDescentOptimizer(lr)
        # self.opt_op = self.opt.minimize(-self.MI_estimate, global_step=self.global_step)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            self.opt_op = self.opt.minimize(-self.MI_estimate, global_step=self.global_step,
                    var_list=theta_MINE)
            self.grad_loss = tf.get_variable(name='grad_loss', shape=[], trainable=False)
            self.grads = self.opt.compute_gradients(-self.MI_estimate, grad_loss=self.grad_loss)

            self.ema = tf.train.ExponentialMovingAverage(decay=0.995, num_updates=self.global_step)
            maintain_averages_op = self.ema.apply(tf.trainable_variables())

            with tf.control_dependencies(update_ops+[self.opt_op]):
                self.train_op = tf.group(maintain_averages_op)

        tf.summary.scalar('MI_estimate', self.MI_estimate)
        self.merge_op = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(os.path.join('tensorboard', 'MINE_estimator_{}'.format(time.strftime('%d-%m_%I:%M'))), 
            graph=tf.get_default_graph())


def train(args):

    start_time = time.time()
    ckpt = tf.train.get_checkpoint_state('checkpoints')

    if args.feed_input:
        df = pd.read_hdf('/data/cephfs/punim0011/jtan/data/dense_results.h5')
        X = df['logits']
        Z = df['pivots']
        Z_prime = df['pivots_marginal']
    else:
        X, Z, Z_prime = None, None, None

    # Build graph
    miner = Model(args, X, Z, Z_prime)
    MI_true = miner.MI_true
    saver = tf.train.Saver()
    print(tf.all_variables())
    estimates = list()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if args.feed_input:
            # Feed in input instead of using multivariate Gaussians
            sess.run(miner.iterator.initializer, feed_dict={miner.X_placeholder: X, miner.Z_placeholder: Z,
                miner.Z_prime_placeholder: Z_prime})

        tf_kl = 1.0
        for step in range(int(1e6)):
            # Update weights
            # MI_est, *ops = sess.run([miner.MI_estimate, miner.mvn_sample, miner.mvn_sample_2, miner.train_op])
            # sess.run(miner.opt_op)
            sess.run(miner.opt_op)

            if step % 1000 == 0:
                if args.dims == 2:
                    # MI_est, MI_kNN, *ops = sess.run([miner.MI_estimate, miner.MI_estimate_kNN, miner.mvn_sample])
                    MI_est, MI_kNN, mX, mZ, mZp = sess.run([miner.MI_estimate, miner.MI_estimate_kNN, miner.X, miner.Z, miner.Z_prime])
                    # print('X', mX)
                    # print('============================================================>>>')
                    # print('Z', mZ)
                    # print('============================================================>>>')
                    # print('Z_prime', mZp)

                    
                    grads = sess.run(miner.grad_loss)
                    print('MINE: {:.3f} | kNN: {:.3f} | Analytic: {:.3f} | tf_KL: {:.3f} | MINE-True Difference: {:.3f} % | kNN-True Difference: {:.3f} % | ({:.2f} s)'.format(MI_est,
                        MI_kNN, MI_true, tf_kl, (MI_est-MI_true)/MI_true*100, (MI_kNN-MI_true)/MI_true*100, time.time()-start_time))
                    estimates.append(MI_est)
                else:
                    # MI_est, sample = sess.run([miner.MI_estimate, miner.mvn_sample])
                    MI_est = sess.run(miner.MI_estimate)
                    
                    grads = sess.run(miner.grad_loss)
                    print('MINE: {:.3f} | Analytic: {:.3f} | MINE-True Difference: {:.3f} % | ({:.2f} s)'.format(MI_est, MI_true, (MI_est-MI_true)/MI_true*100, time.time()-start_time))

                print('Loss gradient', grads)

        save_path = saver.save(sess, os.path.join('checkpoints',
                               'miner_{}_end.ckpt'.format(args.name)),
                               global_step=step)

    print("Training Complete. Model saved to file: {} Time elapsed: {:.3f} s".format(save_path, time.time()-start_time))

def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-rho", "--rho", default=0.5, help="Correlation coefficient", type=float)
    parser.add_argument("-ss", "--sample_size", default=256, help="MINE sample size", type=int)
    parser.add_argument("-dim", "--dims", default=2, help="Normal distribution dimension", type=int)
    parser.add_argument("-opt", "--optimizer", default="sgd", help="Selected optimizer", type=str)
    parser.add_argument("-n", "--name", default="MINE", help="Checkpoint/Tensorboard label", type=str)
    parser.add_argument("-fi", "--feed_input", help="Feed in input directly", action="store_true")

    args = parser.parse_args()

    # Launch training
    train(args)

if __name__ == '__main__':
    main() 
