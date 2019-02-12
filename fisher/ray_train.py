#!/usr/bin/python3
"""
Simple hyperparameter optimization using the Tune class based API
JT 2019
Usage: 
python3 train.py -i /path/to/train -test /path/to/test -arch dense -pq -n testing
"""
import tensorflow as tf
import numpy as np
import pandas as pd
import time, os, sys
import argparse
import functools

import ray
import ray.tune as tune

# User-defined
from network import Network
from utils import Utils
from data import Data
from tune_model import Model
from config import config_train, directories

tf.logging.set_verbosity(tf.logging.ERROR)

def trial_str_creator(trial, name):
    return "{}_{}_{}".format(trial.trainable_name, trial.trial_id, name)

class TrainModel(tune.Trainable):

    def _retrieve_objects(self, object_ids):
        return [tune.util.get_pinned_object(object_id) for object_id in object_ids]

    def _initialize_train_iterator(self, sess, model, features, labels, pivots):
        sess.run(model.train_iterator.initializer, feed_dict={
            model.features_placeholder: features,
            model.labels_placeholder: labels,
            model.pivots_placeholder: pivots})

    def _setup(self, config):
        """
        args/user_config: User-controlled hyperparameters, that will be held constant
        config = tune_config: parameters to be optimized
        """
        self.args = config.pop("args")
        self.user_config = config.pop("user_config")
        train_object_ids = config.pop("train_ids")
        test_object_ids = config.pop("test_ids")
        self.tune_config = config

        # Model initialization
        assert(self.user_config.use_adversary is False), 'To use adversarial training, run `adv_train.py`'
        self.start_time = time.time()
        self.train_blocks = 0
        self.global_step, self.n_checkpoints, self.v_auc_best = 0, 0, 0.
        self.ckpt = tf.train.get_checkpoint_state(directories.checkpoints)

        print("Retrieving data ...")
        self.features, self.labels, self.pivots = self._retrieve_objects(train_object_ids)
        test_features, test_labels, test_pivots = self._retrieve_objects(test_object_ids)
        self.user_config.max_seq_len = int(self.features.shape[1]/self.user_config.features_per_particle)

        # Build graph
        self.model = Model(self.user_config, tune_config=self.tune_config, features=self.features, 
            labels=self.labels, args=self.args)
        self.saver = tf.train.Saver()

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
            log_device_placement=False))
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        self.train_handle = self.sess.run(self.model.train_iterator.string_handle())
        self.test_handle = self.sess.run(self.model.test_iterator.string_handle())

        self.sess.run(self.model.test_iterator.initializer, feed_dict={
            self.model.test_features_placeholder: test_features,
            self.model.test_labels_placeholder: test_labels,
            self.model.test_pivots_placeholder: test_pivots,
            self.model.pivots_placeholder: test_pivots})

        self._initialize_train_iterator(sess=self.sess, model=self.model, features=self.features, 
            labels=self.labels, pivots=self.pivots)


    def _train(self):
        """
        Run one logical block of training
        Number of iterations should balance overhead with diagnostic frequency
        """
        block_iterations = 1024  # capped by equivalent number of epochjs

        for i in range(block_iterations):
            try:
                global_step, *ops = self.sess.run([self.model.global_step,
                    self.model.train_op], feed_dict={self.model.training_phase: True,
                    self.model.handle: self.train_handle})

                # Inner loop
                if self.args.mutual_information_penalty:
                    for _ in range(int(self.tune_config['MINE_iters'])):
                        self.sess.run(self.model.MINE_train_op, 
                            feed_dict={self.model.training_phase:True, 
                            self.model.handle: self.test_handle})
            except tf.errors.OutOfRangeError:
                print('You have reached the end of the epoch.')
                break

        # Calculate metrics - incurs overhead!
        self._initialize_train_iterator(self.sess, self.model, self.features, self.labels, self.pivots)
        self.train_blocks += 1
        v_MI_kraskov, v_MI_MINE = self.sess.run([self.model.MI_logits_theta_kraskov, 
            self.model.MI_logits_theta], feed_dict={self.model.training_phase: False, 
            self.model.handle: self.test_handle})
        self.v_auc_best, v_auc, v_acc, v_loss = Utils.run_tune_diagnostics(self.model, self.user_config, 
            directories, self.sess, self.saver, self.train_handle, self.test_handle, self.start_time, 
            self.v_auc_best, self.train_blocks, self.global_step, self.args.name)

        # Reported on validation set
        metrics = {
            "episode_reward_mean": v_auc,
            "mean_loss": v_MI_kraskov,
            "mean_accuracy": v_acc
        }

        return metrics

    def _save(self, checkpoint_dir):
        # Save weights to checkpoint $save_path
        save_path = os.path.join(checkpoint_dir, "save")
        # save_path = os.path.join(checkpoint_dir, 'model_{}'.format(self.args.name))
        print('Saving to checkpoint {}'.format(save_path))
        target = self.saver.save(self.sess, save_path, global_step=self.train_blocks)
        print('TARGET:', target)
        return target


    def _restore(self, checkpoint_path):
        # Restore from checkpoint $path
        print('Loading from checkpoint {}'.format(checkpoint_path))
        return self.saver.restore(self.sess, checkpoint_path)

def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=None, help="Path to training file", type=str)
    parser.add_argument("-test", "--test", default=None, help="Path to test file", type=str)
    parser.add_argument("-opt", "--optimizer", default="adam", help="Selected optimizer", type=str)
    parser.add_argument("-n", "--name", default="tune_hb", help="Checkpoint/Tensorboard label")
    parser.add_argument("-arch", "--architecture", default="dense", help="Neural architecture",
        choices=set(('deep_conv', 'recurrent', 'simple_conv', 'conv_projection', 'dense')))
    parser.add_argument("-pq", "--parquet", help="Use if dataset is in parquet format", action="store_true")
    parser.add_argument("-MI", "--mutual_information_penalty", help="Penalize mutual information between pivots and logits", action="store_true")
    parser.add_argument("-JSD", "--JSD", help="Use Jensen-Shannon approximation of mutual information", action="store_true")
    parser.add_argument("-reg", "--regularizer", help="Toggle gradient-based regularization", action="store_true")
    parser.add_argument("-lambda", "--MI_lambda", default=0.0, help="Control tradeoff between xentropy and MI penalization", type=float)
    parser.add_argument("-re", "--restart_epoch", default=0, help="Epoch to restart from", type=int)
    parser.add_argument("-pbt", "--pbt", help="Use population based training scheduler", action="store_true")
    args = parser.parse_args()

    ray.init(num_gpus=4, object_store_memory=24*10**9)
    # Pin dataset to common store for efficient retrival by workers
    print('Reading data ...')
    if args.input is None or args.test is None:
        input_file = directories.train
        test_file = directories.test
    else:
        input_file = args.input
        test_file = args.test
        features_id, labels_id, pivots_id = Data.load_data(input_file, parquet=args.parquet, tune=True)
        test_features_id, test_labels_id, test_pivots_id = Data.load_data(test_file, parquet=args.parquet, tune=True)

    # Subclass trainable
    tune.register_trainable("model_name", TrainModel)
    stopping_criteria = {
        "time_total_s": 7200,
        "episode_reward_mean": 1.0
    }

    # Hyperparameters to be optimized
    # Defining the search space is critical for sensible results
    config = {
        "args": args,
        "user_config": config_train,
        "train_ids": [features_id, labels_id, pivots_id],
        "test_ids": [test_features_id, test_labels_id, test_pivots_id],
        "learning_rate": tune.sample_from(lambda spec: 10**np.random.uniform(-5,-3)),
        "MINE_learning_rate": tune.sample_from(lambda spec: 10**np.random.uniform(-5,-3)),
        "MINE_iters": tune.sample_from(lambda spec: np.random.uniform(1,32)), 
        "MI_lambda": tune.sample_from(lambda spec: np.random.uniform(0,100)),
        "jsd": tune.grid_search([True, False]),
        "heuristic": tune.grid_search([True, False]),
        "n_layers": tune.grid_search([1,2,3,4])
    }

    hp_config = {k: config[k] for k in config.keys() if k not in ['args', 'user_config', 'train_ids', 'test_ids']}
    # For population-based training
    hp_resamples_pbt = {
        "learning_rate": lambda: 10**np.random.uniform(-5,-3),
        "MINE_learning_rate": lambda: 10**np.random.uniform(-5,-3),
        "MINE_iters": lambda: np.random.uniform(1,32), 
        "MI_lambda": lambda: np.random.uniform(0,100),
        "jsd": [True, False],
        "heuristic": [True, False],
        "n_layers": [1,2,3,4]
    }

    # Specify experiment configuration
    # Done on a machine with 24 CPUs / 4 GPUs
    experiment_spec = tune.Experiment(
        name=args.name,
        run=TrainModel,
        # run=functools.partial(TrainModel, args=args, user_config=config_train),
        stop=stopping_criteria,
        config=config,
        resources_per_trial={'cpu': 4, 'gpu': 1},
        # extra_cpu=2,
        num_samples=16,
        local_dir='/data/cephfs/punim0011/jtan/ray_results',
        checkpoint_freq=1,
        checkpoint_at_end=True,
        trial_name_creator=tune.function(functools.partial(trial_str_creator, 
            name=args.name))
    )

    # Launch training
    # time_attr corresponds to one logical training block

    pbt = tune.schedulers.PopulationBasedTraining(
        time_attr="training_iteration",
        reward_attr="episode_reward_mean",
        perturbation_interval=4,  # Mutation interval in time_attr units
        hyperparam_mutations=hp_resamples_pbt,
        resample_probability=0.25  # Resampling resets to value sampled from specified function
    )

    ahb = tune.schedulers.AsyncHyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="episode_reward_mean",
        max_t=64,
        grace_period=4,
        reduction_factor=3,
        brackets=3
    )

    scheduler = ahb
    if args.pbt is True:
        scheduler = pbt

    trials = tune.run_experiments(
        experiments=experiment_spec,
        scheduler=scheduler,
        resume=False  # "prompt"
    )

    t_ids = [t.trial_id for t in trials]
    t_config = [t.config for t in trials]
    t_result = [t.last_result for t in trials]
    df = pd.DataFrame([t_ids, t_config, t_result]).transpose()
    df.columns = ['name', 'config', 'result']
    df.to_hdf('{}_results.h5'.format(args.name)) 

if __name__ == '__main__':
    main()
