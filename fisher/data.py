#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
from config import config_train, config_test, directories

class Data(object):

    @staticmethod
    def load_data(filename, evaluate=False, adversary=False, parquet=False, tune=False):

        if evaluate:
            config = config_test
        else:
            config = config_train

        if parquet:
            import pyarrow.parquet as pq
            dataset = pq.ParquetDataset(filename)
            df = dataset.read(nthreads=4).to_pandas()
        else:
            df = pd.read_hdf(filename, key='df')

        if not evaluate:
            df = df.sample(frac=1).reset_index(drop=True)

        auxillary = ['label'] + [col for col in df.columns if 'deltaE' in col or 'Mbc' in col or 'evtNum' in col or
            'hadronic_mass' in col or 'mctype' in col or 'channel' in col or 'nCands' in col or 'DecayHash' in col or 'decstring' in col]

        df_features = df.drop(auxillary, axis=1)

        if adversary:
            # Bin variable -> discrete classification problem
            pivots = ['B_Mbc']  # select pivots
            pivot_bins = ['mbc_labels']
            pivot_df = df[pivots]
            pivot_df = pivot_df.assign(mbc_labels=pd.qcut(df['B_Mbc'], q=config.adv_n_classes, labels=False))
            pivot_features = pivot_df['B_Mbc']
            pivot_labels = pivot_df['mbc_labels']
        else:
            pivots = ['B_Mbc'] #, 'B_cms_p', 'B_cms_pt'] #, 'B_cms_q2Bh']

        pivot_df = df[pivots]
        pivot_features = pivot_df[pivots]

        marginal_df = pivot_features[pivots]
        marginal_df = marginal_df.sample(frac=1)
        marginal_df = marginal_df.reset_index(drop=True)
        pivot_features[[pivot + '_marginal' for pivot in pivots]] = marginal_df


        if evaluate:
            return df, np.nan_to_num(df_features.values), df['label'].values.astype(np.int32), \
                pivot_features.values.astype(np.float32)
        else:
            if adversary:
                return np.nan_to_num(df_features.values), df['label'].values.astype(np.int32), \
                    pivot_features.values.astype(np.float32), pivot_labels.values.astype(np.int32)
            else:
                if tune:
                    from ray.tune.util import pin_in_object_store as pin
                    return pin(np.nan_to_num(df_features.values)), pin(df['label'].values.astype(np.int32)), \
                        pin(pivot_features.values.astype(np.float32))
                else:
                    return np.nan_to_num(df_features.values), df['label'].values.astype(np.int32), \
                        pivot_features.values.astype(np.float32)


    @staticmethod
    def load_dataset(features_placeholder, labels_placeholder, pivots_placeholder, batch_size, test=False,
            evaluate=False, sequential=True, prefetch_size=2):
    
        # def _preprocess(features, label):
        #     return features, label

        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder,
            pivots_placeholder))
        # dataset = dataset.map(_preprocess)
        
        if evaluate is False:
            dataset = dataset.shuffle(buffer_size=10**5)

        if sequential:
            dataset = dataset.padded_batch(
                batch_size,
                padded_shapes=(tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([])),
                padding_values=(0.,0,0.),
                drop_remainder=True)
        else:
            #if evaluate:
            #    dataset = dataset.batch(batch_size)
            #else:
            #    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
            dataset = dataset.batch(batch_size, drop_remainder=True)
            # dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
            dataset = dataset.prefetch(prefetch_size)

        if test is True:
            dataset = dataset.repeat()

        return dataset

    @staticmethod
    def load_dataset_adversary(features_placeholder, labels_placeholder, pivots_placeholder, 
            pivot_labels_placeholder, batch_size, test=False, evaluate=False, sequential=True):

        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder, 
            pivots_placeholder, pivot_labels_placeholder))

        if evaluate is False:
            dataset = dataset.shuffle(buffer_size=512)

        if sequential:
            dataset = dataset.padded_batch(
                batch_size,
                padded_shapes=(tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])),
                padding_values=(0.,0,0.,0))
        else:   
            dataset = dataset.batch(batch_size)

        if test is True:
            dataset = dataset.repeat()

        return dataset
