#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
from config import config_train, config_test, directories

class Data(object):

    @staticmethod
    def load_data(filename, evaluate=False, adversary=False):

        if evaluate:
            config = config_test
        else:
            config = config_train

        df = pd.read_hdf(filename, key='df').sample(frac=1).reset_index(drop=True)
        # auxillary = ['labels', 'MCtype', 'channel', 'evtNum', 'idx', 'mbc', 'nCands', 'deltae']
        auxillary = ['label', 'B_deltaE', 'B_Mbc', 'l_pt', 'q_sq', 'B_eventCached_boevtNum', 'B_ewp_channel', 'B_dilepton_type']
        auxillary += ['nCands', 'B_TagVNTracks']
        auxillary += ['B_cms_p', 'B_cms_pt', 'B_cms_q2Bh']
        auxillary += ['B_ell0_cms_E', 'B_ell0_cms_eRecoil', 'B_ell0_cms_m2Recoil', 'B_ell0_cms_p', 'B_ell0_cms_pt']
        auxillary += ['B_ell1_cms_E', 'B_ell1_cms_eRecoil', 'B_ell1_cms_m2Recoil', 'B_ell1_cms_p', 'B_ell1_cms_pt']
        df_features = df.drop(auxillary, axis=1)

        if adversary:
            pivots = ['B_Mbc']  # select pivots
            pivot_bins = ['mbc_labels']
            pivot_df = df[pivots]
            pivot_df = pivot_df.assign(mbc_labels=pd.qcut(df['B_Mbc'], q=config.adv_n_classes, labels=False))
            pivot_features = pivot_df['B_Mbc']
            pivot_labels = pivot_df['mbc_labels']
        else:
            pivots = ['B_Mbc'] #, 'B_cms_p', 'B_cms_pt'] #, 'B_cms_q2Bh']
            # pivots += ['B_ell0_cms_E', 'B_ell0_cms_eRecoil', 'B_ell0_cms_m2Recoil', 'B_ell0_cms_p', 'B_ell0_cms_pt']
            # pivots += ['B_ell1_cms_E', 'B_ell1_cms_eRecoil', 'B_ell1_cms_m2Recoil', 'B_ell1_cms_p', 'B_ell1_cms_pt']
        pivot_df = df[pivots]
        pivot_features = pivot_df[pivots]
        pivot_marginal = pivot_df['B_Mbc'].sample(frac=1).reset_index(drop=True)

        if evaluate:
            return df, np.nan_to_num(df_features.values), df['label'].values.astype(np.int32), \
                pivot_features.values.astype(np.float32), pivot_marginal.values.astype(np.float32)
        else:
            if adversary:
                return np.nan_to_num(df_features.values), df['label'].values.astype(np.int32), \
                    pivot_features.values.astype(np.float32), pivot_labels.values.astype(np.int32)
            else:
                return np.nan_to_num(df_features.values), df['label'].values.astype(np.int32), \
                    pivot_features.values.astype(np.float32), pivot_marginal.values.astype(np.int32)


    @staticmethod
    def load_dataset(features_placeholder, labels_placeholder, pivots_placeholder, marginal_placeholder, batch_size, test=False,
            evaluate=False, sequential=True):
    
        # def _preprocess(tokens, label):
        #     return tokens, label

        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder,
            pivots_placeholder, marginal_placeholder))
        # dataset = dataset.map(_preprocess)
        
        if evaluate is False:
            dataset = dataset.shuffle(buffer_size=512)

        if sequential:
            dataset = dataset.padded_batch(
                batch_size,
                padded_shapes=(tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])),
                padding_values=(0.,0,0.,0.))
        else:
            dataset = dataset.batch(batch_size)
            # dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

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
