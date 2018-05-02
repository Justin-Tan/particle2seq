#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
from config import config_train, config_test, directories

class Data(object):

    @staticmethod
    def load_data(filename, evaluate=False):
        df = pd.read_hdf(filename, key='df').sample(frac=1).reset_index(drop=True)
        # auxillary = ['labels', 'MCtype', 'channel', 'evtNum', 'idx', 'mbc', 'nCands', 'deltae']
        auxillary = ['label', 'B_deltaE', 'B_Mbc', 'B_eventCached_boevtNum', 'ewp_channel']
        df_features = df.drop(auxillary, axis=1)

        if evaluate:
            return df, np.nan_to_num(df_features.values), df['label'].values
        else:
            return np.nan_to_num(df_features.values), df['label'].values


    @staticmethod
    def load_dataset(features_placeholder, labels_placeholder, batch_size, test=False, evaluate=False):
    
        # def _preprocess(tokens, label):
        #     return tokens, label

        dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
        # dataset = dataset.map(_preprocess)
        
        if evaluate is False:
            dataset = dataset.shuffle(buffer_size=512)

        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])),
            padding_values=(0.,0))

        if test is True:
            dataset = dataset.repeat()

        return dataset
