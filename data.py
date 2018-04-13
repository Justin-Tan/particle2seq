#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd
from config import config_train, config_test, directories

class Data(object):

    @staticmethod
    def load_data(filename, evaluate=False):
        df = pd.read_hdf(filename, key='df').sample(frac=1).reset_index(drop=True)
        aux = ['labels', 'MCtype', 'channel', 'evtNum', 'idx', 'mbc', 'nCands']
        df_features = df.drop(aux, axis=1)

        if evaluate:
            return df, np.nan_to_num(df_features.values), df['labels'].values
        else:
            return np.nan_to_num(df_features.values), df['labels'].values

    @staticmethod
    def load_tokenized_data(filename):
        print('Reading data from', filename)
        df = pd.read_hdf(filename, key='df').sample(frac=1).reset_index(drop=True)
        tokens = df['tokens']

        # Get lengths of each row of data
        lens = np.array([len(tokens[i]) for i in range(len(tokens))])

        # Mask of valid places in each row
        mask = np.arange(lens.max()) < lens[:,None]

        # Setup output array and put elements from data into masked positions
        padded_tokens = np.zeros(mask.shape)
        padded_tokens[mask] = np.hstack((tokens[:]))

        return padded_tokens, df['category'].values

    @staticmethod
    def load_dataset(features_placeholder, labels_placeholder, batch_size, test=False):
    
        # def _preprocess(tokens, label):
        #     return tokens, label

        dataset = tf.contrib.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
        # dataset = dataset.map(_preprocess)
        dataset = dataset.shuffle(buffer_size=512)

        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None]), tf.TensorShape([])),
            padding_values=(0.,0))

        if test:
            dataset = dataset.repeat()

        return dataset
