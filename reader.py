#! /usr/bin/python
import numpy as np
import h5py

class Reader(object):
    def __init__(self):
        # Loading the data (cat/non-cat)
        train_dataset = h5py.File('train_catvnoncat.h5', "r")
        self.train_set_X = np.array(train_dataset["train_set_x"][:])  # train set features
        self.train_set_y = np.array(train_dataset["train_set_y"][:])  # train set labels

        test_dataset = h5py.File('test_catvnoncat.h5', "r")
        self.test_set_X = np.array(test_dataset["test_set_x"][:])  # test set features
        self.test_set_y = np.array(test_dataset["test_set_y"][:])  # test set labels

        self.classes = np.array(test_dataset["list_classes"][:])  # the list of classes: 'non-cat', 'cat'

        self.train_len = self.train_set_X.shape[0]  # Number of training examples: m_train = 209
        self.test_len = self.test_set_X.shape[0]  # Number of testing examples: m_test = 50
        self.img_size = self.train_set_X.shape[1]  # Height/Width of each image: num_px = 64
        
        '''
        train_set_X shape: (209, 64, 64, 3)
        train_set_y shape: (209,)
        test_set_X shape: (50, 64, 64, 3)
        test_set_y shape: (50,)
        '''

