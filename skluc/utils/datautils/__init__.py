"""
Utility functions that deal with numpy arrays
"""
import keras
import numpy as np
from sklearn.metrics.pairwise import additive_chi2_kernel
import scipy
from skluc.utils import logger


def dimensionality_constraints(d):
    """
    Enforce d to be a power of 2

    :param d: the original dimension
    :return: the final dimension
    """
    if not is_power_of_two(d):
        # find d that fulfills 2^l
        d = np.power(2, np.floor(np.log2(d)) + 1)
    return d


def is_power_of_two(input_integer):
    """ Test if an integer is a power of two. """
    if input_integer == 1:
        return False
    return input_integer != 0 and ((input_integer & (input_integer - 1)) == 0)


def build_hadamard(n_neurons):
    return scipy.linalg.hadamard(n_neurons)


def compute_euristic_sigma_chi2(dataset_full, slice_size=100):
    """
    Given a dataset, return the gamma that should be used (euristically) when using a rbf kernel on this dataset.

    The formula: $\sigma^2 = 1/n^2 * \sum_{i, j}^{n}||x_i - x_j||^2$

    :param dataset: The dataset on which to look for the best sigma
    :return:
    """
    dataset_full = np.reshape(dataset_full, (-1, 1))
    results = []
    if slice_size > dataset_full.shape[0]:
        slice_size = dataset_full.shape[0]
    for i in range(dataset_full.shape[0] // slice_size):
        if (i+1) * slice_size <= dataset_full.shape[0]:
            dataset = dataset_full[i * slice_size: (i+1) * slice_size]
            slice_size_tmp = slice_size
        else:
            dataset = dataset_full[i * slice_size:]
            slice_size_tmp = len(dataset)
        # wall = np.expand_dims(dataset, axis=1)
        # # the drawing of the matrix Y expanded looks like a floor
        # floor = np.expand_dims(dataset, axis=0)
        # numerator = np.square((wall - floor))
        # denominator = wall + floor
        # quotient = numerator / denominator
        # quotient_without_nan = replace_nan(quotient)
        quotient_without_nan = additive_chi2_kernel(dataset)
        results.append(1/slice_size_tmp**2 * np.sum(quotient_without_nan))
        logger.debug("Compute sigma chi2; current mean: {}".format(np.mean(results)))
    return np.mean(results)

def compute_euristic_sigma(dataset_full, slice_size=1000):
    """
    Given a dataset, return the gamma that should be used (euristically) when using a rbf kernel on this dataset.

    The formula: $\sigma^2 = 1/n^2 * \sum_{i, j}^{n}||x_i - x_j||^2$

    :param dataset: The dataset on which to look for the best sigma
    :return:
    """
    results = []
    dataset_full = np.reshape(dataset_full, (-1, 1))
    if slice_size > dataset_full.shape[0]:
        slice_size = dataset_full.shape[0]
    for i in range(dataset_full.shape[0] // slice_size):
        if (i+1) * slice_size <= dataset_full.shape[0]:
            dataset = dataset_full[i * slice_size: (i+1) * slice_size]
            slice_size_tmp = slice_size
        else:
            dataset = dataset_full[i * slice_size:]
            slice_size_tmp = len(dataset)
        r1 = np.sum(dataset * dataset, axis=1)
        r1 = np.reshape(r1, [-1, 1])
        r2 = np.reshape(r1, [1, -1])
        d_mat = np.dot(dataset, dataset.T)
        d_mat = r1 - 2 * d_mat + r2
        results.append(1/slice_size_tmp**2 * np.sum(d_mat))
    return np.mean(results)

def replace_nan(tensor):
    return np.where(np.isnan(tensor), np.zeros_like(tensor), tensor)

def rearrange_2D_array_by_2D_array(array, index_arr):
    """
    Inplace rearrange array following indexes of index_arr.

    :param array:
    :param index_arr:
    :return:
    """
    raveled = np.ravel(array)
    raveled_ind = np.ravel(index_arr)
    raveled[:] = raveled[np.repeat(np.arange(array.shape[0]), array.shape[1]) * array.shape[1] + raveled_ind]
    return array

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

