# -*- coding: utf-8 -*-
"""
Created on Wed Aug 07 10:43:37 2020

@author: DeSI-Anjin Liu
"""

# liac-arff
import arff
# matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import KFold


class DriftDataset():
    """stream learning dataset loading class"""

    _datasetDict = {
        # RealData_UnknDrift data sets
        "elec": "3_RealData_UnknDrift/elecNorm.arff",
        "weat": "3_RealData_UnknDrift/weather_NSE.arff",
        "spam": "3_RealData_UnknDrift/spam_corpus_x2_feature_selected.arff",
        "airl": "3_RealData_UnknDrift/airline.arff",
        "covt-binary": "3_RealData_UnknDrift/covtypeNormBinary.arff",
        "poke-binary": "3_RealData_UnknDrift/PokerHandBinary.arff",
        "noaa_042700": "3_RealData_UnknDrift/noaa_042700.arff",
        "noaa_100370": "3_RealData_UnknDrift/noaa_100370.arff",
        "noaa_265090": "3_RealData_UnknDrift/noaa_265090.arff",
        "noaa_424750": "3_RealData_UnknDrift/noaa_424750.arff",
        "noaa_567780": "3_RealData_UnknDrift/noaa_567780.arff",
        "noaa_606560": "3_RealData_UnknDrift/noaa_606560.arff",
        "noaa_702220": "3_RealData_UnknDrift/noaa_702220.arff",
        "noaa_802220": "3_RealData_UnknDrift/noaa_802220.arff",
        "noaa_911820": "3_RealData_UnknDrift/noaa_911820.arff",

        # RealData_SyntDrift data sets
        "use1": "2_RealData_SyntDrift/usenet1.arff",
        "use2": "2_RealData_SyntDrift/usenet2.arff",
        "wdbc": "2_RealData_SyntDrift/wdbc.arff",
        "glas": "2_RealData_SyntDrift/glass.arff",
        "iono": "2_RealData_SyntDrift/iono.arff",
        "soyb": "2_RealData_SyntDrift/soyb.arff",
        "iris": "2_RealData_SyntDrift/iris.arff",
        # "wine": "2_RealData_UnknDrift/wine.arff",

        # SyntData_SyntDrift data sets
        "SEAa0": "1_SyntData_SyntDrift/SEAa0.arff",
        "SEAa1": "1_SyntData_SyntDrift/SEAa1.arff",
        "SEAa2": "1_SyntData_SyntDrift/SEAa2.arff",
        "SEAa3": "1_SyntData_SyntDrift/SEAa3.arff",
        "SEAa4": "1_SyntData_SyntDrift/SEAa4.arff",
        "SEAa5": "1_SyntData_SyntDrift/SEAa5.arff",
        "SEAa6": "1_SyntData_SyntDrift/SEAa6.arff",
        "SEAa7": "1_SyntData_SyntDrift/SEAa7.arff",
        "SEAa8": "1_SyntData_SyntDrift/SEAa8.arff",
        "SEAa9": "1_SyntData_SyntDrift/SEAa9.arff",
        "SEAa10": "1_SyntData_SyntDrift/SEAa10.arff",
        "SEAa11": "1_SyntData_SyntDrift/SEAa11.arff",
        "SEAa12": "1_SyntData_SyntDrift/SEAa12.arff",
        "SEAa13": "1_SyntData_SyntDrift/SEAa13.arff",
        "SEAa14": "1_SyntData_SyntDrift/SEAa14.arff",
        "SEAg": "1_SyntData_SyntDrift/SEAg0.arff",
        "HYPi": "1_SyntData_SyntDrift/HYP0.arff",
        "AGRa": "1_SyntData_SyntDrift/AGRa0.arff",
        "AGRg": "1_SyntData_SyntDrift/AGRg0.arff",
        "LEDa": "1_SyntData_SyntDrift/LEDa0.arff",
        "LEDg": "1_SyntData_SyntDrift/LEDg0.arff",
        "RBFi": "1_SyntData_SyntDrift/RBF0.arff",
        "RBFr": "1_SyntData_SyntDrift/RBFr0.arff",
        "RTGn": "1_SyntData_SyntDrift/RTG0.arff"
    }
    for i in range(15):
        _datasetDict['AGRa' + str(i)] = '1_SyntData_SyntDrift/AGRa{}.arff'.format(i)

    def __str__(self):
        return "Class: concept_drift_dataset_loader"

    def __init__(self, path, dataset_name):
        self.DATA_FILE_PATH = path
        self.load_np(dataset_name)

    def load_np(self, dataset_name):
        file_ = self.DATA_FILE_PATH + self._datasetDict[dataset_name]
        dataset = arff.load(open(file_), encode_nominal=True)
        self.np_data = np.array(dataset["data"])


def flip_label(noise_ratio, Y):
    n_samples = Y.shape[0]
    if noise_ratio > 0:
        n_inverse_label = np.floor(n_samples * noise_ratio).astype(int)
        inverse_idx = np.random.permutation(n_samples)[:n_inverse_label]
        Y[inverse_idx] = np.abs(Y[inverse_idx] - 1)
    return Y


def generator_blob_batch(r_seed, n_samples, n_noisy_dim=1, noise_ratio=0.1):
    np.random.seed(r_seed)
    #    centers = [(2, 14), (6, 14), (10, 14),
    #               (2, 10), (6, 10), (10, 10),
    #               (2, 6), (6, 6), (10, 6),
    #               (2, 2), (6, 2), (10, 2),
    #               ]
    # centers = [(2, 14), (6, 14), (6, 10), (2, 10)]
    #    centers = [(2, 14), (6, 14), (10, 14),
    #               (2, 10), (6, 10), (10, 10)]
    centers = [(2, 2), (6, 2), (10, 2)]
    X, Y = datasets.make_blobs(n_samples=n_samples, centers=centers, shuffle=False)
    Y = Y % 2
    Y = flip_label(noise_ratio, Y)
    noise_feats = np.random.uniform(size=[n_samples, n_noisy_dim])
    X = np.hstack([X, noise_feats])
    shuffle_idx = np.random.permutation(X.shape[0])
    X = X[shuffle_idx]
    Y = Y[shuffle_idx]
    return X, Y


def generator_line_batch(r_seed, n_samples, n_noisy_dim=1, noise_ratio=0.1):
    np.random.seed(r_seed)
    X = np.random.normal(0, 1, [n_samples, 2 + n_samples])
    Y = np.zeros(n_samples)
    Y[np.where(X[:, 0] + X[:, 1] >= 0)] = 1
    Y = flip_label(noise_ratio, Y)
    return X, Y


def generator_circ_batch(r_seed, n_samples, n_noisy_dim=1, noise_ratio=0.1):
    np.random.seed(r_seed)
    X, Y = datasets.make_circles(n_samples=n_samples, factor=.4, noise=0.1)
    if n_noisy_dim > 0:
        noisy_dim = np.random.uniform(size=[n_samples, n_noisy_dim])
        X = np.hstack([X, noisy_dim])
    Y = flip_label(noise_ratio, Y)
    shuffle_idx = np.random.permutation(X.shape[0])
    X = X[shuffle_idx]
    Y = Y[shuffle_idx]
    return X, Y


def generator_moon_batch(r_seed, n_samples, n_noisy_dim=1, noise_ratio=0.1):
    np.random.seed(r_seed)
    X, Y = datasets.make_moons(n_samples=n_samples, noise=0.1)
    if n_noisy_dim > 0:
        noisy_dim = np.random.uniform(size=[n_samples, n_noisy_dim])
        X = np.hstack([X, noisy_dim])
    Y = flip_label(noise_ratio, Y)

    shuffle_idx = np.random.permutation(X.shape[0])
    X = X[shuffle_idx]
    Y = Y[shuffle_idx]
    return X, Y


def get_toy_stream(r_seed, batch_generator, delta_list, n_samples_per_batch, n_noisy_dim, noise_ratio,
                   plot_stream_2d=False):
    delta_size = len(delta_list)
    toy_stream_X, toy_stream_Y = batch_generator(r_seed, delta_size * n_samples_per_batch, n_noisy_dim, noise_ratio)
    stream_spliter = KFold(n_splits=delta_size, random_state=None, shuffle=False)
    b_id = 0
    
    if plot_stream_2d:
        plt.figure()
        legend_list = []
        # marker_list_class0 = ['o', '', '', '', '']
        # marker_list_class1 = ['P', '', '', '', '']
        color_list = ['b', 'tab:orange', 'g']
        for _, drift_batch_index in stream_spliter.split(toy_stream_X):
            drift_mat = (toy_stream_X[drift_batch_index].max(0) - toy_stream_X[drift_batch_index].min(0)) * delta_list[
                b_id]
            drift_mat = np.tile(drift_mat, [toy_stream_X[drift_batch_index].shape[0], 1])
            toy_stream_X[drift_batch_index] = toy_stream_X[drift_batch_index] + drift_mat

            X_plot = toy_stream_X[drift_batch_index]
            Y_plot = toy_stream_Y[drift_batch_index]
            lab_0_idx = np.where(Y_plot == 0)
            lab_1_idx = np.where(Y_plot == 1)
            plt.scatter(X_plot[lab_0_idx, 0], X_plot[lab_0_idx, 1], marker='.', c=color_list[b_id])
            plt.scatter(X_plot[lab_1_idx, 0], X_plot[lab_1_idx, 1], marker='x', c=color_list[b_id])
            legend_list.append(r'$\delta=$' + str(delta_list[b_id]) + ', class0')
            legend_list.append(r'$\delta=$' + str(delta_list[b_id]) + ', class1')

            b_id += 1
        plt.title(batch_generator.__name__ + 'toy stream')
        plt.legend(legend_list)
        plt.savefig('img/toy_stream_{}.pdf'.format(batch_generator.__name__.split('_')[1]), bbox_inches='tight')
        plt.show()
        return toy_stream_X, toy_stream_Y
    else:
        for _, drift_batch_index in stream_spliter.split(toy_stream_X):
            drift_mat = (toy_stream_X[drift_batch_index].max(0) - toy_stream_X[drift_batch_index].min(0)) * delta_list[
                b_id]
            drift_mat = np.tile(drift_mat, [toy_stream_X[drift_batch_index].shape[0], 1])
            toy_stream_X[drift_batch_index] = toy_stream_X[drift_batch_index] + drift_mat
            b_id += 1
        return toy_stream_X, toy_stream_Y


def display_toy_data(X, Y):
    lab_0_idx = np.where(Y == 0)
    lab_1_idx = np.where(Y == 1)
    plt.scatter(X[lab_0_idx, 0], X[lab_0_idx, 1], marker='.')
    plt.scatter(X[lab_1_idx, 0], X[lab_1_idx, 1], marker='+')
    plt.show()


def simulate_drift_stream(r_seed, original_stream, delta_list):
    np.random.seed(r_seed)
    delta_size = len(delta_list)
    stream_x, stream_y = original_stream[:, :-1], original_stream[:, -1]
    stream_splitter = KFold(n_splits=delta_size, random_state=None, shuffle=True)

    b_id = 0
    for _, drift_batch_index in stream_splitter.split(stream_x):
        drift_mat = (stream_x[drift_batch_index].max(0) - stream_x[drift_batch_index].min(0)) * delta_list[b_id]
        drift_mat = np.tile(drift_mat, [stream_x[drift_batch_index].shape[0], 1])
        stream_x[drift_batch_index] = stream_x[drift_batch_index] + drift_mat
        b_id += 1
    return stream_x, stream_y


if __name__ == "__main__":
    path = "Datasets/"
    dataset_name_list = DriftDataset._datasetDict.keys()
    for ds_name in dataset_name_list:
        data = DriftDataset(path, ds_name)
        print('Testing: Load Dataset', ds_name)
        print(data.np_data.shape)
