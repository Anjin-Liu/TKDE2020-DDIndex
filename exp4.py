# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:34:37 2020

@author: Anjin Liu
@email: Anjin.Liu@uts.edu.au
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm

import data_handler as dh
import stream_learning_lib as sl_lib


def exp4_SyntData_SyntDrift(learner, learner_param, target_ddi_t1):

    path = "Datasets/"
    dataset_name_list = ['SEAa0', 'SEAg', 'HYPi', 'AGRa', 'AGRg', 'LEDa', 'LEDg', 'RBFi', 'RBFr', 'RTGn']
    dataset_name_list = ['AGRa', 'RTGn']
    dataset_name_list = ['SEAa0', 'SEAg', 'HYPi', 'AGRg', 'LEDa', 'LEDg', 'RBFi', 'RBFr']

    # =====================================#
    # stream learning evaluation settings #
    # =====================================#
    train_size_min = 200

    df_result_list = []
    for dataset_name in tqdm(dataset_name_list):
        stream = dh.DriftDataset(path, dataset_name).np_data
        df_result = sl_lib.eval_stream_on_all_skmultiflow_ddm(dataset_name, stream[:, :-1], stream[:, -1],
                                                              train_size_min, learner, learner_param, target_ddi_t1)
        df_result_list.append(df_result)

    df_result_all = pd.concat(df_result_list, axis=0)
    df_result_all.reset_index(drop=True, inplace=True)

    return df_result_all


def exp4_RealData_SyntDrift(learner, learner_param, target_ddi_t1):

    path = "Datasets/"
    dataset_name_list = ['use1', 'use2', 'wdbc', 'glas', 'iono', 'soyb', 'iris']

    # =====================================#
    # stream learning evaluation settings #
    # =====================================#
    train_size_min = 200

    df_result_list = []
    for dataset_name in tqdm(dataset_name_list):
        stream = dh.DriftDataset(path, dataset_name).np_data
        df_result = sl_lib.eval_stream_on_all_skmultiflow_ddm(dataset_name, stream[:, :-1], stream[:, -1],
                                                              train_size_min, learner, learner_param, target_ddi_t1)
        df_result_list.append(df_result)

    df_result_all = pd.concat(df_result_list, axis=0)
    df_result_all.reset_index(drop=True, inplace=True)

    return df_result_all


def exp4_RealData_UnknDrift(learner, learner_param, target_ddi_t1):

    path = "Datasets/"
    dataset_name_list = ['elec', 'weat', 'spam', 'airl', 'covt-binary', 'poke-binary']
    dataset_name_list = ['elec', 'weat', 'spam', 'airl']
    # =====================================#
    # stream learning evaluation settings #
    # =====================================#
    train_size_min = 200

    df_result_list = []
    for dataset_name in tqdm(dataset_name_list):
        stream = dh.DriftDataset(path, dataset_name).np_data
        df_result = sl_lib.eval_stream_on_all_skmultiflow_ddm(dataset_name, stream[:, :-1], stream[:, -1],
                                                              train_size_min, learner, learner_param, target_ddi_t1)
        df_result_list.append(df_result)

    df_result_all = pd.concat(df_result_list, axis=0)
    df_result_all.reset_index(drop=True, inplace=True)

    return df_result_all



if __name__ == '__main__':

    # target_t1_list = [0.99, 0.95, 0.9, 0.85, 0.8]
    # merged_result = []
    # for t1 in target_t1_list:
    #     df_result = exp4_SyntData_SyntDrift(GaussianNB, {}, target_ddi_t1=t1)
    #     df_result['TargetDDI'] = t1
    #     merged_result.append(df_result)
    # merged_result_df = pd.concat(merged_result)
    # merged_result_df.to_csv('exp4_result_30sep_2.csv')
    import sys
    sys.setrecursionlimit(5000)
    target_t1_list = [0.99, 0.95, 0.9, 0.85, 0.8]
    target_t1_list = [0.99, 0.95, 0.9]
    merged_result = []
    for t1 in target_t1_list:
        df_result = exp4_RealData_UnknDrift(GaussianNB, {}, target_ddi_t1=t1)
        df_result['TargetDDI'] = t1
        merged_result.append(df_result)
    merged_result_df = pd.concat(merged_result)
    merged_result_df.to_csv('exp4_result_07Oct_RealData_UnknDrift_0.9_0.99.csv')
