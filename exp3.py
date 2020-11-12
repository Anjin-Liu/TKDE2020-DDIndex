# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:34:37 2020

@author: Anjin Liu
@email: Anjin.Liu@uts.edu.au
True"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm

import data_handler as dh
import stream_learning_lib as sl_lib


def exp3_toy_stream_with_simulated_drift():
    generator_list = [dh.generator_moon_batch, dh.generator_circ_batch, dh.generator_blob_batch]
    for generator in generator_list:
        toy_stream_param = {
            'r_seed': 1,
            'batch_generator': generator,
            'delta_list': [0, 0.4, 1.2],
            'n_samples_per_batch': 100,
            'n_noisy_dim': 0,
            'noise_ratio': 0,
            'plot_stream_2d': True}
        dh.get_toy_stream(**toy_stream_param)


def exp3_toy_stream_eval(learner, learner_param, num_eval_per_dataset, target_ddi_t1, concept_size):
    # =====================#
    # toy stream settings #
    # =====================#
    generator_list = [dh.generator_moon_batch, dh.generator_circ_batch, dh.generator_blob_batch]
    delta_list = [0, 0, 0.2, 0.4, 0.6, 0.8, 1]
    delta_list = np.cumsum(delta_list)
    noisy_dim = 18
    noise_rat = 0.1
    toy_stream_param = {'r_seed': None,
                        'batch_generator': None,
                        'delta_list': delta_list,
                        'n_samples_per_batch': concept_size,
                        'n_noisy_dim': noisy_dim,
                        'noise_ratio': noise_rat,
                        'plot_stream_2d': False}

    # =====================================#
    # stream learning evaluation settings #
    # =====================================#
    train_size_min = 200
    df_result_list = []

    tqdm_total = len(generator_list) * num_eval_per_dataset
    with tqdm(total=tqdm_total) as pbar:
        for generator in generator_list:
            for eval_idx in range(num_eval_per_dataset):
                toy_stream_param['r_seed'] = eval_idx + 1000
                toy_stream_param['batch_generator'] = generator
                toy_stream_X, toy_stream_Y = dh.get_toy_stream(**toy_stream_param)
                df_result = sl_lib.eval_stream_on_all_skmultiflow_ddm(generator.__name__, toy_stream_X, toy_stream_Y,
                                                                      train_size_min, learner, learner_param,
                                                                      target_ddi_t1)

                df_result_list.append(df_result)
                pbar.update(1)
    df_result_all = pd.concat(df_result_list, axis=0)
    df_result_all.reset_index(drop=True, inplace=True)

    return df_result_all


if __name__ == '__main__':
   
   exp3_toy_stream_with_simulated_drift()
   
    # param_file_path_temp = 'param_{}.txt'.format(int(target_t1*100))
    # param_dict_merged = load_param(param_file_path_temp)'
    # '''
#     target_t1_list = [0.99, 0.95, 0.9, 0.85, 0.8]
#     concept_size_list = [300, 800, 1300, 1800, 2300]
#     target_t1_list = [0.99, 0.95, 0.9, 0.85, 0.8]
#     concept_size_list = [2300]
#     merged_result = []
#     for t1 in target_t1_list:
#         for c_size in concept_size_list:
#             df_result = exp3_toy_stream_eval(GaussianNB, {}, num_eval_per_dataset=15,
#                                              target_ddi_t1=t1, concept_size=c_size)
#             df_result['TargetDDI'] = t1
#             df_result['ConceptSize'] = c_size
#             merged_result.append(df_result)
#     merged_result_df = pd.concat(merged_result)
#     merged_result_df.to_csv('exp3_result_test_26sep_2300.csv')
    # '''
