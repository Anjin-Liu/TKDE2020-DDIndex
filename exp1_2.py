# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:34:37 2020

@author: Anjin Liu
@email: Anjin.Liu@uts.edu.au
"""
import numpy as np
import pandas as pd
from skmultiflow.drift_detection import ADWIN
from skmultiflow.drift_detection import DDM
from skmultiflow.drift_detection import HDDM_A
from skmultiflow.drift_detection import HDDM_W
from skmultiflow.drift_detection import PageHinkley
from tqdm import tqdm

import detection_delay_index as ddi
# parameter not searchable in skmultiflow at current version
from external_ddm.eddm import EDDM
from external_ddm.kswin import KSWIN


def exp1_demo_default_param(e_grid):
    ddm_list = [ADWIN, DDM, EDDM, HDDM_A, HDDM_W, PageHinkley, KSWIN]

    valid_size_list = [0, 30, 80, 130, 180]
    ddm_param_list = [{} for i in range(len(ddm_list))]
    exp1_demo_result_list = []

    with tqdm(total=len(valid_size_list) * len(ddm_list)) as pbar:
        for ddm_idx, ddm in enumerate(ddm_list):
            for v_s in valid_size_list:
                df_result = ddi.ddi_bernoulli_error_grid_eval(ddm, ddm_param_list[ddm_idx], v_s, e_grid, r_seed=0)
                exp1_demo_result_list.append(df_result)
                pbar.update(1)

    exp_result_df = pd.concat(exp1_demo_result_list)
    ddi.ddi_bernoulli_error_grid_plot(exp_result_df, '_exp1_default')


def exp2_align_param_by_ddi(error_ini, v_size, target_ddi):
    np.random.seed(0)
    ddm_list = [ADWIN, DDM, EDDM, HDDM_A, HDDM_W, PageHinkley, KSWIN]
    ddm_param_name_list = ['delta', 'out_control_level', 'FDDM_OUTCONTROL',
                           'drift_confidence', 'drift_confidence', 'threshold', 'alpha']
    ddm_param_value_min_list = [10, 0.001, 10, 1, 1, 0.001, 1]
    ddm_param_value_max_list = [0.0001, 10, 0.001, 0.001, 0.001, 100, 0.001]
    ddm_param_list = []

    param_align_settings = {
        'ddm': None,
        'param_name': '',
        'desired_ddi': target_ddi,
        'param_value_min_sensitive': None,
        'param_value_max_sensitive': None,
        'param_detail_level': 0.001,
        'ini_error': error_ini,
        'valid_size': v_size,
        'r_seed': 0
    }

    with tqdm(total=len(ddm_list)) as pbar:
        for ddm_idx, ddm in enumerate(ddm_list):
            param_align_settings['ddm'] = ddm
            param_align_settings['param_name'] = ddm_param_name_list[ddm_idx]
            param_align_settings['param_value_min_sensitive'] = ddm_param_value_min_list[ddm_idx]
            param_align_settings['param_value_max_sensitive'] = ddm_param_value_max_list[ddm_idx]
            param_ddm = ddi.ddi_bernoulli_drift_param_alignment_avg(**param_align_settings)
            ddm_param_list.append(param_ddm)
            pbar.update(1)

    return ddm_param_list


def exp2_demo_aligned_param(e_grid, ddm_param_list, plot_file_suffix):
    ddm_list = [ADWIN, DDM, EDDM, HDDM_A, HDDM_W, PageHinkley, KSWIN]
    valid_size_list = [80]
    exp2_demo_result_list = []

    with tqdm(total=len(valid_size_list) * len(ddm_list)) as pbar:
        for ddm_idx, ddm in enumerate(ddm_list):
            for v_s in valid_size_list:
                df_result = ddi.ddi_bernoulli_error_grid_eval(ddm, ddm_param_list[ddm_idx], v_s, e_grid, r_seed=0)
                exp2_demo_result_list.append(df_result)
                pbar.update(1)

    exp_result_df = pd.concat(exp2_demo_result_list)
    ddi.ddi_bernoulli_error_grid_plot(exp_result_df, plot_file_suffix)


if __name__ == '__main__':
    '''
    error_ini = 0.15
    error_grid = np.arange(error_ini, 1, 0.1)

    exp1_demo_default_param(error_grid)
    
    '''
    error_ini = 0.1
    error_grid = np.arange(error_ini, 1, 0.1)
    v_size = 80
    target_ddi_list = [0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
    target_ddi_list = [0.99, 0.95, 0.90, 0.80]
    for target_ddi in target_ddi_list:
        print(target_ddi)
        param_list = exp2_align_param_by_ddi(error_ini, v_size, target_ddi)
        with open("param_{:.2}_{:.2}.txt".format(target_ddi, error_ini), 'w') as output:
            for param in param_list:
                output.write(str(param) + '\n')
        # exp2_demo_aligned_param(error_grid, param_list, '_exp2_aligned_{:.2}'.format(target_ddi))
    

