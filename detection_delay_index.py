# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:34:37 2020

@author: Anjin Liu
@email: Anjin.Liu@uts.edu.au
"""
import datetime
from multiprocessing import Manager
from multiprocessing import Process

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skmultiflow.drift_detection import ADWIN

DDI_NUM_RUN = 200
DDI_NUM_TEST = 200
DDI_NUM_PROCESS = 20


def ddi_bernoulli(ddm, ddm_param, ini_error, cur_error, v_size, r_seed, return_list):
    '''

    @param ddm:
    @param ddm_param:
    @param ini_error:
    @param cur_error:
    @param v_size:
    @param r_seed:
    @param return_list: Paralla computing parameter, used to store the result of each process
    @return:
    '''
    np.random.seed(r_seed)
    detection_t_avg = 0
    for r_id in range(DDI_NUM_RUN):
        pred_valid_error = []
        if v_size > 0:
            pred_valid_error = np.random.binomial(1, ini_error, v_size)
        pred_test_error = np.random.binomial(1, cur_error, DDI_NUM_TEST)
        dd_method = ddm(**ddm_param)

        for valid_e in pred_valid_error:
            dd_method.add_element(valid_e)

        detection_t = DDI_NUM_TEST
        for t, test_e in enumerate(pred_test_error):
            dd_method.add_element(test_e)
            if dd_method.detected_change():
                detection_t = t + 1
                break
        detection_t_avg += detection_t
    detection_t_avg = detection_t_avg / DDI_NUM_TEST / DDI_NUM_RUN
    return_list.append(detection_t_avg)


def ddi_bernoulli_multi_process(ddm, ddm_param, ini_error, cur_error, v_size, r_seed):
    '''
    Multi thread running to estimate the drift detection delay index in terms of given
    drift detection method (ddm), parameters (ddm_param), initial error (ini_error),
    current error (cur_error), validation size (v_size) and random seed (r_seed=0)
    ==================================================
    @param ddm: drift detection method
    @param ddm_param: drift detection parameters
    @param ini_error: initial error rate
    @param cur_error: current error rate
    @param v_size: validation size
    @param r_seed: random seed
    @return: drift detection delay index (DDI)
    '''
    process_list = []
    manager = Manager()
    return_list = manager.list()
    for i in range(DDI_NUM_PROCESS):
        process_i = Process(target=ddi_bernoulli,
                            args=(ddm, ddm_param, ini_error, cur_error, v_size, r_seed + i * 10000, return_list))
        process_list.append(process_i)
        process_i.start()

    for process_i in process_list:
        process_i.join()

    return np.mean(return_list)


def ddi_bernoulli_drift_param_alignment(ddm, param_name, desired_ddi,
                                        param_value_min_sensitive, param_value_max_sensitive, param_detail_level,
                                        ini_error, valid_size, param_min_ddi_t1=None, param_max_ddi_t1=None,
                                        r_seed=0):
    '''
    @param ddm:
    @param param_name:
    @param desired_t1:
    @param param_value_min_sensitive:
    @param param_value_max_sensitive:
    @param param_detail_level:
    @param ini_error:
    @param valid_size:
    @param param_min_ddi_t1:
    @param param_max_ddi_t1:
    @param r_seed:
    @return:
    '''
    param_item_min = {param_name: param_value_min_sensitive}
    param_item_max = {param_name: param_value_max_sensitive}
    param_best, ddi_t1_best = None, None

    if param_min_ddi_t1 is None:
        # ===================================#
        # iteration converage condition 0.1  #
        # ===================================#
        param_min_ddi_t1 = ddi_bernoulli_multi_process(ddm, param_item_min, ini_error, ini_error, valid_size, r_seed)
        if param_min_ddi_t1 == desired_ddi:
            print('Condition 0: Desired DDI_T1 found at param_value_min')
            return param_item_min, param_min_ddi_t1

        # ===================================#
        # iteration converage condition 0.2  #
        # ===================================#
        param_max_ddi_t1 = ddi_bernoulli_multi_process(ddm, param_item_max, ini_error, ini_error, valid_size, r_seed)
        if param_max_ddi_t1 == desired_ddi:
            print('Condition 0: Desired DDI_T1 found at param_value_max')
            return param_item_max, param_max_ddi_t1

        # =================================#
        # iteration converage condition 1. #
        # non param in given range satisfy #
        # =================================#
        if param_min_ddi_t1 < desired_ddi and param_max_ddi_t1 < desired_ddi:
            print('Condition 1: Desired DDI_T1 out of param_value (min, max) range')
            return param_item_max, param_max_ddi_t1

    param_value_descending = False
    if param_value_max_sensitive < param_value_min_sensitive:
        param_value_descending = True

    # =================================#
    # iteration converage condition 2. #
    # param detail level reached       #
    # closet param setting found with  #
    # given param_detail_level         #
    # =================================#
    if param_value_descending:
        max_min_diff = param_value_min_sensitive - param_value_max_sensitive
    else:
        max_min_diff = param_value_max_sensitive - param_value_min_sensitive
    if param_detail_level >= max_min_diff:
        print('Condition 2: Closet param setting found with given param_detail_level')
        if param_min_ddi_t1 > desired_ddi and param_max_ddi_t1 > desired_ddi:
            if param_min_ddi_t1 - desired_ddi < param_max_ddi_t1 - desired_ddi:
                return param_item_min, param_min_ddi_t1
            else:
                return param_item_max, param_max_ddi_t1
        else:
            if param_max_ddi_t1 - desired_ddi < 0:
                return param_item_min, param_min_ddi_t1
            else:
                return param_item_max, param_max_ddi_t1
    else:
        if param_value_descending:
            param_value_mid = (param_value_min_sensitive - param_value_max_sensitive) / 2 + param_value_max_sensitive
        else:
            param_value_mid = (param_value_max_sensitive - param_value_min_sensitive) / 2 + param_value_min_sensitive

        param_item_mid = {param_name: param_value_mid}
        param_mid_ddi_t1 = ddi_bernoulli_multi_process(ddm, param_item_mid, ini_error, ini_error, valid_size, r_seed)
        if param_mid_ddi_t1 == desired_ddi:
            return param_item_mid, param_mid_ddi_t1

        if param_mid_ddi_t1 > desired_ddi:
            param_best, ddi_t1_best = ddi_bernoulli_drift_param_alignment(ddm, param_name, desired_ddi,
                                                                          param_value_min_sensitive, param_value_mid,
                                                                          param_detail_level,
                                                                          ini_error, valid_size, param_min_ddi_t1,
                                                                          param_mid_ddi_t1, r_seed + 1)
        else:
            param_best, ddi_t1_best = ddi_bernoulli_drift_param_alignment(ddm, param_name, desired_ddi,
                                                                          param_value_mid, param_value_max_sensitive,
                                                                          param_detail_level,
                                                                          ini_error, valid_size, param_mid_ddi_t1,
                                                                          param_max_ddi_t1, r_seed + 1)

    return param_best, ddi_t1_best


def ddi_bernoulli_drift_param_alignment_avg(ddm, param_name, desired_ddi,
                                            param_value_min_sensitive, param_value_max_sensitive, param_detail_level,
                                            ini_error, valid_size, r_seed):
    ddi_avg = []
    for ddi_run_idx in range(10):
        ddi_param, ddi_t1_best = ddi_bernoulli_drift_param_alignment(ddm, param_name, desired_ddi,
                                                                     param_value_min_sensitive,
                                                                     param_value_max_sensitive,
                                                                     param_detail_level,
                                                                     ini_error, valid_size,
                                                                     r_seed=ddi_run_idx * 1000 + r_seed)
        ddi_avg.append(ddi_param[param_name])
    return {param_name: np.mean(ddi_avg)}


def ddi_bernoulli_error_grid_eval(ddm, ddm_param, v_size, error_grid, r_seed=0):

    np.random.seed(r_seed)
    df_result = pd.DataFrame(columns=['ErrorGrid', 'ValidSize', 'DetectionMethod', 'DetectionDelayIndex'])
    result_counter = 0

    for e_idx, e_rate in enumerate(error_grid):
        ddi = ddi_bernoulli_multi_process(ddm, ddm_param, error_grid[0], e_rate, v_size, r_seed)
        df_result.loc[result_counter] = ['{:.2}-{:.2}'.format(error_grid[0], e_rate), v_size, ddm.__name__, ddi]
        result_counter += 1

    df_result.reset_index(drop=True, inplace=True)

    return df_result


def ddi_bernoulli_error_grid_plot(ddi_df, plot_file_suffix):

    if ddi_df.shape[0] == 0:
        return None

    ddm_list = ddi_df.DetectionMethod.unique()
    valid_size_list = ddi_df.ValidSize.unique()
    error_list_raw = ddi_df.ErrorGrid.unique()
    error_list = []
    for e_raw in error_list_raw:
        error_list.append(float(e_raw.split('-')[1]))
    error_list.sort()

    for v_size in valid_size_list:

        df_filter = ddi_df[ddi_df.ValidSize == v_size]
        df_plot = df_filter[df_filter.DetectionMethod == ddm_list[0]]
        df_plot = df_plot[['ErrorGrid', 'DetectionDelayIndex']]
        df_plot = df_plot.sort_values(by=['ErrorGrid'])
        df_plot.columns = ['ErrorGrid', ddm_list[0]]
        df_plot['ErrorGrid'] = error_list

        for ddm in ddm_list[1:]:
            df_plot_filter = df_filter[df_filter.DetectionMethod == ddm]
            df_plot_filter = df_plot_filter.sort_values(by=['ErrorGrid'])
            df_plot_filter = df_plot_filter['DetectionDelayIndex']
            df_plot[ddm] = df_plot_filter.values

        t1_scatter_y = df_plot.iloc[0, 1:].values
        t1_scatter_x = np.zeros(t1_scatter_y.shape) + error_list[0]

        t2_scatter_y = df_plot.iloc[1:, 1:].values
        t2_scatter_x = np.array(error_list[1:])
        t2_scatter_x = np.tile(t2_scatter_x, [ddm_list.shape[0], 1]).T

        df_plot.set_index('ErrorGrid', drop=True, inplace=True)
        df_plot.plot(legend=False)
        plt.ylim([0, 1.1])
        plt.ylabel('DetectionDelayIndex')
        plt.title('Detection Delay Index with ValidSzie=' + '{}'.format(v_size))
        plt.scatter(t1_scatter_x, t1_scatter_y, marker='X', c='r', s=20, zorder=100)
        plt.scatter(t2_scatter_x, t2_scatter_y, marker='o', c='midnightblue', s=20, zorder=100)
        if v_size == valid_size_list[0]:
            plt.legend(ddm_list.tolist() + ['TI-DDI', 'TII-DDI'], ncol=3)
        plt.savefig('img/DDI_{}'.format(v_size) + plot_file_suffix + '.pdf', bbox_inches='tight')
        plt.show()


def load_ddi_t1_table(table_file_name):
    table_df = pd.read_csv(table_file_name)
    return table_df


def get_param_from_ddi_t1_table(ddm_name, target_ddi_t1, ini_error, v_size, ddi_t1_table_df, include_warn_param=True):
    if ddi_t1_table_df is None:
        return {}

    # TODO: for experiment only
    # should be removed after acceptence
    #    if v_size == 0:
    #        v_size = 30

    target_row = ddi_t1_table_df[ddi_t1_table_df.ValidSize == v_size]
    if target_row.shape[0] == 0:
        print(
            'get_param_from_ddi_t1_table get ValidSize {} not found, return default drift detection parameters'.format(
                v_size))
        return {}

    tar_ddi_list = target_row.Target_DDI_T1.unique()
    ini_err_list = target_row.IniError.unique()

    closest_tar_ddi = min(tar_ddi_list.tolist(), key=lambda x: abs(x - target_ddi_t1))
    closest_ini_err = min(ini_err_list.tolist(), key=lambda x: abs(x - ini_error))

    target_row = target_row[target_row.DriftDetectionMethod == ddm_name]
    target_row = target_row[target_row.Target_DDI_T1 == closest_tar_ddi]
    target_row = target_row[target_row.IniError == closest_ini_err]

    param_name, param_value = target_row['ParamName'].values[0], target_row['ParamValue'].values[0]
    ddm_param = {param_name: param_value}

    if include_warn_param:
        if ddm_name == 'DDM':
            ddm_param['warning_level'] = param_value
        if ddm_name == 'EDDM':
            ddm_param['FDDM_WARNING'] = param_value
        if ddm_name == 'HDDM_A':
            ddm_param['warning_confidence'] = param_value
        if ddm_name == 'HDDM_W':
            ddm_param['warning_confidence'] = param_value

    return ddm_param


if __name__ == '__main__':
    start_t = datetime.datetime.now()
    # ddi_0_adwin = ddi_bernoulli_multi_process(ADWIN, {}, 0.15, 0.15, 80, 0)
    # print(ddi_0_adwin)
    adwin_random_list = []

    for i in range(10, 11):
        adwin_ddi_param = ddi_bernoulli_drift_param_alignment_avg(ADWIN, 'delta', 0.95,
                                                                  10, 0.001, 0.01,
                                                                  0.15, 80, r_seed=i)
        adwin_random_list.append(adwin_ddi_param)

    for item in adwin_random_list:
        print(item)

    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print('total time: {:.2f}'.format(elapsed_sec))
