# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:34:37 2020

@author: Anjin Liu
@email: Anjin.Liu@uts.edu.au
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from skmultiflow.drift_detection import ADWIN
from skmultiflow.drift_detection import DDM
from skmultiflow.drift_detection import HDDM_A
from skmultiflow.drift_detection import HDDM_W
from skmultiflow.drift_detection import PageHinkley
from tqdm import tqdm

import detection_delay_index as ddi
from external_ddm.eddm import EDDM
from external_ddm.kswin import KSWIN


def eval_stream_on_all_skmultiflow_ddm(dataset_name, stream_X, stream_Y, train_size_min,
                                       learner, learner_param, target_ddi_t1):
    # ================================================= #
    # Initial drift detection algorithms for evaluation #
    # ================================================= #
    ddm_list = [ADWIN, DDM, EDDM, HDDM_A, HDDM_W, PageHinkley, KSWIN]
    ddm_name_list = []
    for ddm in ddm_list:
        ddm_name_list.append(ddm.__name__)

    # ================================================= #
    # Initial drift detection parameters for evaluation #
    # ================================================= #
    # setting the warning-level threshold equal to drift-level threshold
    # so no warning buffer will be triggered
    ddm_param_table_df = ddi.load_ddi_t1_table('DDI_Table/ddi_table_25Sep.csv')
    ddm_param_table_df_default_no_warning = ddi.load_ddi_t1_table('DDI_Table/table_ddi_t1_default_no_warning.csv')

    # ============================== #
    # stream learning basic settings #
    # ============================== #
    stream_learning_param = {'stream_X': stream_X,
                             'stream_Y': stream_Y,
                             'dd_method': None,
                             'ddm_param_table': None,
                             'learner': learner,
                             'learner_param': learner_param,
                             'is_partial_fit': True,
                             'train_size_min': train_size_min,
                             'train_buff_dict': {'X': np.array([]), 'Y': np.array([])},
                             'target_ddi_t1': target_ddi_t1,
                             'valid_size': 0}

    # ================ #
    # result container #
    # ================ #
    # ndd: number of detected drift
    base_acc_cols = ['Dataset', 'DetectionMethod', 'acc_slide_chunk', 'acc_std_default', 'acc_val_single',
                     'acc_val_align', 'ndd_std_default', 'ndd_std_single', 'ndd_std_align']
    df_result = pd.DataFrame(columns=base_acc_cols)

    # =============================================================== #
    # Baseline: Evaluate Stream on chunk-based stream learning method #
    # =============================================================== #
    Y_pred = stream_learning_prequential_chunk(stream_X, stream_Y, train_size_min, learner, learner_param)
    acc_slide_chunk = accuracy_score(stream_Y, Y_pred, normalize=True)

    for ddm_idx, ddm in enumerate(ddm_list):
        # tqdm.write('evaluating - ' + ddm.__name__.ljust(12) + ' - ' + dataset_name)
        # ============================================= #
        # Ablation 1. default drift detection paramters #
        # default parameter                             #
        # ============================================= #
        stream_learning_param['dd_method'] = ddm
        stream_learning_param['ddm_param_table'] = None
        stream_learning_param['valid_size'] = 0
        # standard warn-drift level data stream learning
        Y_pred, ndd_std_default = stream_learning_prequential(**stream_learning_param)
        acc_std_default = accuracy_score(stream_Y, Y_pred, normalize=True)

        # =================================================================== #
        # Ablation 2. default drift detection parameters without warning-level #
        # standard drift level data stream learning                           #
        # For ADWIN, PageHinkley, KSWIN, there is no warning windows          #
        # single parameter
        # =================================================================== #
        stream_learning_param['dd_method'] = ddm
        stream_learning_param['ddm_param_table'] = ddm_param_table_df_default_no_warning
        stream_learning_param['valid_size'] = 80
        Y_pred, ndd_val_single = stream_learning_prequential(**stream_learning_param)
        acc_val_single = accuracy_score(stream_Y, Y_pred, normalize=True)

        # ======================================== #
        # Ablation 3. introducing local validation #
        # with default drift detection parameters  #
        # optimized parameter
        # ======================================== #
        stream_learning_param['dd_method'] = ddm
        stream_learning_param['ddm_param_table'] = ddm_param_table_df
        stream_learning_param['valid_size'] = 80
        Y_pred, ndd_val_align = stream_learning_prequential(**stream_learning_param)
        acc_val_align = accuracy_score(stream_Y, Y_pred, normalize=True)

        df_row_bas = [dataset_name, ddm.__name__, acc_slide_chunk]
        df_row_acc_base = [acc_std_default, acc_val_single, acc_val_align]
        df_row_ndd_base = [ndd_std_default, ndd_val_single, ndd_val_align]

        df_row = []
        df_row += df_row_bas
        df_row += df_row_acc_base
        df_row += df_row_ndd_base
        df_result.loc[ddm_idx] = df_row

    return df_result


def eval_stream_on_all_skmultiflow_ddm_backup(dataset_name, stream_X, stream_Y, train_size_min,
                                              learner, learner_param, valid_size_list, target_ddi_t1):
    # ================================================= #
    # Initial drift detection algorithms for evaluation #
    # ================================================= #
    ddm_list = [ADWIN, DDM, EDDM, HDDM_A, HDDM_W, PageHinkley, KSWIN]
    ddm_name_list = []
    for ddm in ddm_list:
        ddm_name_list.append(ddm.__name__)

    # ================================================= #
    # Initial drift detection parameters for evaluation #
    # ================================================= #
    # setting the warning-level threshold equal to drift-level threshold
    # so no warning buffer will be triggered
    ddm_param_table_df = ddi.load_ddi_t1_table('DDI_Table/table_ddi_aligned.csv')
    ddm_param_table_df_default_no_warning = ddi.load_ddi_t1_table('DDI_Table/table_ddi_t1_default_no_warning.csv')
    # ============================== #
    # stream learning basic settings #
    # ============================== #
    stream_learning_param = {'stream_X': stream_X,
                             'stream_Y': stream_Y,
                             'dd_method': None,
                             'ddm_param_table': None,
                             'learner': learner,
                             'learner_param': learner_param,
                             'is_partial_fit': True,
                             'train_size_min': train_size_min,
                             'train_buff_dict': {'X': np.array([]), 'Y': np.array([])},
                             'target_ddi_t1': target_ddi_t1,
                             'valid_size': 0}

    # ================ #
    # result container #
    # ================ #
    base_acc_cols = ['Dataset', 'DetectionMethod', 'acc_slide_chunk', 'acc_std_default', 'acc_std_single',
                     'acc_std_optimi']
    dyna_acc_cols = ['acc_val{}_default'.format(v_size) for v_size in valid_size_list]
    dyna_acc_cols += ['acc_val{}_single'.format(v_size) for v_size in valid_size_list]
    dyna_acc_cols += ['acc_val{}_optimi'.format(v_size) for v_size in valid_size_list]
    # ndd: number of detected drift
    base_ndd_cols = ['ndd_std_default', 'ndd_std_single', 'ndd_std_optimi']
    dyna_ndd_cols = ['ndd_val{}_default'.format(v_size) for v_size in valid_size_list]
    dyna_ndd_cols += ['ndd_val{}_single'.format(v_size) for v_size in valid_size_list]
    dyna_ndd_cols += ['ndd_val{}_optimi'.format(v_size) for v_size in valid_size_list]
    key_cols = base_acc_cols + dyna_acc_cols + base_ndd_cols + dyna_ndd_cols
    df_result = pd.DataFrame(columns=key_cols)

    # =============================================================== #
    # Baseline: Evaluate Stream on chunk-based stream learning method #
    # =============================================================== #
    Y_pred = stream_learning_prequential_chunk(stream_X, stream_Y, train_size_min, learner, learner_param)
    acc_slide_chunk = accuracy_score(stream_Y, Y_pred, normalize=True)

    for ddm_idx, ddm in enumerate(ddm_list):

        tqdm.write('evaluating - ' + ddm.__name__.ljust(12) + ' - ' + dataset_name)
        # ============================================= #
        # Ablation 1. default drift detection paramters #
        # default parameter                             #
        # ============================================= #
        stream_learning_param['dd_method'] = ddm
        stream_learning_param['ddm_param_table'] = None
        stream_learning_param['valid_size'] = 0
        # standard warn-drift level data stream learning
        Y_pred, ndd_std_default = stream_learning_prequential(**stream_learning_param)
        acc_std_default = accuracy_score(stream_Y, Y_pred, normalize=True)

        # =================================================================== #
        # Ablation 2. default drift detection parameters without warning-level #
        # standard drift level data stream learning                           #
        # For ADWIN, PageHinkley, KSWIN, there is no warning windows          #
        # single parameter
        # =================================================================== #
        if ddm.__name__ == 'ADWIN' or ddm.__name__ == 'PageHinkley' or ddm.__name__ == 'KSWIN':
            acc_std_single = acc_std_default
            ndd_std_single = ndd_std_default
        else:
            stream_learning_param['ddm_param_table'] = ddm_param_table_df_default_no_warning
            stream_learning_param['valid_size'] = 0
            Y_pred, ndd_std_single = stream_learning_prequential(**stream_learning_param)
            acc_std_single = accuracy_score(stream_Y, Y_pred, normalize=True)

        # ========================================= #
        # Ablation 3. introducing local validation #
        # with default drift detection parameters   #
        # optimized parameter
        # ========================================= #
        stream_learning_param['ddm_param_table'] = ddm_param_table_df
        stream_learning_param['valid_size'] = 0
        Y_pred, ndd_std_optimi = stream_learning_prequential(**stream_learning_param)
        acc_std_optimi = accuracy_score(stream_Y, Y_pred, normalize=True)

        df_row_bas = [dataset_name, ddm.__name__, acc_slide_chunk]
        df_row_acc_base = [acc_std_default, acc_std_single, acc_std_optimi]
        df_row_ndd_base = [ndd_std_default, ndd_std_single, ndd_std_optimi]
        df_row_acc_val_default = []
        df_row_acc_val_single = []
        df_row_acc_val_optimi = []
        df_row_ndd_val_default = []
        df_row_ndd_val_single = []
        df_row_ndd_val_optimi = []

        for valid_size in valid_size_list:
            # ========================================= #
            # Ablation 4. introducing local validataion #
            # default parameter + local validation
            # ========================================= #
            stream_learning_param['ddm_param_table'] = None
            stream_learning_param['valid_size'] = valid_size
            Y_pred, ndd_val_default = stream_learning_prequential(**stream_learning_param)
            acc_val_default = accuracy_score(stream_Y, Y_pred, normalize=True)

            # ========================================= #
            # Ablation 5. introducing local validataion #
            # single parameter + local validation
            # ========================================= #
            stream_learning_param['ddm_param_table'] = ddm_param_table_df_default_no_warning
            stream_learning_param['valid_size'] = valid_size
            Y_pred, ndd_val_single = stream_learning_prequential(**stream_learning_param)
            acc_val_single = accuracy_score(stream_Y, Y_pred, normalize=True)

            # ========================================= #
            # Ablation 6. introducing local validataion #
            # and optimized drift detection parameters  #
            # optimized parameter + local validation
            # ========================================= #
            stream_learning_param['ddm_param_table'] = ddm_param_table_df
            stream_learning_param['valid_size'] = valid_size
            Y_pred, ndd_val_optimi = stream_learning_prequential(**stream_learning_param)
            acc_val_optimi = accuracy_score(stream_Y, Y_pred, normalize=True)

            df_row_acc_val_default.append(acc_val_default)
            df_row_acc_val_single.append(acc_val_single)
            df_row_acc_val_optimi.append(acc_val_optimi)
            df_row_ndd_val_default.append(ndd_val_default)
            df_row_ndd_val_single.append(ndd_val_single)
            df_row_ndd_val_optimi.append(ndd_val_optimi)

        df_row = []
        df_row += df_row_bas
        df_row += df_row_acc_base
        df_row += df_row_acc_val_default
        df_row += df_row_acc_val_single
        df_row += df_row_acc_val_optimi
        df_row += df_row_ndd_base
        df_row += df_row_ndd_val_default
        df_row += df_row_ndd_val_single
        df_row += df_row_ndd_val_optimi
        df_result.loc[ddm_idx] = df_row

        df_result[base_ndd_cols + dyna_ndd_cols] = df_result[base_ndd_cols + dyna_ndd_cols].astype(int)

    return df_result


def initial_drift_detection_algorihtm(trai_buff_X, trai_buff_Y, learner, learner_param, dd_method, ddm_param_table,
                                      target_ddi_t1, valid_size):
    valid_error = 0
    if valid_size == 0:
        dd_method_param = ddi.get_param_from_ddi_t1_table(dd_method.__name__, target_ddi_t1, valid_error, valid_size,
                                                          ddm_param_table)
        detection_alg = dd_method(**dd_method_param)
        return detection_alg

    num_validation = 5
    num_validation_replacement = 6
    detection_alg = None
    # ========================#
    # kfold with replacement #
    # ========================#
    valid_Y = np.array([])
    predi_Y = np.array([])
    for k_seed in range(num_validation_replacement):
        kf = KFold(n_splits=num_validation, random_state=k_seed, shuffle=True)
        for train_idx, valid_idx in kf.split(trai_buff_X):
            ini_model = learner(**learner_param)
            ini_model.fit(trai_buff_X[train_idx], trai_buff_Y[train_idx])
            Y_hat = ini_model.predict(trai_buff_X[valid_idx])
            valid_Y = np.append(valid_Y, trai_buff_Y[valid_idx])
            predi_Y = np.append(predi_Y, Y_hat)
    valid_error = 1 - accuracy_score(valid_Y, predi_Y, normalize=True)
    # ======================================#
    # initialize drift detection algorihtm #
    # ======================================#
    dd_method_param = ddi.get_param_from_ddi_t1_table(dd_method.__name__, target_ddi_t1, valid_error, valid_size,
                                                      ddm_param_table)
    detection_alg = dd_method(**dd_method_param)
    # =============================#
    # add validataion error items #
    # =============================#
    error_list = np.abs(valid_Y - predi_Y)
    np.random.shuffle(error_list)
    for i in range(valid_size):
        detection_alg.add_element(error_list[i])
    return detection_alg


def initial_learning_model(trai_buff_X, trai_buff_Y, learner, learner_param, train_size_min):
    # ============================================================#
    # to ensure the learning model in sklearn can be initialized #
    # because for some learning algorihtms,                      #
    # if there is only one class in the training data            #
    # the learning algorihtm will raise exceptions               #
    # ============================================================#
    learning_model = None
    learning_model_temp = None
    # ================================================================================#
    # if the buffer size is large enough and the number of label greater than 1      #
    # then we build a new learner to replace to drifted-learner.                     #
    # number of label greater than 1 is to ensure the learning model can be trained. #
    # ================================================================================#
    # TODO: Multiclass issue may cause errors
    _, label_counts = np.unique(trai_buff_Y, return_counts=True)
    if trai_buff_X.shape[0] >= train_size_min and np.min(label_counts) > 1 and label_counts.shape[0] > 1:
        learning_model = learner(**learner_param)
        learning_model.fit(trai_buff_X, trai_buff_Y)
    else:
        if trai_buff_X.shape[0] >= 3:
            if np.unique(trai_buff_Y).shape[0] > 1:
                learning_model_temp = learner(**learner_param)
                learning_model_temp.fit(trai_buff_X, trai_buff_Y)
            else:
                learning_model_temp = GaussianNB()
                learning_model_temp.fit(trai_buff_X, trai_buff_Y)
    return learning_model, learning_model_temp


def update_train_buffer(trai_buff_X, trai_buff_Y, X_ti, Y_ti):
    if trai_buff_X.shape[0] == 0:
        trai_buff_X = X_ti
        trai_buff_Y = np.array([Y_ti])
    else:
        trai_buff_X = np.vstack([trai_buff_X, X_ti])
        trai_buff_Y = np.append(trai_buff_Y, Y_ti)

    return trai_buff_X, trai_buff_Y


def update_warn_buffer(warn_buff_dict, X_ti, Y_ti):
    # if in warning level, append current observation into warning buffer
    if warn_buff_dict['X'].shape[0] == 0:
        warn_buff_dict['X'] = X_ti
        warn_buff_dict['Y'] = np.array([Y_ti])
    else:
        warn_buff_dict['X'] = np.vstack([warn_buff_dict['X'], X_ti])
        warn_buff_dict['Y'] = np.append(warn_buff_dict['Y'], Y_ti)

    return warn_buff_dict


def stream_learning_prequential(stream_X, stream_Y,
                                dd_method, ddm_param_table,
                                learner, learner_param, is_partial_fit,
                                train_size_min, train_buff_dict,
                                target_ddi_t1=0.9999, valid_size=30, current_t=0):
    # ======================================= #
    # Initialize training buffere and learner #
    # ======================================= #
    trai_buff_X = train_buff_dict['X']
    trai_buff_Y = train_buff_dict['Y']
    learning_model, learning_model_temp = initial_learning_model(trai_buff_X, trai_buff_Y, learner, learner_param,
                                                                 train_size_min)
    # ======================================#
    # initialize drift detection algorihtm #
    # ======================================#
    if learning_model is not None:
        detection_alg = initial_drift_detection_algorihtm(trai_buff_X, trai_buff_Y, learner, learner_param, dd_method,
                                                          ddm_param_table, target_ddi_t1, valid_size)
    # ================================ #
    # Reset warning observation buffer #
    # ================================ #
    warn_buff_dict = {'X': np.array([]), 'Y': np.array([])}
    Y_pred = []
    num_drift = 0
    # ================================ #
    # Stream Learning for loop stream  #
    # ================================ #
    for ti in range(stream_X.shape[0]):
        # get next observation
        X_ti = stream_X[ti: ti + 1]
        Y_ti = stream_Y[ti]
        if learning_model is not None:
            # making prediction
            Y_ti_hat = learning_model.predict(X_ti)
            Y_pred.append(Y_ti_hat[0])
            # detect concept drift (learner performance change), drift level 
            detection_alg.add_element(np.abs(Y_ti - Y_ti_hat[0]))
            if detection_alg.detected_change():
                # if detected a drift, then iterativly solve the substream by
                # stream_learning_prequential_standard
                Y_hat_list, num_drift_update = stream_learning_prequential(stream_X[ti + 1:],
                                                                           stream_Y[ti + 1:],
                                                                           dd_method, ddm_param_table,
                                                                           learner, learner_param,
                                                                           is_partial_fit,
                                                                           train_size_min, warn_buff_dict,
                                                                           target_ddi_t1, valid_size,
                                                                           current_t=ti + current_t + 1)
                num_drift += num_drift_update
                num_drift += 1
                return Y_pred + Y_hat_list, num_drift
            # detect concep drift (learner performance change), warning level
            if detection_alg.detected_warning_zone():
                warn_buff_dict = update_warn_buffer(warn_buff_dict, X_ti, Y_ti)
            else:
                # if not in warning level, or previous warning level change back to normal
                # reset the warning observation buffer
                warn_buff_dict = {'X': np.array([]), 'Y': np.array([])}
            # check if allow incremental fit to new observations    
            if is_partial_fit:
                learning_model.partial_fit(X_ti, np.array([Y_ti]))
        else:
            # check if temporary GaussianNB model is available
            if learning_model_temp is not None:
                # making prediction
                Y_ti_hat = learning_model_temp.predict(X_ti)
                Y_pred.append(Y_ti_hat[0])
                # fit to new observations
                if np.unique(trai_buff_Y).shape[0] > 1:
                    learning_model_temp.fit(trai_buff_X, trai_buff_Y)
                else:
                    learning_model_temp = GaussianNB()
                    learning_model_temp.fit(trai_buff_X, trai_buff_Y)
            else:
                # if no training data is available
                # we randomly guess the observation as label 0
                Y_pred.append(0)
            trai_buff_X, trai_buff_Y = update_train_buffer(trai_buff_X, trai_buff_Y, X_ti, Y_ti)
            learning_model, learning_model_temp = initial_learning_model(trai_buff_X, trai_buff_Y, learner,
                                                                         learner_param, train_size_min)
            if learning_model is not None:
                detection_alg = initial_drift_detection_algorihtm(trai_buff_X, trai_buff_Y, learner, learner_param,
                                                                  dd_method, ddm_param_table, target_ddi_t1, valid_size)

    return Y_pred, num_drift


def stream_learning_prequential_chunk(stream_X, stream_Y, init_train_size, learner, learner_param):
    first_batch = True
    num_batch = int(stream_X.shape[0] / init_train_size)
    stream_spliter = KFold(n_splits=num_batch, random_state=None, shuffle=False)
    Y_pred = [0, 0, 0, 0, 0]

    for _, batch_idx in stream_spliter.split(stream_X):

        batch_X = stream_X[batch_idx]
        batch_Y = stream_Y[batch_idx]

        if first_batch:
            learning_model = GaussianNB()
            learning_model.fit(batch_X[:5], batch_Y[:5])
            for i in range(5, batch_X.shape[0]):
                Y_hat = learning_model.predict(batch_X[i:i + 1])
                Y_pred.append(Y_hat[0])
                learning_model.fit(batch_X[:i + 1], batch_Y[:i + 1])

            learning_model = learner(**learner_param)
            learning_model.fit(batch_X, batch_Y)
            first_batch = False
        else:
            batch_Y_hat = learning_model.predict(batch_X)
            Y_pred = np.append(Y_pred, batch_Y_hat)
            learning_model = learner(**learner_param)
            learning_model.fit(batch_X, batch_Y)
    return Y_pred


def stream_learning_prequential_online(stream_X, stream_Y, init_train_size, learner, learner_param):
    Y_pred = [0, 0, 0, 0, 0]
    learning_model = learner(**learner_param)
    learning_model.fit(stream_X[:5], stream_Y[:5])

    for i in range(5, stream_X.shape[0]):
        Y_hat = learning_model.predict(stream_X[i:i + 1])
        Y_pred.append(Y_hat[0])
        learning_model.fit(stream_X[:i + 1], stream_Y[:i + 1])

    return Y_pred


def stream_learning_prequential_plain(stream_X, stream_Y, init_train_size, learner, learner_param):
    trai_buff_X = stream_X[:init_train_size]
    trai_buff_Y = stream_Y[:init_train_size]
    stream_test_X = stream_X[init_train_size:]

    if np.unique(trai_buff_Y).shape[0] > 1:
        learner = learner(**learner_param)
        learner.fit(trai_buff_X, trai_buff_Y)
    else:
        learner = GaussianNB()
        learner.fit(trai_buff_X, trai_buff_Y)

    pred_Y = []
    for i in range(stream_test_X.shape[0]):
        Y_hat = learner.predict(stream_test_X[i:i + 1])
        pred_Y.append(Y_hat[0])

    return pred_Y
