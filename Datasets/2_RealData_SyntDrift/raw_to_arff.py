# -*- coding: utf-8 -*-
"""
Created on Wed Aug 07 10:43:37 2020

@author: DeSI-Anjin Liu, AAII, FEIT, UTS
@email: anjin.liu@uts.edu.au
"""
import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

_DataFolder = './RealData_SyntDrift_raw/'


def simulate_drift(df, drift_param):
    
    print(df[df.columns[-1]].value_counts())

    drift_description = ''
    r_seed = drift_param['r_seed']
    drift_fea = drift_param['drift_fea']
    drift_num = drift_param['drift_num']
    drift_del = drift_param['drift_del']
    
    if drift_num > 0:
        np.random.seed(r_seed)
        kf = KFold(n_splits=drift_num+1, shuffle=True, random_state=r_seed)
        
        stream_batch_list = []
        
        del_idx = 0
        for _, stream_batch_idx in kf.split(df):
            stream_batch_df = df.iloc[stream_batch_idx]
            stream_batch_df.iloc[:, drift_fea] = stream_batch_df.iloc[:, drift_fea] + drift_del[del_idx]
            stream_batch_list.append(stream_batch_df)
            del_idx += 1
        
        df = pd.concat(stream_batch_list)
    
    return df, drift_description


def dump_to_arff(df, relation_name, description, output_file):
    
    attributes = []
    for col_n in df.columns:
        
        if df[col_n].dtypes == 'object':
            # sort nominal attributes
            nominal_att_list = df[col_n].unique().tolist()
            list.sort(nominal_att_list)
            # remove missing value mark from the attribute list
            if '?' in nominal_att_list:
                nominal_att_list.remove('?')
            attributes.append((col_n, nominal_att_list))
        else:
            attributes.append((col_n, 'NUMERIC'))

    arff_dic = {
        'attributes': attributes,
        'data': df.values,
        'relation': relation_name,
        'description': description
    }

    with open(output_file, 'w', encoding="utf8") as f:
        arff.dump(arff_dic, f)


def wdbc_preprocess(drift_param):
    """
    wdbc.data
    """
    file_name = 'wbdc/wdbc.data'
    file_path = _DataFolder + file_name
    data = pd.read_csv(file_path, header=None)

    col_names = ['ID', 'Diagnosis', 'radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity',
                 'concave_points', 'symmetry', 'fractal_dimension', 'radius_mean', 'texture_mean', 'perimeter_mean',
                 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
                 'symmetry_mean', 'fractal_dimension_mean', 'radius_worst', 'texture_worst', 'perimeter_worst',
                 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst',
                 'symmetry_worst', 'fractal_dimension_worst']

    data.columns = col_names

    col_usefu = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points',
                 'symmetry', 'fractal_dimension', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean',
                 'fractal_dimension_mean', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst',
                 'fractal_dimension_worst', 'Diagnosis']

    data = data[col_usefu]

    data, description = simulate_drift(data, drift_param)
    dump_to_arff(data, 'Wisconsin Diagnostic Breast Cancer', description, 'wdbc.arff')


def cred_preprocess(drift_param):
    """
    Credit-g.data
    """
    file_name = 'Credit-g.data'
    file_path = _DataFolder + file_name

    data = simulate_drift(data, drift_param)
    dump_to_arff(data, 'Wisconsin Diagnostic Breast Cancer', 'some description\nlalala', 'wdbc.arff')
    pass


def glas_preprocess(drift_param):
    """
    glass.data
    """
    file_name = 'glass.data'
    file_path = _DataFolder + file_name
    data = pd.read_csv(file_path, header=None)

    col_names = ['ID', 'refractive_index', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'class']

    data.columns = col_names

    col_usefu = ['refractive_index', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'class']

    data = data[col_usefu]
    data['class'] = data['class'].astype(str)

    # filtering minor classes
    # data = data.loc[(data['class']=='1') | (data['class']=='2')]
    class_merge_dict = {'7': '1', '3': '1', '5': '1', '6': '1'}
    data.replace(class_merge_dict, inplace=True)
    data, drift_description = simulate_drift(data, drift_param)
    dump_to_arff(data, 'Glass Identification Data Set', drift_description, 'glas.arff')


def iono_preprocess(drift_param):
    """
    ionosphere.data
    """
    file_name = 'iono/ionosphere.data'
    file_path = _DataFolder + file_name
    data = pd.read_csv(file_path, header=None)

    col_names = []
    for i in range(34):
        col_names.append('f' + str(i))
    col_names.append('class')
    data.columns = col_names

    data, drift_description = simulate_drift(data, drift_param)
    dump_to_arff(data, 'ionosphere Data Set', drift_description, 'iono.arff')


def iris_preprocess(drift_param):
    """
    iris.data
    """
    file_name = 'iris/iris.data'
    file_path = _DataFolder + file_name
    data = pd.read_csv(file_path, header=None)

    col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    data.columns = col_names

    data, drift_description = simulate_drift(data, drift_param)
    dump_to_arff(data, 'iris', drift_description, 'iris.arff')


def soyb_preprocess(drift_param):
    """
    soybean-large.data
    """
    file_name = 'soyb/soybean-large.data'
    file_path = _DataFolder + file_name
    data1 = pd.read_csv(file_path, header=None)

    file_name = 'soyb/soybean-large.test'
    file_path = _DataFolder + file_name
    data2 = pd.read_csv(file_path, header=None)

    data = pd.concat([data1, data2])

    col_names = ['class', 'date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged', 'severity',
                 'seed-tmt', 'germination', 'plant-growth', 'leaves', 'leafspots-halo', 'leafspots-marg',
                 'leafspot-size', 'leaf-shread', 'leaf-malf', 'leaf-mild', 'stem', 'lodging', 'stem-cankers',
                 'canker-lesion', 'fruiting-bodies', 'external_decay', 'mycelium', 'int-discolor', 'sclerotia',
                 'fruit-pods', 'fruit_spots', 'seed', 'mold-growth', 'seed-discolor', 'seed-size', 'shriveling',
                 'roots']

    col_usefu = ['date', 'plant-stand', 'precip', 'temp', 'hail', 'crop-hist', 'area-damaged', 'severity', 'seed-tmt',
                 'germination', 'plant-growth', 'leaves', 'leafspots-halo', 'leafspots-marg', 'leafspot-size',
                 'leaf-shread', 'leaf-malf', 'leaf-mild', 'stem', 'lodging', 'stem-cankers', 'canker-lesion',
                 'fruiting-bodies', 'external_decay', 'mycelium', 'int-discolor', 'sclerotia', 'fruit-pods',
                 'fruit_spots', 'seed', 'mold-growth', 'seed-discolor', 'seed-size', 'shriveling', 'roots', 'class']

    data.columns = col_names
    data = data[col_usefu]
    for col_n in data.columns:
        data[col_n] = data[col_n].astype(str)
    data, drift_description = simulate_drift(data, drift_param)
    dump_to_arff(data, 'soybean-large', drift_description, 'soyb.arff')


if __name__ == "__main__":
    
    wdbc_drift_param = {'r_seed': 1,
                        'drift_num': 4,
                        'drift_fea': [0, 1, 2, 3],
                        'drift_del': [0, 0.1, 0.2, 0.3]}
    data = wdbc_preprocess(wdbc_drift_param)

#    glas_drift_param = {'r_seed': 1,
#                        'drift_num': 4,
#                        'drift_fea': [0, 1],
#                        'drift_del': [0, 0.1, 0.2, 0.3]}
#    
#    glas_preprocess(glas_drift_param)
#    
#    iono_drift_param = {'r_seed': 1,
#                        'drift_num': 4,
#                        'drift_fea': [0, 1],
#                        'drift_del': [0, 0.1, 0.2, 0.3]}
#    iono_preprocess(iono_drift_param)
#
#    iris_drift_param = {'r_seed': 1,
#                        'drift_num': 4,
#                        'drift_fea': [0, 1],
#                        'drift_del': [0, 0.1, 0.2, 0.3]}
#    iris_preprocess(iris_drift_param)
#
#    soyb_drift_param = {'r_seed': 1,
#                        'drift_num': 4,
#                        'drift_fea': [0, 1],
#                        'drift_del': [0, 0.1, 0.2, 0.3]}
#    soyb_preprocess(soyb_drift_param)
