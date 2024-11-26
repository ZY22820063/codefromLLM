# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 13:19:02 2020

@author: luktian
"""


from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler, Normalizer


def del_na_sd_mask(array, sd_criterion=0.00001):
    """
    array: x_maxtrix, ndarray
    return: mask in T or F
    """
    mask = np.array(range(1, array.shape[1]+1), dtype=bool)
    for col in range(array.shape[1]):
        
        try:
            col_data = array[:, col].astype(float)
            std = np.std(col_data)
            if std < sd_criterion or str(std) == "nan":
                # print(f"{col}: {std}")
                mask[col] = False
        except:
            mask[col] = False
            # print(f"{col}: {array[:,col][0]}")
    
    return mask


def del_corr_mask(array, corr_criterion=0.90):
    """
    """
    mask = np.array(range(1, array.shape[1]+1), dtype=bool)
    print("mask",mask)
    corr_matrix = np.abs(np.tril(np.corrcoef(array.T), k=-1))
    print("corr_matrix",corr_matrix) 
    while corr_matrix.max() >= corr_criterion:
        print("456",corr_matrix.max())
        tuple_location = np.unravel_index(corr_matrix.argmax(), corr_matrix.shape)
        # print(tuple_location)
        corr_matrix[tuple_location] = 0
        if mask[min(tuple_location)]:
            if mask[max(tuple_location)]:
                mask[max(tuple_location)] = False
        # mask[max(tuple_location)] = False
        
        
    return mask




def nasd(data, tar_col=1):
    tar_col = tar_col - 1
    targets = data.pop(data.columns[tar_col])
    descriptors = data
    dp_name_list = list(descriptors.columns)
    dp_name_list_mask = np.ones(len(dp_name_list), dtype=bool)
    dp_value_list = descriptors.T.values.tolist()
    for index, name in enumerate(dp_name_list):
        series = np.array(dp_value_list[index])
        try:
            std = np.std([float(i) for i in series])
            if std < 0.00001:
                dp_name_list_mask[index] = False
                continue
        except:
            dp_name_list_mask[index] = False
            continue
            
        # elif max([float(i) for i in series]) > 1000:
        #     dp_name_list_mask[index] = False
        #     continue
    descriptors = descriptors.loc[:, dp_name_list_mask]
    try:
        targets.apply(float)
        dataset = pd.concat([targets, descriptors], axis=1)
        dataset = dataset.apply(lambda x: x.apply(float))
    except:
        dataset = descriptors.apply(lambda x: x.apply(float))
    return dataset

def delcorr(data, correlation=0.90, tar_col=1):
    tar_col = tar_col - 1
    targets = data.pop(data.columns[tar_col])
    descriptors = data

    descriptors_columns = list(descriptors.columns)  # 变量名
    descriptors = descriptors.applymap(lambda x: float(x))  # 将字符串数值转化为float
    
    # 删除高相关变量 本代码重点
    start_time = datetime.now()  # 开始计时
    #print(f"Start to delete highly correlated varaibles ...\n ")
    corr = descriptors.corr()  # 得到descriptors的相关性矩阵
    corr_dict = {}  # 建立一个空字典用于存放相关性强的变量信息，格式为"检视变量": list[相关强的其他变量]，比如"A": ["B", "C", "D", ...]
    count = 0  # 添加一个计数， 用于提前停止。
    for colname, series in corr.items():  # 在corr这个df中循环，每次循环变量为colname（列名）和series（一列数据,也就是colname变量与其他变量的相关性系数这一列数据）
        tmp = []  # 临时列表用于存放与colname高相关性的变量名，比如"B", "C", "D"
        count += 1  # 添加的计数+1， 用于提前停止
        for i in range(len(series)):  # series中做循环，每次循环为series中具体的每个相关性系数
            if abs(series[i]) > correlation and abs(series[i]) < 1:  # 判断series中每个系数是否在0.95~1之间
                tmp.append(series.index[i])  # 是的话把这个变量名存放到tmp中
        if len(tmp) > 0:  # 如果tmp中有变量名，也就是如果与colname有系数为0.95~1之间的其他变量
            corr_dict[colname] = tmp  # 放到corr_dict里去
        # if count == 10:  # 用于提前停止，来进行test
        #     break  # 达到上限就跳出
    # 以上生成了一个corr_dict，存放了相关性信息
    
    # 以下开始进行高相关性变量删除
    corr_mask = np.ones([descriptors.shape[1]])  # 生成一个与descriptors等行长的矩阵，默认为1，需要删除的变量则改成0
    while len(corr_dict):  # while循环，因为是不等长循环，当corr_dict为空时跳出
        tmp = sorted(corr_dict.items(), key=lambda item: len(item[1]), reverse=True)[0]  # 先对corr_dict进行排序，排序依据为根据corr_dict的values长度从大到小排序，返回的tmp为一个tuple，格式为("A", ("B", "C", "D", ...))，A是corr_dict的keys，也就是考察的变量，BCD等为corr_dict的values，也就是与A强相关的变量
        corr_dict.pop(tmp[0])  # corr_dict中删除"A"
        if len(tmp[1]) > 1:
            corr_mask[descriptors_columns.index(tmp[0])] = 0  # corr_mask中"A"对应的1改为0
        for i, j in corr_dict.items():  # corr_dict剩余values的list，将每个list的"A"删除
            try:
                j.pop(j.index(tmp[0]))  # 若A不在list中，会报出ValueError
            except ValueError:
                pass
    corr_mask = pd.Series(corr_mask).apply(lambda x: bool(x))  # 将corr_mask布尔化
    descriptors = descriptors[descriptors.columns[corr_mask]]  # 选取True的变量
    
    end_time = datetime.now()  # 结束计时
    #print(f"End to delete highly correlated varaibles ... \n ")
    #print(f"Deleting high correlations: totally spend time of {(end_time-start_time).seconds/60} minutes \n ")
    
    return pd.concat([targets, descriptors], axis=1)


# del mutual info = 0
def mirf(data, tar_col=1):
    tar_col = tar_col - 1
    targets = data.pop(data.columns[tar_col])
    descriptors = data
    mir = list(mutual_info_regression(descriptors, targets, random_state=0))
    dp_name_list_mask = np.ones(len(mir), dtype=bool)
    for index, mir_value in enumerate(mir):
        if mir_value == 0:
            dp_name_list_mask[index] = False
    descriptors = descriptors.loc[:, dp_name_list_mask]
    return pd.concat([targets, descriptors], axis=1)

import os
def check_dir(dirpath):
    if dirpath in os.listdir("output"):
        pass
    else:
        os.mkdir("output/" + dirpath)
        
def datascale(dataset, scaler):
    if scaler == 'minmax':
        scaler = MinMaxScaler()
    elif scaler == 'normalizer':
        scaler = Normalizer()
    
    pass

def reg2cls(data, tar_col=1):
    tar_col = tar_col - 1
    targets = data.pop(data.columns[tar_col])
    descriptors = data
    try:
        targets.apply(float)
        if len(set(targets.tolist())) > 8:
            targets_mask = pd.Series([ 1 if i > targets.median() else 0 for i in targets])
            targets_mask.index = targets.index
            targets = targets_mask
        return pd.concat([targets, descriptors], axis=1)
    except:
        return 0
        
    






















