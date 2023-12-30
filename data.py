#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 17:06:03 2022

@author: system01
"""

import pandas as pd
import numpy as np
from collections import namedtuple
from typing import Optional, List
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split as sk_train_test_split, KFold
from sklearn.datasets import fetch_california_housing as sk_fetch_california_housing


def train_test_split(features: pd.DataFrame, 
                     test_size: float, 
                     include_valid=True,
                     shuffle=False) -> namedtuple:
    '''
    Parameters
    ----------
    features : pd.DataFrame
    test_size : float
        The value range is (0, 1).
    include_valid : float
        If True, include validation data.

    Returns
    -------
    namedtuple('Data', ['train', 'test'])
    '''
    
    train, test = sk_train_test_split(features,
                                      test_size=test_size, 
                                      shuffle=shuffle)
    if include_valid:
        valid, test = sk_train_test_split(test,
                                          test_size=0.5, 
                                          shuffle=shuffle)
        Data = namedtuple('Data', ['train', 'valid', 'test'])
        return Data(train, valid, test)    
    else:
        Data = namedtuple('Data', ['train', 'test'])
        return Data(train, test)        
    

def k_fold(features: pd.DataFrame, 
           n_splits: int, 
           purge_window: Optional[int]=None) -> namedtuple:
    '''
    Parameters
    ----------
    features : pd.DataFrame
        deature DataFrame for cross validation.
    n_splits : int
        K parameter of K-Fold split.
    purge_window : Optional[int], optional
        Remove latest data window to avoid leakage.
    
    Yields
    ------
    namedtuple('Data', ['train', 'valid', 'test'])
    '''
    
    Data = namedtuple('Data', ['train', 'valid', 'test'])
    for train_index, test_index in KFold(n_splits, shuffle=False).split(features):
        center_index = int(len(test_index) / 2)
        valid_index = test_index[:center_index + 1]
        test_index = test_index[center_index + 1:]
        
        if purge_window is not None:
            if purge_window < 1:
                raise ValueError('purge_window should be greater than 0.')
            
            drop_index = []
            for i in range(len(train_index)):
                if i == len(train_index) - purge_window:
                    if train_index[i] < len(features) - purge_window:
                        while i < len(train_index):
                            drop_index.append(i)
                            i += 1
                    break
                else:
                    if train_index[i + 1] - train_index[i] > 1:
                        count = 0
                        while count < purge_window:
                            drop_index.append(i - count)
                            count += 1
            
            train_index = np.delete(train_index, drop_index)
            
            valid_index = valid_index[:-purge_window]
            
            if test_index[-1] != len(features) - 1:
                test_index = test_index[:-purge_window]
        
        # TODO: refactor
        # train, validのみを返すようにし、またidによるsplitを実装
        # yield Data(train=features.iloc[train_index],
        #            valid=features.iloc[valid_index],
        #            test=features.iloc[test_index])
        yield features.iloc[train_index], features.iloc[valid_index]


def load_table(path: str,
               object_columns: List[str]=[],
               int_columns: List[str]=[],
               bool_columns: List[str]=[],
               parse_dates: List[str]=[],
               use_columns: Optional[List[str]]=None):
    '''
    Parameters
    ----------
    path : str
        The path of csv file.
    object_columns : List[str], optional
        List of column names whose type is object.
    int_columns : List[str], optional
        List of column names whose type is int.
    bool_columns : List[str], optional
        List of column names whose type is bool.
    parse_dates : List[str], optional
        List of column names whose type is datetime.
    use_columns : Optional[List[str]], optional
        List of column names to use.

    Returns
    -------
    pd.DataFrame
    '''
    
    columns = pd.read_csv(path, nrows=0, usecols=use_columns).columns
    dtype = get_dtype_dict(columns=columns,
                           object_columns=object_columns,
                           int_columns=int_columns,
                           bool_columns=bool_columns,
                           parse_dates=parse_dates)
    
    return pd.read_csv(path, dtype=dtype, parse_dates=parse_dates, usecols=use_columns)


def get_dtype_dict(columns: List[str],
                   object_columns: List[str]=[],
                   int_columns: List[str]=[],
                   bool_columns: List[str]=[],
                   parse_dates: List[str]=[]) -> dict:
    '''
    Parameters
    ----------
    columns : List[str]
        List of DataFrame's columns.
    object_columns : List[str], optional
        List of column names whose type is object.
    int_columns : List[str], optional
        List of column names whose type is int.
    bool_columns : List[str], optional
        List of column names whose type is bool.
    parse_dates : List[str], optional
        List of column names whose type is datetime.
    
    Returns
    -------
    pd.DataFrame
    '''
    
    dtype = {}
    for i in object_columns:
        dtype[i] = 'object'
    for i in int_columns:
        dtype[i] = 'int32'
    for i in bool_columns:
        dtype[i] = 'bool'
    
    specified_columns = object_columns + int_columns + bool_columns + parse_dates
    for i in columns:
        if i in specified_columns:
            continue
        else:
            dtype[i] = 'float32'
    
    return dtype
    
    


# 一般的な箱髭図の外れ値に従う。
# https://bellcurve.jp/statistics/course/5222.html
def detect_outliers(df: pd.DataFrame,
                    target_column: str,
                    groupby_column: Optional[str]=None,
                    iqr_coef: int=1.5):
    '''
    Parameters
    ----------
    df : pd.DataFrame
        Target DataFrame.
    target_column : str
        Column name of objective variable.
    groupby_column : Optional[str], optional
        Columns name used for groupby.
    iqr_coef : int, optional
        The coefficient of interquartile range. The default is 1.5.

    Returns
    -------
    pd.DataFrame
    '''
    
    def _detect_outliers(series):
        flag = series.sort_values()
        first_quarter = flag.iloc[int(len(flag) / 4)]
        third_quarter = flag.iloc[int(len(flag) * 3 / 4)]
        iqr = third_quarter - first_quarter
        higher_threshold = third_quarter + iqr * iqr_coef
        lower_threshold = first_quarter - iqr * iqr_coef
        flag = (lower_threshold <= flag) & (flag <= higher_threshold)
        # inplaceになるのを防ぐ
        _series = series.copy()
        _series[flag.index] = flag.values
        
        return _series
    
    if groupby_column is None:
        return df[_detect_outliers(df[target_column])]
    else:
        return df[df.groupby(groupby_column)[target_column].transform(_detect_outliers)]
    

def binning_features(df: pd.DataFrame, 
                     train: bool=True, 
                     n_bins: int=10,
                     encode: str='ordinal',
                     strategy: str='quantile',
                     model: Optional[KBinsDiscretizer]=None):
    '''
    Parameters
    ----------
    df : pd.DataFrame
        Target DataFrame.
    train : bool, optional
        Specify true for training data. The default is True.
    n_bins : int, optional
        The number of bins to produce. The default is 10.
    encode : str, optional
        Method used to encode the transformed result. The default is 'ordinal'.
            ・‘onehot’: Encode the transformed result with one-hot encoding and return a sparse matrix. Ignored features are always stacked to the right.
            ・‘onehot-dense’: Encode the transformed result with one-hot encoding and return a dense array. Ignored features are always stacked to the right.
            ・‘ordinal’: Return the bin identifier encoded as an integer value. 
    strategy : str, optional
        Strategy used to define the widths of the bins. The default is 'quantile'.
            ・‘uniform’: All bins in each feature have identical widths.
            ・‘quantile’: All bins in each feature have the same number of points.
            ・‘kmeans’: Values in each bin have the same nearest center of a 1D k-means cluster.
    model : Optional(KBinsDiscretizer)
        A trained KBinsDiscretizer object. Specify if train == False. The default is None.
    
    Returns
    -------
    namedtuple('Result', ['model', 'df'])
        namedtuple object composed of a trained KBinsDiscretizer and transformed data.

    '''
    
    use_columns = [i for i in df.columns if len(df[i].unique()) > 3]
    
    # naは受け付けてくれない
    processed = df[use_columns].copy()
    processed = processed.fillna(processed.median())
    
    if train:
        model = KBinsDiscretizer(n_bins=n_bins,
                                 encode=encode,
                                 strategy=strategy)
        model.fit(processed)
    else:
        if not isinstance(model, KBinsDiscretizer):
            raise TypeError(f'invalid model type ({type(model)}).')
    
    processed = model.transform(processed.values)
    df[use_columns] = processed
    Result = namedtuple('Result', ['model', 'df'])
    
    return Result(model, df)


def ranking_features(df: pd.DataFrame,
                     groupby_column: Optional[str]=None,
                     ascending: bool=False):
    '''
    Parameters
    ----------
    df : pd.DataFrame
        Target DataFrame.
    groupby_column : Optional[str], optional
        Columns name used for groupby.
    ascending : bool, optional
        if False, the higher the value, the higher the rank.
        A lower value for the rank indicates a higher rank.

    Returns
    -------
    pd.DataFrame
    '''
    
    def to_rank(series):
        if len(series.unique()) <= 3:
            return series
        
        rank = series.sort_values(ascending=ascending)
        unique_rank = rank.drop_duplicates(keep='first')
        rank[:] = np.nan
        rank[unique_rank.index] = unique_rank.reset_index().index
        rank = rank.fillna(method='ffill')
        series[rank.index] = rank.values
        
        return series.astype('int32')
    
    if groupby_column == None:
        return df.apply(to_rank)
    else:
        return df.groupby(groupby_column).transform(to_rank)


# def load_boston() -> pd.DataFrame:
#     '''
#     load scikit-learn boston data as pd.DataFrame.

#     Returns
#     -------
#     ps.DataFrame
#     '''
    
#     boston = sk_load_boston()
#     df = pd.DataFrame(boston.data, columns=boston.feature_names)
#     df['y'] = boston.target
    
#     return df


def load_california_housing() -> pd.DataFrame:
    '''
    load scikit-learn california_housing data as pd.DataFrame.

    Returns
    -------
    ps.DataFrame
    '''
    
    data = sk_fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['y'] = data.target
    
    return df




    