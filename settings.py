#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 16:07:43 2022

@author: system01
"""

lgb_default_train_params = {
    "learning_rate": 5e-2,
    "n_iter": 500,
    "force_col_wise": True,
    "extra_trees": False,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "num_leaves": 31,
    "max_depth": -1,
    "colsample_bytree": 1.0,
    "subsample": 1.0,
    "subsample_freq": 1,
    "min_data_in_leaf": 20,
}

cb_default_train_params = {
    "iterations": 500,
    "learning_rate": 5e-2,
    "verbose": True,
}
