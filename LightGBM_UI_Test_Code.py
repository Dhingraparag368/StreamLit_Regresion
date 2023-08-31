# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:58:44 2023

@author: Parag.Dhingra
"""


import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error

import streamlit as st
import lightgbm as lgb

# Local library
from Support_file import *



feat_contri_dict = {}
feat_mono_dict = {}
exclude_from_model = {}
categorical_cols = {}
key_counter = 1
granularity_list = []
granularity_filter_selected_dict = {}
# To be provided as Input by User
use_date_for_group_by = True
groupby_var = 'DATE'
potential_target_var_names = ['TARGET', 'SALES', 'UNITS']
target_var = ''



st.header('Light GBM Regression UI')


# Input LGBM Parameters
tree_params = lightgbm_hyperparam_block()


# Widget for uploading file
uploaded_file = st.file_uploader(label = 'Upload file for regression', type=("csv"))


if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    dataframe.columns = [x.upper() for x in dataframe.columns]
    column_list = dataframe.columns
    
    if 'DATE' in dataframe.columns:
        dataframe['DATE'] = pd.to_datetime(dataframe['DATE'])

    
    # Setting Target Variable
    target_var = set_target_var_block(potential_target_var_names, column_list)
    
    # Setting feature list
    feature_list = [x for x in column_list if x != target_var]
    
    # Category col block
    categorical_cols = set_categorical_cols_block(feature_list, categorical_cols)
    
    
    # Code for Side Bar
    feature_list, feat_contri_dict, feat_mono_dict, exclude_from_model = side_bar_for_features_contri_mono(feature_list, feat_contri_dict, feat_mono_dict, exclude_from_model)
    

    # Granularity, May not be used in later code, more like a replacement for categorical dict
    granularity_list = [x for x in categorical_cols if categorical_cols[x]]
    dataframe['granularity_col'] = ''
    for x in granularity_list:
        dataframe['granularity_col'] = dataframe['granularity_col'] + ' ' + dataframe[x].astype('str')
        
    # Scale Data for better Accuracy
    dataframe, target_var_old, target_var = scale_data_block(dataframe, granularity_list, target_var)
    
    #Running Model
    data, model = model_func(dataframe, target_var, target_var_old, feature_list, \
                             feat_contri_dict, feat_mono_dict, \
                             categorical_cols, exclude_from_model, tree_params)
    
    
    with st.expander('Feature Importance: '):
        get_feature_importance_chart(model)
    
    # Actual Vs Predicted Widget
    key_counter = actual_vs_predicted_block(data, granularity_list, key_counter, column_list, target_var_old)
    

    # Base Vs Incremental (Total)
    data, key_counter = base_vs_incremental_block(data, model, target_var_old, feature_list, feat_mono_dict, column_list, key_counter)
        
    # Feature Wise Contribution
    data, key_counter = variable_contribution_block(data, model, feature_list, exclude_from_model, feat_mono_dict, 
                                    granularity_list, target_var, target_var_old, key_counter)

    download_model_data_block(data)