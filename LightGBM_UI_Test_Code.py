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


st.header('Light GBM Regression UI')


# Input LGBM Parameters
with st.expander('Input LGBM Parameters'):
    n_trees = st.slider('Number of Trees', min_value=1, max_value=1000, value= 200)
    tree_depth = st.slider('Depth of Trees', min_value=1, max_value=10, value= 5)
    min_leaf_size = st.slider('Number of leaves in Trees', min_value=4, max_value=100, value= 15)
    learning_rate = st.slider('Learning Rate ', min_value=1, max_value=1000, value= 10)
    subsample = st.slider('Row Sample ', min_value=0.5, max_value=1.0, value= 0.8)
    colsample_bytree = st.slider('Column Sample ', min_value=0.5, max_value=1.0, value= 0.8)
    bagging_freq = st.slider('Bagging Freq ', min_value=1, max_value=10, value= 2)
    
    
    tree_params = {'n_trees':n_trees, 'tree_depth':tree_depth, 'min_leaf_size': min_leaf_size, \
                   'learning_rate': learning_rate/1000.0, 'subsample': subsample, 'colsample_bytree': colsample_bytree,\
                   'bagging_freq': bagging_freq}
# Input LGBM Parameters Ends


# Widget for uploading file
uploaded_file = st.file_uploader(label = 'Upload file for regression', type=("csv"))

# Columns/Features have to come from CSV
# Currently based on static list

feat_contri_dict = {}
feat_mono_dict = {}
exclude_from_model = {}
categorical_cols = {}
    

if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    dataframe['DATE'] = pd.to_datetime(dataframe['DATE'])
    # st.write(dataframe)
    
    # break
    
    column_list = dataframe.columns
    
    target_var = st.selectbox("Select Name of Target/Dependent Variable", column_list)
    feature_list = [x for x in column_list if x != target_var]
    
    with st.expander("Select Categorical Columns: "):
        
        for col in feature_list:
            categorical_cols[col] = st.checkbox('Is '+col+' categorical', value=False)
    
    
    with st.sidebar:
        
        for col in feature_list:
            with st.expander(col):
                feat_contri_dict[col] = st.slider("Contribution "+ col, min_value=0, max_value=100, value=100)
                        
                feat_mono_dict[col] = st.selectbox(
                    'How '+col+' and Target variable are correlated?',
                    ('Neutral', 'Positive', 'Negative'))
                
                exclude_from_model[col] = st.checkbox(
                    'Exclude '+col+' from Model', value=False)
            
    
    for col in exclude_from_model:
        if exclude_from_model[col]:
            feature_list = [x for x in feature_list if x != col]
            feat_contri_dict.pop(col)
            feat_mono_dict.pop(col)
            
    granularity_list = ['BRAND', 'CENSUS_TRADING_AREAS', 'PACK SIZE']
    
    
    # st.write(factor_df)
    
    # Scaling DF, for better results
    factor_df = scale_data(dataframe, granularity_list, target_var)
    data_rows = dataframe.shape[0]
    dataframe = dataframe.merge(factor_df, on= granularity_list, how = 'left')
    assert dataframe.shape[0] == data_rows
    dataframe[target_var + '_new'] = dataframe[target_var]/dataframe['factor']

    target_var_old = target_var
    target_var = target_var + '_new'
    
    #Running Model
    data, model = model_func(dataframe, target_var, target_var_old, feature_list, list(feat_contri_dict.values()), list(feat_mono_dict.values()), \
               categorical_cols, exclude_from_model, tree_params)
        
    with st.expander('Feature Importance: '):
        # Plotting Feature Importance
        # Plot is inside function, returns None
        get_feature_importance_chart(model)
        
    with st.expander('Actual Vs Predicted: '):
        st.write('')
        st.write('Filter for AVP Plot')
        
        brand_selected = st.selectbox(
            'Select Brand: ',
            list(data['BRAND'].unique()) + ['None'], key = 1)
        
        pack_size_selected = st.selectbox(
            'Select PACK SIZE: ',
            list(data['PACK SIZE'].unique())+ ['None'], key = 2)
        
        CTA_selected = st.selectbox(
            'Select CTA: ',
            list(data['CENSUS_TRADING_AREAS'].unique()) + ['None'], key = 3)
        
        temp = data.copy()
        if brand_selected != 'None':
            temp = temp[temp['BRAND'] == brand_selected]
            
        if pack_size_selected != 'None':
            temp = temp[temp['PACK SIZE'] == pack_size_selected]
            
        if CTA_selected != 'None':
            temp = temp[temp['CENSUS_TRADING_AREAS'] == CTA_selected]
            
        
        chart_data = temp.groupby('DATE')[[target_var_old, 'predicted_old']].sum().reset_index()
        chart_data = chart_data.sort_values(by = 'DATE', ascending = False)
        st.line_chart(chart_data, x ='DATE', y = [target_var_old, 'predicted_old'])
    
    
    # Base Vs Incremental (Total)
    with st.expander('Aggregated Base Vs Incremental: '):
        agg_base_inc_df = get_base_contribution(data, model, target_var_old, feature_list, feat_mono_dict, granularity_list)
        
        brand_selected = st.selectbox(
            'Select Brand: ',
            list(data['BRAND'].unique()) + ['None'], key = 4)
        
        pack_size_selected = st.selectbox(
            'Select PACK SIZE: ',
            list(data['PACK SIZE'].unique()) + ['None'], key = 5)
        
        CTA_selected = st.selectbox(
            'Select CTA: ',
            list(data['CENSUS_TRADING_AREAS'].unique())+ ['None'], key = 6)
        
        temp = agg_base_inc_df.copy()
        if brand_selected != 'None':
            temp = temp[temp['BRAND'] == brand_selected]
            
        if pack_size_selected != 'None':
            temp = temp[temp['PACK SIZE'] == pack_size_selected]
            
        if CTA_selected != 'None':
            temp = temp[temp['CENSUS_TRADING_AREAS'] == CTA_selected]
        
        chart_data = temp.groupby('DATE')[[target_var_old, 'base_predict', 'predicted_old', 'incremental']].sum().reset_index()
        chart_data = chart_data.sort_values(by = 'DATE', ascending = False)
        st.line_chart(chart_data, x = 'DATE', y = [target_var_old, 'predicted_old', 'base_predict', 'incremental'])
        
    # Base Vs Incremental Ends
    
    
    # Feature wise Incremental/Simulation
    with st.expander('Feature wise Incremental/Simulation '):
        st.write('')
        st.write('Select Variable to see relationship with Target:')
        
        simulation_variables = []
        for var in feature_list:
            if var not in granularity_list:
                for excluded in exclude_from_model:
                    if exclude_from_model[excluded]:
                        if var != excluded:
                            simulation_variables.append(var)
                            
        if target_var in simulation_variables:
            simulation_variables.remove(target_var)
        
        variable_selected = st.selectbox(
            'Select Variable: ', simulation_variables)
        
        simulation_data = get_contribution(data, model, feature_list, variable_selected, granularity_list)
        
        sim_chart_data = simulation_data.sum()
        st.line_chart(sim_chart_data)