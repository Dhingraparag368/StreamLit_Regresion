# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 14:58:44 2023

@author: Parag.Dhingra
"""

import streamlit as st
import lightgbm as lgb
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np




def model_func(data, target_var, target_var_old, feature_list, feat_contri_list, \
               monotone_list, categorical_cols, exclude_from_model, tree_params):
    
    #Data Prep
    for col in categorical_cols:
        if categorical_cols[col]:
            data[col] = data[col].astype('category')
    
    
    # st.write(feature_list)
    monotone_list_int = []
    for i in monotone_list:
        if i == 'Positive':
            monotone_list_int.append(1)
        elif i == 'Negative':
            monotone_list_int.append(-1)
        else:
            monotone_list_int.append(0)
            
    
    
    tree = tree_params['n_trees']
    depth = tree_params['tree_depth']
    leaf_size = tree_params['min_leaf_size']
    
    learning_rate = tree_params['learning_rate']
    subsample = tree_params['subsample']
    colsample_bytree = tree_params['colsample_bytree']
    bagging_freq = tree_params['bagging_freq']
    
    # st.write(feat_contri_list)
    # st.write(monotone_list)
    
    with st.expander("Data snapshot: "):
        st.write("Features in Model: "+ str(len(feature_list)))
        st.write("Rows in Model: "+ str(data.shape[0]))
        st.write(data)
    
    model = lgb.LGBMRegressor(num_tree = tree, max_depth = depth, min_data_in_leaf = leaf_size, \
                              feature_contri = feat_contri_list, monotone_constraints = monotone_list_int,\
                              colsample_bytree = colsample_bytree, learning_rate = learning_rate, \
                              subsample = subsample, bagging_freq = bagging_freq)

    # st.write('')
    # st.write(str(model))
    # st.write(target_var)
    
    X_train = data[feature_list]
    Y_train = data[target_var]
    
    # st.write(X_train.dtypes)
    # st.write(Y_train.dtypes)
    
    # st.write(str(type(X_train)))
    # st.write(str(type(Y_train)))
    
    # st.write(X_train)
    
    model.fit(X_train, Y_train)
    
    # st.write('After Model Fit')
    data['predicted'] = model.predict(X_train)
    data['predicted'] = np.where(data['predicted']< 0, 0, data['predicted'])
    data['predicted_old'] = data['predicted']*data['factor']
    
    with st.expander("Model Accuracy Metrics: "):
        st.write('')
        st.write('Model Ran Successfully !!!')
        st.write('Model Accuracy Metrics are: ')
        st.write('')
        
        r_sq  = np.round(r2_score(data[target_var_old], data['predicted_old']), 2)
        mae   = np.round(mean_absolute_error(data[target_var_old], data['predicted_old']), 2)
        wmape = np.round((mae/np.mean(data[target_var_old]))*100, 2)
        
        st.write("R Square for Model: " + str(r_sq))
        st.write("Mean Absolute Error for Model: " + str(mae))
        st.write("% WMAPE for Model: " + str(wmape))
    
    return data, model



def get_feature_importance_chart(model):
        
    imp_df = pd.DataFrame({'Feature': model.feature_name_, 'Importance': model.feature_importances_})
    st.write(imp_df)
    
    imp_df.set_index(['Feature'], inplace = True)
    imp_df.sort_values(by = 'Importance', ascending=False, inplace = True)
    
    st.bar_chart(imp_df)
    return None



def scale_data(data, scale_groupby, scale_col):
    
    factor_df = data.groupby(scale_groupby)[scale_col].max().reset_index()
    factor_df['factor'] = factor_df[scale_col]/10000
    factor_df.drop([scale_col], axis = 1, inplace = True)
    
    return factor_df



def get_contribution(data, model, feature_list, variable, granularity_list):
    
    output_cols = []
    data['base_predict'] = model.predict(data[feature_list])
    output_cols.append('base_predict')
    
    st.write('Starting simulation process...')
    
    for i in range(0, 100, 10):
        
        temp = data.copy()
        temp[variable] = temp[variable]*i/100
        
        data['predict_'+str(i)] = model.predict(temp[feature_list])
        output_cols.append('predict_'+str(i))
        
    
    # st.write(output_cols)
    # st.write(data.columns)
    st.write('Starting simulation completed...')
    return data.groupby(granularity_list)[output_cols].sum().reset_index()



def get_base_contribution(data, model, target_var_old, feature_list, feat_mono_dict, granularity_list):
    
    temp = data.copy()
    for feat in feat_mono_dict:
        if feat_mono_dict[feat] == 'Positive':
            temp[feat] = temp[feat].min()
        
        if feat_mono_dict[feat] == 'Negitive':
            temp[feat] = temp[feat].max()
            
    data['base_predict'] = model.predict(temp[feature_list])
    data['base_predict'] = data['base_predict']*data['factor']
    data['incremental'] = data['predicted_old'] - data['base_predict']


    return data.groupby(granularity_list + ['DATE'])[[target_var_old, 'predicted_old', 'base_predict', 'incremental']]\
        .sum().reset_index()