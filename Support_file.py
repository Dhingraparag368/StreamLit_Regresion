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



def model_func(data, target_var, target_var_old, feature_list, feat_contri_dict, \
               feat_monot_dict, categorical_cols, exclude_from_model, tree_params):
    
    #Data Prep
    for col in categorical_cols:
        if categorical_cols[col]:
            data[col] = data[col].astype('category')
    
    
    # st.write(feature_list)
    monotone_list_int = []
    for var in feature_list:
        if feat_monot_dict[var] == 'Positive':
            monotone_list_int.append(1)
        elif feat_monot_dict[var] == 'Negative':
            monotone_list_int.append(-1)
        elif feat_monot_dict[var] == 'Neutral':
            monotone_list_int.append(0)
            
    
    contri_list= []
    for var in feature_list:
        if var in feat_contri_dict:
            contri_list.append(feat_contri_dict[var]/100)
            
    # st.write(len(monotone_list_int), len(feature_list), len(contri_list))
    # st.write(monotone_list_int)
    # st.write(feature_list)
    
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
                              feature_contri = contri_list, monotone_constraints = monotone_list_int,\
                              colsample_bytree = colsample_bytree, learning_rate = learning_rate, \
                              subsample = subsample, bagging_freq = bagging_freq)

    
    X_train = data[feature_list]
    Y_train = data[target_var]
    
    
    model.fit(X_train, Y_train)
    
    
    data['predicted'] = model.predict(X_train)
    data['predicted'] = np.where(data['predicted']< 0, 0, data['predicted'])
    data['predicted_old'] = data['predicted']*data['factor']
    data['abs_error'] = np.abs(data[target_var_old] - data['predicted_old'])
    
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
    output_cols.append('predicted_old')
    
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



def get_base_contribution(data, model, feature_list, feat_mono_dict, feature = ''):
    
    temp = data.copy()
    
    if feature == '':
        for feat in feature_list:
            if feat_mono_dict[feat] == 'Positive':
                temp[feat] = temp[feat].min()
            
            if feat_mono_dict[feat] == 'Negitive':
                temp[feat] = temp[feat].max()
                
        data['base_predict'] = model.predict(temp[feature_list])
        data['base_predict'] = data['base_predict']*data['factor']
        data['incremental']  = data['predicted_old'] - data['base_predict']
        
    else:
        if feat_mono_dict[feature] == 'Positive':
            temp[feature] = temp[feature].min()
        
        if feat_mono_dict[feature] == 'Negitive':
            temp[feature] = temp[feature].max()
            
        data['predict_'+ feature] = model.predict(temp[feature_list])
        data['predict_'+ feature] = data['predict_'+ feature]*data['factor']
        data['inc_'+ feature]  = data['predicted_old'] - data['predict_'+ feature]

    
    return data

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')



def set_target_var_block(potential_target_var_names, column_list):
    
    target_var = ''
    
    for var in potential_target_var_names:
        if var in column_list:
            target_var = var
        
    if target_var == '':
        target_var = st.selectbox("Select Name of Target/Dependent Variable", column_list)
        
    return target_var


def set_categorical_cols_block(feature_list, categorical_cols):
    
    with st.expander("Select Categorical Columns: "):
        
        for col in feature_list:
            categorical_cols[col] = st.checkbox('Is '+col+' categorical', value=False)
            
    return categorical_cols


def side_bar_for_features_contri_mono(feature_list, feat_contri_dict, feat_mono_dict, exclude_from_model):
    
    with st.sidebar:
        
        for col in feature_list:
            with st.expander(col):
                feat_contri_dict[col] = st.slider("Contribution "+ col, min_value=0, max_value=200, value=100)
                        
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
                
    return feature_list, feat_contri_dict, feat_mono_dict, exclude_from_model


def scale_data_block(dataframe, granularity_list, target_var):
    
    with st.expander('Scale Data: '):
        do_scale = st.checkbox('Tick if data needs to be Scaled at Categorical Columns: ', value = False)
    
    # Scaling DF, for better results
    # st.write(dataframe.columns)
    if (len(granularity_list) > 0) & (do_scale):
        factor_df = scale_data(dataframe, granularity_list, target_var)
        data_rows = dataframe.shape[0]
        dataframe = dataframe.merge(factor_df, on= granularity_list, how = 'left')
        assert dataframe.shape[0] == data_rows
        dataframe[target_var + '_new'] = dataframe[target_var]/dataframe['factor']


    else:
        dataframe['factor'] = 1
        dataframe[target_var + '_new'] = dataframe[target_var]

        
    target_var_old = target_var
    target_var = target_var + '_new'
    
    return dataframe, target_var_old, target_var

def lightgbm_hyperparam_block():
    
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
            
    return tree_params

def base_vs_incremental_block(data, model, target_var_old, feature_list, feat_mono_dict, column_list, key_counter):
    
    with st.expander('Aggregated Base Vs Incremental: '):
        
        data = get_base_contribution(data, model, feature_list, feat_mono_dict)
        
        temp = data.copy()
        
        filter_cols = st.multiselect('Select Columns for filter', column_list, key = key_counter)
        key_counter = key_counter + 1
        

        for filter_col in filter_cols:
            type_of_col = str(data[filter_col].dtype)
            
            if ('int' in type_of_col) or ('float' in type_of_col):
                
                start_val, end_val = st.select_slider('Select Range for '+filter_col, \
                                                      range(0, int(data[filter_col].max()+1)), \
                                                      value = (0, int(data[filter_col].max()-1)), \
                                                      key = key_counter)
                key_counter = key_counter + 1
                    
                temp = temp[(temp[filter_col] >= start_val) & (temp[filter_col] <= end_val)]
            
            else:
                
                filter_values = st.multiselect('Select Values for '+ filter_col, data[filter_col].unique(), \
                                               key = key_counter)
                key_counter = key_counter + 1
                temp = temp[temp[filter_col].isin(filter_values)]
        

        x_axis = st.selectbox('Select Column for X Axis ', column_list, key = key_counter)
        key_counter = key_counter + 1

        chart_data = temp.groupby(x_axis)[[target_var_old, 'base_predict', 'predicted_old', 'incremental']].sum().reset_index()
        chart_data = chart_data.sort_values(by = x_axis, ascending = False)
        st.line_chart(chart_data, x = x_axis, y = [target_var_old, 'predicted_old', 'base_predict', 'incremental'])
        
    return data, key_counter

def actual_vs_predicted_block(data, granularity_list, key_counter, column_list, target_var_old):
    
    temp = data.copy()
    
    with st.expander('Actual Vs Predicted: '):
        st.write('')
        st.write('Filter for AVP Plot')
        
        # st.write(column_list)
        filter_cols = st.multiselect('Select Columns for filter', column_list, key = key_counter, )
        key_counter = key_counter + 1
        

        for filter_col in filter_cols:
            type_of_col = str(data[filter_col].dtype)
            
            if ('int' in type_of_col) or ('float' in type_of_col):
                
                start_val, end_val = st.select_slider('Select Range for '+filter_col, \
                                                      range(0, int(data[filter_col].max()+1)), \
                                                      value = (0, int(data[filter_col].max()-1)), \
                                                      key = key_counter)
                key_counter = key_counter + 1
                    
                temp = temp[(temp[filter_col] >= start_val) & (temp[filter_col] <= end_val)]
            
            else:
                
                filter_values = st.multiselect('Select Values for '+ filter_col, data[filter_col].unique(), \
                                               key = key_counter)
                key_counter = key_counter + 1
                temp = temp[temp[filter_col].isin(filter_values)]
        

        x_axis = st.selectbox('Select Column for X Axis ', column_list, key = key_counter)
        key_counter = key_counter + 1
        
        
        chart_data = temp.groupby(x_axis)[[target_var_old, 'predicted_old', 'abs_error']].sum().reset_index()
        chart_data = chart_data.sort_values(by = x_axis, ascending = False)
        st.line_chart(chart_data, x =x_axis, y = [target_var_old, 'predicted_old', 'abs_error'])
        
        
        return key_counter
    
    
def variable_contribution_block(data, model, feature_list, exclude_from_model, feat_mono_dict, 
                                granularity_list, target_var, target_var_old, key_counter):
    
    
    with st.expander('Feature wise Contribution: '):
        st.write('')
        
        simulation_variables = []
        
        # st.write(len(feature_list), len(granularity_list), len(exclude_from_model))
        
        for var in feature_list:
            if var not in granularity_list:
                if exclude_from_model[var]:
                    pass
                else:
                   simulation_variables.append(var) 
                            
        if target_var in simulation_variables:
            simulation_variables.remove(target_var)
        
        # st.write(len(simulation_variables))
        features_selected = st.multiselect(
            'Select Variable: ', list(set(simulation_variables)), key = key_counter)
        key_counter = key_counter + 1
        
        for feature_ in features_selected:
            data = get_base_contribution(data, model, feature_list, feat_mono_dict, feature = feature_)
        
        inc_col_list = [x for x in data.columns if 'inc_' in x]
        # inc_col_list = inc_col_list + [target_var_old]
        inc_df= data[inc_col_list].sum().reset_index()
        
        inc_df.columns = ['Feature', 'Contribution']
        inc_df['% Contribution'] = np.round((inc_df['Contribution']/data[target_var_old].sum())*100, 2)
        st.write(inc_df)
        # sim_chart_data = simulation_data.sum()
        # st.line_chart(sim_chart_data)
        
        return data, key_counter
    
    
def download_model_data_block(data):
    
    with st.expander('Download output file : '):
        
        csv = convert_df(data)
        
        st.download_button(
           "Press to Download",
           csv,
           "file.csv",
           "text/csv",
           key='download-csv'
        )
        
    return None