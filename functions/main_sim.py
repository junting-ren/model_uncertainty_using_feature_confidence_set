import os
import multiprocessing
import numpy as np
import pandas as pd
import time
from datetime import date
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import torch
# custom functions
from CS_sim_helper import sim_CS_wrapper, generate_sim_data,check_overfitting, transform_X_poly,transform_X
from models import NueralNet, logistic_regression
import argparse


today = date.today()
date = today.strftime("%d%m%Y")

if __name__ ==  '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='corollary_1')
    model_type = parser.parse_args().model
    df_total = pd.DataFrame()
    n_sim = 1000
    n_boot_v = [300]
    L_v = [0.9, 0.6]
    N_v = [100, 200, 400, 800]
    N_test_v = [500]
    p_v = [3, 6, 10]
    data_sim_func_v = [generate_sim_data]
    center_G_v = [True]
    center_pred_v = [False]
    use_true_contour_v = [False]
    df_result = []
    error_sd_v = [1]
    boundary_point_ind_v = [False]
    if model_type == "corollary_1":
        level_v = [0]
        L_v = [0.6]
        p_v = [3]
        N_v = [100, 200, 400, 800,1600, 3200, 6400, 12800]
        models_l_v = [[LinearRegression]]
        models_kwargs_l_v = [[{}]]
        data_kwargs_v = [{'uniform_range_x':[-2,2]}]
        boundary_point_ind_v = [True]
        folder_path = './linear_result_corrolary1'
    if model_type == "logistic_regression":
        level_v = [0.5]
        models_l_v = [[logistic_regression]]
        models_kwargs_l_v = [[{}]]
        data_kwargs_v = [{'uniform_range_x':[-2,2], "binary": True, "uniform_range_beta": [1,3]}]
        folder_path = './linear_result'
    elif model_type == "linear_regression":
        level_v = [0]
        models_l_v = [[LinearRegression]]
        models_kwargs_l_v = [[{}]]
        data_kwargs_v = [{'uniform_range_x':[-2,2]}]
        folder_path = './linear_result'
    elif model_type == "neural_net":
        n_sim = 500
        N_v = [100, 200, 400,800]
        level_v = [0]
        models_l_v = [[NueralNet]]
        models_kwargs_l_v = [[{'n_iter':200,'patience':100,'input_size': 1, 'h_sizes':[40,40]}]]
        p_v = [1] # overwrite 
        data_kwargs_v = [{'transform_func':transform_X, 'beta':np.array([1,2,3,4]), 'uniform_range_x':[-2,2]}]
        folder_path = './nonlinear_result'
    elif model_type == "xgboost":
        n_sim = 500
        N_v = [100, 200, 400,800]
        level_v = [0]
        models_l_v = [[XGBRegressor]]
        models_kwargs_l_v = [[{'n_estimators':10,'max_depth':6, 'subsample':0.2}]]
        p_v = [1] # overwrite 
        data_kwargs_v = [{'transform_func':transform_X, 'beta':np.array([1,2,3,4]), 'uniform_range_x':[-2,2]}]
        folder_path = './nonlinear_result'
    file_name = model_type+'_'+date+'.csv'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    file_name = os.path.join(folder_path, file_name)
    param_grid = {'L': L_v, 'level': level_v, 'models_l': models_l_v,'model_kwargs_l':models_kwargs_l_v,
                  'N': N_v, 'N_test': N_test_v, 'p': p_v , 'error_sd': error_sd_v, 
                  'data_sim_func':data_sim_func_v, 'data_kwargs':data_kwargs_v,'boundary_point_ind':boundary_point_ind_v, 
                  'center_G':center_G_v, 'center_pred':center_pred_v, 'use_true_contour': use_true_contour_v, 'n_boot':n_boot_v
                 }
    param_grid = ParameterGrid(param_grid)
    start = time.time()
    for param in param_grid:
        #sim_CS_wrapper(param)
        para_name_dict = {k:str(v) for k, v in param.items()}
        df = pd.DataFrame(para_name_dict, index = [0])
        ctx = torch.multiprocessing.get_context('spawn')
        pool_obj = ctx.Pool(processes=os.cpu_count())
        # pool_obj = multiprocessing.Pool()
        cur_para = (n_sim*(param,))
        df_ = pd.concat(pool_obj.map(sim_CS_wrapper, cur_para)).reset_index(drop = True)
        pool_obj.close()
        pool_obj.join()
        para_df = pd.DataFrame(np.repeat(df.values,df_.shape[0], axis=0))
        para_df.columns = df.columns
        df_ = pd.concat([para_df, df_], axis = 1)
        df_total = pd.concat([df_total,df_])
        df_total.to_csv(file_name, index = False)
        print(f"Done: {param}")
    end = time.time()
    print('Runtime of the program is ' + str(end -start) + ' seconds')